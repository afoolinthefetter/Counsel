import json
import torch
import numpy as np
import torch.multiprocessing as mp

from model.env import CloudEnv
from model.a3c import GCNActorCritic
from model.synthetic import set_slo
import model.core as core

import os
import sys
import datetime
import argparse


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        print(self.ptr, self.max_size)
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
    

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.tensor([0])  # Ensure state['step'] is a singleton tensor
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

                
# class SharedAdam(torch.optim.Adam):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
#             weight_decay=0):
#         super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
#                 weight_decay=weight_decay)

#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['exp_avg'] = torch.zeros_like(p.data)
#                 state['exp_avg_sq'] = torch.zeros_like(p.data)

#                 state['exp_avg'].share_memory_()
#                 state['exp_avg_sq'].share_memory_()


class Agent(mp.Process):
    def __init__(self, cloudConf, input_dim, n_actions, global_ac, gamma, pi_optim, vf_optim, name, global_ep_index, epochs):
        super(Agent, self).__init__()
        #creating a new actor critic model for each worker
        self.local_actor_critic = GCNActorCritic(input_dim, n_actions)
        self.global_ac = global_ac
        
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_index
        
        #each worker class will be making their own environment
        self.env = CloudEnv(cloudConf[0]+"/"+str(name), cloudConf[1], cloudConf[2], cloudConf[3], cloudConf[4], cloudConf[5], cloudConf[6], cloudConf[7])

        self.pi_optim = pi_optim
        self.vf_optim = vf_optim

        self.epochs = epochs
        self.steps_in_epoch = cloudConf[1]


    # def run(self):
    #     self.local_actor_critic.load_state_dict(
    #                         self.global_ac.state_dict())
    #     t_step = 1

    #     self.buffer = PPOBuffer(self.env.observation_space.shape, self.env.action_space.shape, self.steps_in_epoch)
    #     while self.episode_idx.value < self.epochs:
    #         done = False
    #         o,m = self.env.reset()
           
            
    #         score = 0
    #         a = 0
    #         while not done:
    #             print("a = ", a)
    #             a += 1
    #             #passing in the observation to make the action choice
    #             action, value, logp = self.local_actor_critic.step(torch.as_tensor(o), torch.as_tensor(m))
    #             obs, mask, reward, done, info = self.env.step(action)
    #             score += reward
    #             self.buffer.store(obs, action, reward, value, logp)
    #             print("steps done? = ", t_step % self.steps_in_epoch)
    #             if t_step % self.steps_in_epoch == 0:
    #                 if done:
    #                     print("-->>>>>>> DONEEEEE? = ",done)
    #             # if 0:
    #                 loss_pi, pi_info = self.local_actor_critic.compute_loss_pi(self.buffer.get(), 0.2)
                    
    #                 print("the loss is = ",loss_pi)
    #                 #calling the optimizer to update the weights
    #                 # self.pi_optim.zero_grad()
    #                 # self.vf_optim.zero_grad()
    #                 #doing backward on the lossed
    #                 # critic_loss = critic_loss.mean()
    #                 # actor_loss = actor_loss.mean()

    #                 # critic_loss.backward()
    #                 # actor_loss.backward()

    #                 #taking the optimizer/s step (if there are two optimizers)
    #                 # self.pi_optim.step()
    #                 # self.vf_optim.step()

    #                 #we set the gradients of the local model to the global model
                    
    #                 for local_param, global_param in zip(
    #                         self.local_actor_critic.parameters(),
    #                         self.global_ac.parameters()):
    #                     global_param._grad = local_param.grad
                    
    #                 #taking the optimizer/s step (if there are two optimizers)
    #                 #load the parameters of the global model to the local model
    #                 self.local_actor_critic.load_state_dict(
    #                         self.global_ac.state_dict())
    #                 self.local_actor_critic.clear_memory()
    #             t_step += 1
    #             o = obs
    #             m = mask
    #             with self.episode_idx.get_lock():
    #                 self.episode_idx.value += 1
    #             print(self.name, 'Ep:', self.episode_idx.value, '| Ep_r: ', round(score, 2))
    
    def run(self):
        self.local_actor_critic.load_state_dict(
                            self.global_ac.state_dict())
        t_step = 1
        
        while self.episode_idx.value < self.epochs:
            done = False
            o,m = self.env.reset()
            
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                #passing in the observation to make the action choice
                action, value, logp = self.local_actor_critic.step(torch.as_tensor(o), torch.as_tensor(m))
                obs, mask, reward, done, info = self.env.step(action)
                # print(obs.shape, mask.shape, type(reward), type(done), type(info), action)
                score += reward
                self.local_actor_critic.remember(obs, action, reward)

                if t_step % self.steps_in_epoch == 0 or done:
                # if 0:
                    critic_loss, actor_loss = self.local_actor_critic.calculate_loss(done)
                    print(critic_loss, actor_loss)
                    #calling the optimizer to update the weights
                    self.pi_optim.zero_grad()
                    self.vf_optim.zero_grad()
                    
                    #doing backward on the lossed
                    critic_loss = critic_loss.mean()
                    actor_loss = actor_loss.mean()
                    
                    critic_loss.backward(retain_graph=True)
                    actor_loss.backward(retain_graph=True)

                    #taking the optimizer/s step (if there are two optimizers)
                    # self.pi_optim.step()

                    #we set the gradients of the local model to the global model
                    
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_ac.parameters()):
                        global_param._grad = local_param.grad
                    
                    self.vf_optim.step()
                    self.pi_optim.step()
                    #taking the optimizer/s step (if there are two optimizers)
                    #load the parameters of the global model to the local model
                    self.local_actor_critic.load_state_dict(
                            self.global_ac.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step += 1
                o = obs
                m = mask
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'Ep:', self.episode_idx.value, '| Ep_r: ', round(score, 2))


if __name__ == '__main__':
    hp_file = "model/configs/hyperparams.json"
    hp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), hp_file)
    hyperparams = json.load(open(hp_file, "r"))


    parser = argparse.ArgumentParser()
    parser.add_argument('-n',"--exp_name", help="exp name", type=str)
    parser.add_argument('-e',"--epochs", help="epochs", type=int)
    parser.add_argument('-c',"--clip_ratio", help="clip ratio (epsilon)", type=float)
    parser.add_argument('-k', "--knob", help="knob", type=float)
    parser.add_argument('-ncp', "--ncomp", help="NFV Components", type=int)
    parser.add_argument('-ncf', "--nconf", help="VM Instances", type=int)
    args = parser.parse_args()

    if args.exp_name:
        hyperparams["exp_name"] = args.exp_name
    if args.clip_ratio:
        hyperparams["clip_ratio"] = float(args.clip_ratio)
    if args.knob:
        hyperparams["knob"] = float(args.knob)
    if args.ncomp:
        hyperparams["ncomp"] = int(args.ncomp)
    if args.nconf:
        hyperparams["nconf"] = int(args.nconf)
    if args.epochs:
        hyperparams["epochs"] = int(args.epochs)

    slo = int(np.exp(np.random.randint(240,840)/100))
    freq = int(1e6 / np.random.randint(int(slo*0.8), int(slo*1.2)))
    knob = hyperparams["knob"] # For over, under and near provisioning
    print(f"SLO: {slo}, Freq: {freq}, Knob: {knob}")
    set_slo(slo, freq, knob)

    budget = hyperparams["budget"]
    overrun_lim = hyperparams["budget_relax"]
    mode = hyperparams["mode"]
    threads = hyperparams["threads"]
    nconf = hyperparams["nconf"]
    ncomp = hyperparams["ncomp"]
    algo = hyperparams["algo"]
    steps_per_epoch = hyperparams["steps_per_epoch"]
    pi_lr = hyperparams["pi_lr"]
    vf_lr = hyperparams["vf_lr"]


    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

    #pass the following to the cloud environment st each agent will have the same thing
    #we need to have different environments for each agent but with same configuration
    #append the log directory with the eorker's identity inside Agent class
    cloudConf = ["a3c/logs"+time_string, steps_per_epoch, budget, slo, overrun_lim, mode, nconf, ncomp]
    epochs = hyperparams["epochs"]


    lr = 1e-4

    #making a dummy env so that we can use its dimensions for the global ac network
    env = CloudEnv(cloudConf[0]+"/"+str(0), cloudConf[1], cloudConf[2], cloudConf[3], cloudConf[4], cloudConf[5], cloudConf[6], cloudConf[7])
    
    input_dims = env.observation_space
    n_actions = env.action_space

    
    global_actor_critic = GCNActorCritic(env.observation_space, env.action_space)
    #the global actor critic needed the env to setup the nn only
    env.close()
    
    #share the global network's memory
    global_actor_critic.share_memory()
    
    #have to figure out this optimizer thing
    #optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
    #                    betas=(0.92, 0.999))
    pi_optim = SharedAdam(global_actor_critic.pi.parameters(), lr=pi_lr)
    v_optim = SharedAdam(global_actor_critic.v.parameters(), lr=vf_lr)
    
    global_ep = mp.Value('i', 0)
    sys.stdout.flush()

    workers = [Agent(cloudConf=cloudConf,
                    global_ac=global_actor_critic,
                    pi_optim=pi_optim,
                    vf_optim=v_optim,
                    input_dim=input_dims,
                    n_actions=n_actions,
                    gamma=0.99,
                    name=i,
                    global_ep_index=global_ep,
                    epochs = epochs) for i in range(mp.cpu_count())]
    
    [w.start() for w in workers]
    [w.join() for w in workers]