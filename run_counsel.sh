#!/bin/bash
#SBATCH --job-name=loadserver	   # Job name
#SBATCH --output=counsel.txt
#SBATCH --error=counsel_error.txt
#SBATCH -N 1
#SBATCH -p small
#SBATCH --cpus-per-task=48

# module load your_module_name_here
module load DL/conda-python/3.7
module load openmpi/4.1.4
module load gcc/10.2
module load gnu8/8.3.0


eval "$(conda shell.bash hook)"
conda activate venv

cd /home/aman.gupta/Thesis/Counsel/load-gen/
# run fastapi server
uvicorn loadserver:app --host 0.0.0.0 --port 8000 --log-level warning &

cd /home/aman.gupta/Thesis/Counsel


# Under, Over and Expert Provisioning
python3 traina3c.py -n std -k 0.05 -e 1000
python3 traina3c.py -n op -k 0.03 -e 1000
python3 traina3c.py -n up -k 0.07 -e 1000

# Varying Chain Length and VM Configuration Counts
python3 traina3c.py -n std-f5-c3 -ncf 5 -ncp 3 -e 1000
# python3 traina3c.py -n std-f10-c3 -ncf 10 -ncp 3 -e 1000
# python3 traina3c.py -n std-f25-c3 -ncf 25 -ncp 3 -e 1000
# python3 traina3c.py -n std-f50-c3 -ncf 50 -ncp 3 -e 1000
# python3 traina3c.py -n std-f100-c3 -ncf 100 -ncp 3 -e 1000
# python3 traina3c.py -n std-f5-c5 -ncf 5 -ncp 5 -e 1000
# python3 traina3c.py -n std-f5-c10 -ncf 5 -ncp 10 -e 1000
# python3 traina3c.py -n std-f5-c20 -ncf 5 -ncp 20 -e 1000

# # hyperparameter evaluation
# python3 traina3c.py -n std01 -c 0.1 -k 0.05 -e 500
# python3 traina3c.py -n std02 -c 0.2 -k 0.05 -e 500
# python3 traina3c.py -n std03 -c 0.3 -k 0.05 -e 500
# python3 traina3c.py -n op01 -c 0.1 -k 0.02 -e 500
# python3 traina3c.py -n op02 -c 0.2 -k 0.02 -e 500
# python3 traina3c.py -n op03 -c 0.3 -k 0.02 -e 500
# python3 traina3c.py -n up01 -c 0.1 -k 0.07 -e 500
# python3 traina3c.py -n up02 -c 0.2 -k 0.07 -e 500
# python3 traina3c.py -n up03 -c 0.3 -k 0.07 -e 500