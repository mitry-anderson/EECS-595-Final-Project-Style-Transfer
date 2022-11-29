#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --time=00-02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB
#SBATCH --job-name=mitryand_EECS595_project_train
#SBATCH --account=eecs595f22_class
#SBATCH --output=/home/%u/EECS-595-Final-Project-Style-Transfer/outputs/%x-%j.log
#SBATCH --mail-user=mitryand@umich.edu

# set up job
module load python/3.10.4 
module load cuda

source venv/bin/activate

# run job
python train_model.py --train=True --batch_size=32 --num_epochs=3
