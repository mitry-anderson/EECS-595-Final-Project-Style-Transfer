#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --time=00-00:50:00
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
python train_model.py --train_all=True --full_dataset=True --batch_size=16 --num_epochs=32
