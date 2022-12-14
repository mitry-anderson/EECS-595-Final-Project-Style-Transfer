#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=00-00:50:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=90G
#SBATCH --job-name=mitryand_EECS595_project_train
#SBATCH --account=eecs595f22_class
#SBATCH --output=/home/%u/EECS-595-Final-Project-Style-Transfer/outputs/%x-%j.log
#SBATCH --mail-user=mitryand@umich.edu

# set up job
module load python/3.10.4 
module load cuda

source venv/bin/activate

# run job
python train_model.py --train_all_checkpoint=True --full_dataset=True --batch_size=16 --num_epochs=28 --model_name="brown_autoencoder_15_1670965860.9319289" --classifier_name="brown_latent_classifier_15_1670965866.0978472.torch"
