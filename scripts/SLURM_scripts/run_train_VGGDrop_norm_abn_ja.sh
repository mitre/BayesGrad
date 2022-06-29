#!/bin/bash

#SBATCH -N1
#SBATCH --time 3:30:00
#SBATCH --job-name=train_PET_VGGish
#SBATCH -o ./logs/norm_abn/%u-%x-job%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --array=1-20
#SBATCH --mem=5GB
#SBATCH --account=MBF-AI


#     This script runs model training 20 times so that we can account for differences
#     in individual trained models in our results for the model trained on the 
#     normal/abnormal detection problem.


# echo "LR $LR REG $REG DROPOUT_RATE $DROPOUT_RATE NUM_BASE_FILTERS $NUM_BASE_FILTERS STD_SCALER $STD_SCALER" 
NUM_BASE_FILTERS=128
STD_SCALER=0.0
LR=0.0001
EPOCHS=20
REG=0.0
DROPOUT_RATE=0.3
SHIFT=0.05
ROTATE=180
echo "LR $LR REG $REG DROPOUT_RATE $DROPOUT_RATE NUM_BASE_FILTERS $NUM_BASE_FILTERS STD_SCALER $STD_SCALER SHIFT $SHIFT ROTATE $ROTATE"
echo



# Print out Job Details
echo "Job ID: "$SLURM_JOB_ID
echo "Job Array ID: "$SLURM_ARRAY_JOB_ID
echo "Job Array Task: "$SLURM_ARRAY_TASK_ID"/"$SLURM_ARRAY_TASK_MAX
echo "Job Account: "$SLURM_JOB_ACCOUNT
echo "Hosts: "$SLURM_NODELIST
echo "Task ID: " $SLURM_ARRAY_TASK_ID
echo "------------"

mkdir -p ../../output/trained_models/vggdrop_norm_abn_final/
# PET 
singularity exec --nv --bind /q/PET-MBF:/q/PET-MBF ../../Singularity/xnn4rad.sif python3 ../train_VGGDrop_norm_abn.py \
    --data_path=../../data \
    --output_dir=../../output/trained_models/vggdrop_norm_abn_final/ \
    --lr=$LR \
    --epoch=$EPOCHS \
    --reg=$REG \
    --dropout_rate=$DROPOUT_RATE \
    --num_base_filters=$NUM_BASE_FILTERS \
    --preproc_std_scaler=$STD_SCALER \
    --vary_std_scaler=False \
    --preproc_rotation_range=$ROTATE \
    --preproc_width_shift_range=$SHIFT \
    --preproc_height_shift_range=$SHIFT \
    --problem=abnormal \
    --suffix="${SLURM_ARRAY_TASK_ID}" \
    --stress \
    --rest

