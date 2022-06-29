#!/bin/bash

#SBATCH -N1
#SBATCH --time 20:00:00
#SBATCH -o ./logs/evaluate/%u-%x-job%j.out
#SBATCH --job-name="evaluate"  
#SBATCH --ntasks=1
#SBATCH --array=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --mem=8GB



OUTPUT_PATH=../../output/performance
TRIALS_PATH=../../output/trained_models/vggdrop_norm_abn_final
DATA_PATH=../../data

mkdir -p $OUTPUT_PATH
singularity exec --nv --bind /q/PET-MBF:/q/PET-MBF /q/PET-MBF/dberman/xnn4rad.sif python3 ../evaluate_performance.py \
    --output_path=$OUTPUT_PATH \
    --trials_path=$TRIALS_PATH \
    --data_path=$DATA_PATH
