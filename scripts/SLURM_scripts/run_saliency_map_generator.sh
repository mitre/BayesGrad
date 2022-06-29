#!/bin/bash

#SBATCH -N1
#SBATCH --time 20:00:00
#SBATCH -o ./logs/saliency_map_gen/%u-%x-job%j.out
#SBATCH --job-name="saliency_map_gen"  
#SBATCH --ntasks=1
#SBATCH --array=1-40
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --mem=8GB


JOB_ARGS="$(sed -n ${SLURM_ARRAY_TASK_ID},${SLURM_ARRAY_TASK_ID}p ../arg_files/saliency_map_generator_20_trials_args.txt )"
set -- $JOB_ARGS
MODEL_PATH=$1
MODEL_ALIAS=$2
SPLIT=$3
S_METHOD=BG_VAR
PROBABILITIES=True

OUTDIR=../../output/smaps_20_trials
DATA_PATH=../../data

mkdir -p $OUTDIR
singularity exec --nv --bind /q/PET-MBF:/q/PET-MBF ../../singularity/xnn4rad.sif python3 ../saliency_map_generator.py \
    --data_path=$DATA_PATH \
    --output_dir=$OUTDIR \
    --model_alias=$MODEL_ALIAS \
    --model_path=$MODEL_PATH  \
    --split=$SPLIT \
    --smethod=$S_METHOD \
    --probabilities=$PROBABILITIES
