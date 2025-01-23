#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate llm-adv

DEVICE=0
TEMPLATE=MODEL_NAME

# Starting index of dataset slice.
START=$1
# Ending index of dataset slice.
END=$2
# Index to resume if run got interrupted.
RESUME_IDX=$3
# Batch size split for forward pass. Size is 512/{BATCH_SPLIT}.
BATCH_SPLIT=$4
# Starting model for checkpoint for binary search.
EPOCHS=$5

STEPS=200
TAG=TAG
DATASET=adv_harm_mistral_simple_goals

FINETUNED_MODEL_PATH=PATH/TO/LORA/ADAPTERS
BASE_MODEL_PATH=PATH/TO/MODEL
JUDGE_PATH=PATH/TO/JUDGE
SAVE_PATH=PATH/TO/SAVE/FOLDER

CUDA_VISIBLE_DEVICES=${DEVICE} python3 fh-autodan.py \
    --template ${TEMPLATE} \
    --finetuned_model_path ${FINETUNED_MODEL_PATH} \
    --base_model_path ${BASE_MODEL_PATH} \
    --judge_path ${JUDGE_PATH} \
    --tag ${TAG} \
    --num_steps ${STEPS} \
    --dataset ${DATASET} \
    --batch_split ${BATCH_SPLIT} \
    --save_path ${SAVE_PATH} \
    --start ${START} \
    --end ${END} \
    --dataset_start_idx ${START} \
    --dataset_end_idx ${END} \
    --log_interval 5 \
    --epochs ${EPOCHS} \
