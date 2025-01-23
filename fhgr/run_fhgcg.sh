#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate llm-adv

DEVICE=0
    
TEMPLATE=MODEL_NAME

FINETUNED_MODEL_PATH=PATH/TO/LORA/ADAPTERS
BASE_MODEL_PATH=PATH/TO/MODEL
JUDGE_PATH=PATH/TO/JUDGE
SAVE_PATH=PATH/TO/SAVE/FOLDER

# Starting index of dataset slice.
START=$1
# Ending index of dataset slice.
END=$2
# Index to resume if run got interrupted.
RESUME_IDX=$3
# Batch size for forward pass during GCG and GR.
FORWARD_BATCH_SIZE=$4
# Starting model checkpoint for binary search
EPOCHS=$5

# Attack iters (for a given prompt)
STEPS=200
DATASET=adv_harm_mistral_simple_goals

TAG=BASE_${DATASET}_epoch_${EPOCHS}_steps_${STEPS}


CUDA_VISIBLE_DEVICES=${DEVICE} python fh-gcg-binsearch.py \
    --template ${TEMPLATE} \
    --finetuned_model_path ${FINETUNED_MODEL_PATH} \
    --base_model_path ${BASE_MODEL_PATH} \
    --judge_path ${JUDGE_PATH} \
    --save_path ${SAVE_PATH} \
    --resume_idx ${RESUME_IDX} \
    --dataset ${DATASET} \
    --dataset_start_idx ${START} \
    --dataset_end_idx ${END} \
    --fbs ${FORWARD_BATCH_SIZE} \
    --tag ${TAG} \
    --epochs ${EPOCHS} \
    --steps ${STEPS} \
    --no_sys_msg \
    # --use_rand_grads \
