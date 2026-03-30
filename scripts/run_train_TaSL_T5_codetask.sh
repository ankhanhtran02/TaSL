#!/bin/bash

begin_id=0
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=wandb_v1_PSYwiv7NgiYBagpUTCO2kTWcgyE
export WANDB_ENTITY=caovdongg18-cl4code
export WANDB_PROJECT=tasl_t5_codetask
for ((ORDER=$begin_id; ORDER<7; ORDER++))
do

    python finetune_continualDST_T5_codetask.py \
        --model_path Salesforce/codet5p-770m \
        --num_epochs 3 \
        --service_begin_id=${ORDER} \
        --train_batch_size 16 \
        --eval_batch_size 16 \

    wait

    python skill_consolidation_T5_codetask.py \
        --service_begin_id=${ORDER} \
        --checkpoint_name codet5p-770m-CodeTask-CL \
        --ipt_file_name codet5p-770m_Importance_Score \
        --model_name codet5p-770m \

done
 