#!/bin/bash

begin_id=0
export CUDA_VISIBLE_DEVICES=0

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
 