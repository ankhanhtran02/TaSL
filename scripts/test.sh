#!/bin/bash

begin_id=0


for ((ORDER=$begin_id; ORDER<4; ORDER++))
do

    python finetune_continualDST_T5_codetask.py \
        --model_path Salesforce/codet5-small \
        --num_epochs 2 \
        --train_size 10 \
        --val_size 10 \
        --task_list CONCODE CodeTrans CodeSearchNet BFP \
        --service_begin_id=${ORDER} \
        --train_batch_size 1 \
        --eval_batch_size 1 \

    wait

    python skill_consolidation_T5_codetask.py \
        --service_begin_id=${ORDER} \
        --checkpoint_name codet5-small-CodeTask-CL \
        --ipt_file_name codet5-small_Importance_Score \
        --model_name codet5-small \

done
