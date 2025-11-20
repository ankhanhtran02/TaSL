#!/bin/bash


begin_id=0


for ((ORDER=$begin_id; ORDER<4; ORDER++))
do

    python finetune_continualDST_T5_codetask.py \
        --num_epochs 5 \
        --train_size -1 \
        --val_size 100 \
        --task_list CONCODE CodeTrans CodeSearchNet BFP \
        --service_begin_id=${ORDER} \
        --train_batch_size 8 \
        --eval_batch_size 8 \

    wait

    python skill_consolidation_T5_codetask.py \
        --service_begin_id=${ORDER} \

done
