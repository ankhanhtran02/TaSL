#!/bin/bash


begin_id=2


for ((ORDER=$begin_id; ORDER<4; ORDER++))
do

    python finetune_continualDST_T5_codetask.py \
        --num_epochs 1 \
        --train_size 10 \
        --val_size 10 \
        --task_list CONCODE CodeTrans CodeSearchNet BFP \
        --service_begin_id=${ORDER} \

    wait

    python skill_consolidation_T5_codetask.py \
        --service_begin_id=${ORDER} \

done
