# !/bin/bash

GPU_ID=7

#SEQ=soccer1
#SEQ_ID=2

SEQ=tennis1
SEQ_ID=0

CUDA_VISIBLE_DEVICES=0 python offline_distillation.py \
    dataset.sequence=$SEQ \
    exp.id=${SEQ}_off_seq0 \
    online_train.max_frames=15000 \
    dataset.sequence_id=${SEQ_ID}