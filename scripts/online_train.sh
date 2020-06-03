# !/bin/bash

GPU=0
#SEQ=soccer1
#SEQ_ID=0
#START_FRAME=0
#MAX_FRAMES=15000

SEQ=tennis1
SEQ_ID=0
START_FRAME=0
MAX_FRAMES=15000

CUDA_VISIBLE_DEVICES=$GPU python online_distillation.py \
    exp.id=${SEQ}_on_seq0 \
    dataset.sequence=$SEQ \
    dataset.sequence_id=$SEQ_ID \
    online_train.max_frames=$MAX_FRAMES \
    dataset.start_frame=$START_FRAME