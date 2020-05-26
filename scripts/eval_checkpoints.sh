# !/bin/bash

set -e

SEQ='samui_walking_street1'

CKPT_PATH="/home/cfan/pytorch_segmentation/results/${SEQ}_ckpts/online_distillation_dataset.sequence=${SEQ},online_train.bg_weight=1,online_train.checkpoint_good_model=true,online_train.fg_weight=10,online_train.max_frames=5000,online_train.max_updates=16"

#1696 3136 3424
for f in 3664 3920 4048 4280 4744 4960
do
    CUDA_VISIBLE_DEVICES=1 python online_distillation.py  \
        online_train.max_frames=5000 \
        exp.id=${SEQ}_ckpts_eval_$f \
        online_train.max_updates=16 \
        dataset.sequence=$SEQ \
        dataset.start_frame=$f \
        model.pretrained_ckpt=$CKPT_PATH/frame_$f.pth \
        model.ignored_vars=[] \
        online_train.online_train=false
done