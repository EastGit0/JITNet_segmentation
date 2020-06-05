# !/bin/bash

GPU=4
SEQ=samui_walking_street1
SEQ_ID=0
CKPT=/home/cfan/pytorch_segmentation/results/samui_walking_street1_on_seq0/online_distillation_dataset.sequence=samui_walking_street1,dataset.sequence_id=0,dataset.start_frame=0,online_train.max_frames=15000/final.pth
START_FRAME=15000
MAX_FRAMES=3000

#SEQ=soccer1
#SEQ_ID=1
#CKPT=/home/cfan/pytorch_segmentation/results/soccer1_on_seq0/online_distillation_dataset.sequence=soccer1,dataset.sequence_id=0,dataset.start_frame=0,online_train.max_frames=15000/final.pth
#START_FRAME=0
#MAX_FRAMES=15000

#SEQ=tennis1
#SEQ_ID=1
#CKPT=/home/cfan/pytorch_segmentation/results/tennis1_on_seq0/online_distillation_dataset.sequence=tennis1,dataset.sequence_id=0,dataset.start_frame=0,online_train.max_frames=15000/final.pth
#START_FRAME=0
#MAX_FRAMES=15000

CUDA_VISIBLE_DEVICES=$GPU python online_distillation.py \
    exp.id=${SEQ}_on_seq1_on_seq0 \
    dataset.sequence=$SEQ \
    dataset.sequence_id=$SEQ_ID \
    model.pretrained_ckpt=$CKPT \
    online_train.max_frames=$MAX_FRAMES \
    dataset.start_frame=$START_FRAME \
    online_train.resume_online=true \
    online_train.training_strides=[8] online_train.video=video.avi