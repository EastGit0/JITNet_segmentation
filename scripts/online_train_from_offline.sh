# !/bin/bash

GPU=2

#SEQ=driving1
#SEQ_ID=1
##CKPT=/home/cfan/pytorch_segmentation/results/driving1_off_seq0_adam_scheduler/offline_distillation_dataset.sequence=driving1/epoch_29.pth.pth
#CKPT=/home/cfan/pytorch_segmentation/results/driving1_off_seq0_on_seq0/offline_distillation_dataset.max_frames=15000,dataset.sequence=driving1,dataset.sequence_id=0,model=jitnet_driving1_on_pretrained,online_train.optimizer.lr=0.05,online_train.optimizer.momentum=0.99,online_train.training_stride=1/epoch_29.pth.pth
#START_FRAME=0
#MAX_FRAMES=15000

#SEQ=table_tennis1
#SEQ_ID=1
##CKPT=/home/cfan/pytorch_segmentation/results/table_tennis1_off_seq0/offline_distillation_dataset.sequence=table_tennis1/epoch_29.pth.pth
##CKPT=/home/cfan/pytorch_segmentation/results/table_tennis1_off_seq0_sweep/5/epoch_29.pth.pth
#CKPT=/home/cfan/pytorch_segmentation/results/table_tennis1_off_seq0_on_seq0/offline_distillation_dataset.max_frames=15000,dataset.sequence=table_tennis1,dataset.sequence_id=0,model=jitnet_table_tennis1_on_pretrained,online_train.optimizer.lr=0.05,online_train.optimizer.momentum=0.99,online_train.training_stride=1/epoch_29.pth.pth
#START_FRAME=0
#MAX_FRAMES=15000

SEQ=samui_walking_street1
SEQ_ID=0
#CKPT=/home/cfan/pytorch_segmentation/results/samui_walking_street1_off_seq0/offline_distillation_dataset.sequence=samui_walking_street1,dataset.sequence_id=1,online_train.max_frames=15000/epoch_29.pth.pth
CKPT=/home/cfan/pytorch_segmentation/results/samui_walking_street1_off_seq0_sweep/7/epoch_29.pth.pth
#CKPT=/home/cfan/pytorch_segmentation/results/samui_wakling_street1_off_seq0_on_seq0/offline_distillation_dataset.max_frames=15000,dataset.sequence=samui_walking_street1,dataset.sequence_id=1,model=jitnet_samui_walking_street1_on_pretrained,online_train.optimizer.lr=0.05,online_train.optimizer.momentum=0.99,online_train.training_stride=1/epoch_29.pth.pth
START_FRAME=15000
MAX_FRAMES=15000

#SEQ=soccer1
#SEQ_ID=1
##CKPT=/home/cfan/pytorch_segmentation/results/soccer1_off_seq0/offline_distillation_dataset.sequence=soccer1,dataset.sequence_id=2,online_train.max_frames=15000/epoch_29.pth.pth
##CKPT=/home/cfan/pytorch_segmentation/results/soccer1_off_seq0_sweep/5/epoch_29.pth.pth
#CKPT=/home/cfan/pytorch_segmentation/results/soccer1_off_seq0_on_seq0/offline_distillation_dataset.max_frames=15000,dataset.sequence=soccer1,dataset.sequence_id=2,model=jitnet_soccer1_on_pretrained,online_train.optimizer.lr=0.05,online_train.optimizer.momentum=0.99,online_train.training_stride=1/epoch_29.pth.pth
#START_FRAME=0
#MAX_FRAMES=15000

#SEQ=tennis1
#SEQ_ID=1
##CKPT=/home/cfan/pytorch_segmentation/results/tennis1_off_seq0/offline_distillation_dataset.sequence=tennis1,dataset.sequence_id=0,online_train.max_frames=15000/epoch_29.pth.pth
##CKPT=/home/cfan/pytorch_segmentation/results/tennis1_off_seq0_sweep/5/epoch_29.pth.pth
#CKPT=/home/cfan/pytorch_segmentation/results/tennis1_off_seq0_on_seq0/offline_distillation_dataset.max_frames=15000,dataset.sequence=tennis1,dataset.sequence_id=0,model=jitnet_tennis1_on_pretrained,online_train.optimizer.lr=0.05,online_train.optimizer.momentum=0.99,online_train.training_stride=1/epoch_29.pth.pth
#START_FRAME=0
#MAX_FRAMES=15000


CUDA_VISIBLE_DEVICES=$GPU python online_distillation.py \
    exp.id=${SEQ}_on_seq1_off_seq0_sweep7 \
    dataset.sequence=$SEQ \
    dataset.sequence_id=$SEQ_ID \
    model.pretrained_ckpt=$CKPT \
    model.ignored_vars=[] \
    online_train.max_frames=$MAX_FRAMES \
    dataset.start_frame=$START_FRAME online_train.online_train=false