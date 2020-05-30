import os
import time
import pdb

import cv2
import hydra
import numpy as np
import logging
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from fvcore.common.checkpoint import Checkpointer

from analytics import full_segment_iou
from models import JITNet
from dataloaders.maskrcnn_stream import (batch_segmentation_masks,
                                         visualize_masks,
                                         MaskRCNNSequenceStream)
import dataloaders.lvs_dataset as lvs_dataset
from online_distillation import (load_model,
                                 configure_optimizer,
                                 calculate_class_iou)

log = logging.getLogger(__name__)

def train(cfg):
    torch.manual_seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)

    # Init model, optimizer, loss, video stream
    class_groups = lvs_dataset.sequence_to_class_groups_stable[cfg.dataset.sequence]
    class_groups = [ [lvs_dataset.detectron_classes.index(c) for c in g] \
                     for g in class_groups]
    num_classes = len(class_groups) + 1
    log.info(f'Number of class {num_classes}')

    dataset = lvs_dataset.LVSDataset(cfg.dataset.data_dir, cfg.dataset.sequence,
                                     '000',
                                     start_frame=0,
                                     max_frames=0,
                                     stride=cfg.online_train.training_stride)

    device = torch.device('cuda')
    model, _ = load_model(cfg.model, num_classes)
    model.to(device)
    optimizer = configure_optimizer(cfg.online_train.optimizer, model)
    scheduler = None
    if cfg.online_train.scheduler is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         cfg.online_train.scheduler.milestones,
                                                         cfg.online_train.scheduler.gamma)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion.to(device)

    start_epoch = 0
    checkpointer = Checkpointer(model, save_dir='./', optimizer=optimizer)
    #states = checkpointer.resume_or_load(None, resume=True)
    #if 'model' in states:
    #    model.load_state_dict(states['model'])
    #if 'optimizer' in states:
    #    optimizer.load_state_dict(states['optimizer'])
    #if 'epoch' in states:
    #    start_epoch = states['epoch'] + 1

    train_cfg = cfg.online_train
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=train_cfg.batch_size,
                                             shuffle=True,
                                             num_workers=4)
    writer = SummaryWriter(log_dir='./', flush_secs=30)

    for epoch in range(start_epoch, train_cfg.epoch):
        pbar = tqdm(total=len(dataset) // train_cfg.batch_size + 1)
        for batch_idx, (frames, labels, label_weights) in enumerate(dataloader):
            optimizer.zero_grad()

            frames = frames.to(device)
            labels = labels.to(device)
            label_weights = label_weights.to(device)

            logits = model(frames)
            loss = criterion(logits, labels)
            loss_weights = torch.ones_like(label_weights) * train_cfg.fg_weight
            bg_mask = label_weights == 0
            loss_weights.masked_fill_(bg_mask, train_cfg.bg_weight)
            loss = (loss * loss_weights).mean()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, preds = torch.max(logits, dim=1)
                tp, fp, fn, cls_scores = \
                    calculate_class_iou(preds, labels, num_classes)

            step = epoch * len(dataset) + batch_idx * train_cfg.batch_size
            if batch_idx % 10 == 0:
                writer.add_scalar('train/loss', loss, step)
                writer.add_scalar('train/bg_iou', cls_scores[0], step)
                writer.add_scalar('train/fg_iou', cls_scores[1:].mean(), step)

            pbar.update(1)
            pbar.set_description(f'loss: {loss:.3f} mIoU: {cls_scores[1:].mean():.3f}')

        checkpointer.save(f'epoch_{epoch}.pth', epoch=epoch)

        if scheduler:
            scheduler.step()

@hydra.main(config_path='conf/config_offline.yaml')
def main(cfg):
    print(cfg.pretty())
    train(cfg)

if __name__=='__main__':
    main()
