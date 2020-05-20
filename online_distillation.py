import time

import cv2
import hydra
import numpy as np
import logging
import torch
import torch.nn.functional as F
from torchvision import transforms

from models import JITNet
from dataloaders import MaskRCNNStream
from dataloaders.maskrcnn_stream import batch_segmentation_masks

log = logging.getLogger(__name__)

def get_class_groups(train_cfg):
    people_cls = [1]
    ball_cls = [33]
    twowheeler_cls = [2, 4]
    vehicle_cls = [3, 6, 7, 8]

    #(40, 'bottle')
    #(41, 'wine glass')
    #(42, 'cup')
    #(43, 'fork')
    #(44, 'knife')
    #(45, 'spoon')
    #(46, 'bowl')

    utensils_cls = [40, 41, 42, 43, 44, 45, 46]

    #(14, 'bench')
    #(57, 'chair')
    #(58, 'couch')
    #(61, 'dining table')
    furniture_cls = [14, 57, 58, 61]

    if train_cfg.fine_classes:
        cls = [1, 2, 4, 10, 40, 42, 46, 57, 61]
        class_groups = [[x] for x in cls]
        class_groups.append([3, 6, 8])
    elif train_cfg.sports_classes:
        class_groups = [people_cls, ball_cls]
    else:
        class_groups = [people_cls, twowheeler_cls, vehicle_cls, utensils_cls, furniture_cls]

    return class_groups

def configure_optimizer(optimizer_cfg, model):
    if optimizer_cfg.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=optimizer_cfg.lr,
                                     eps=optimizer_cfg.eps,
                                     weight_decay=optimizer_cfg.weight_decay)
    return optimizer

def load_model(model_cfg, num_classes):
    model = JITNet(num_classes)

    states = torch.load(model_cfg.pretrained_ckpt)
    model_states = {k.replace('module.', ''): v for k,
                    v in states['state_dict'].items()}
    filtered_model_states = {}
    for k, v in model_states.items():
        ignore = False
        for i in model_cfg.ignored_vars:
            if i in k:
                ignore = True
                break
        if ignore:
            continue
        filtered_model_states[k] = v
    load_ret = model.load_state_dict(filtered_model_states, strict=False)
    log.info(f"Vars not loaded {load_ret[0]}")

    return model

def load_video_stream(dataset_cfg):
    stream = MaskRCNNStream(dataset_cfg.video_path,
                            dataset_cfg.label_path)
    return stream

def inference(model, images):
    logits = model(images)  # [B, C, H, W]
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        log_probs = (probs + 1e-9).log()
        entropy = -(probs * log_probs).sum(1).mean()

    return logits, probs, entropy

def train(cfg):
    class_groups = get_class_groups(cfg.online_train)
    log.info(f'Number of class {len(class_groups) + 1}')

    # Init model, optimizer, loss, video stream
    device = torch.device('cuda')
    model = load_model(cfg.model, len(class_groups) + 1)
    model.to(device)
    stream = load_video_stream(cfg.dataset)
    optimizer = configure_optimizer(cfg.online_train.optimizer, model)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Online training stats
    train_cfg = cfg.online_train
    training_strides = train_cfg.training_strides
    curr_stride_idx = 0
    num_teacher_samples = 0
    num_updates = 0

    # Online training
    for curr_frame, (frame, boxes, classes, scores, masks, num_objects, frame_id) in enumerate(stream):
        if curr_frame > train_cfg.max_frames:
            break

        # Video frame and maskrcnn outputs
        frame = cv2.resize(frame, (train_cfg.image_width, train_cfg.image_height))
        frame = (frame - np.array(train_cfg.image_mean)) / np.array(train_cfg.image_std)
        frame = np.expand_dims(frame, axis=0)
        boxes = np.expand_dims(boxes, axis=0)
        classes = np.expand_dims(classes, axis=0)
        scores = np.expand_dims(scores, axis=0)
        masks = np.expand_dims(masks, axis=0)
        num_objects = np.expand_dims(num_objects, axis=0)

        train_stride = training_strides[curr_stride_idx]

        if curr_frame % train_stride == 0 and train_cfg.online_train:
            num_teacher_samples += 1
            start = time.time()

            # Convert maskrcnn outputs to dense labels
            labels_vals, label_weights_vals = \
                batch_segmentation_masks(1,
                                         (train_cfg.image_height, train_cfg.image_width),
                                         boxes, classes, masks, scores,
                                         num_objects, True,
                                         class_groups,
                                         scale_boxes=train_cfg.scale_boxes)
            labels_vals = labels_vals.astype(np.int32)
            label_weights_vals = label_weights_vals.astype(np.int32)

            # Make a batch of size 1
            frame_list = [frame]
            labels_list = [labels_vals]
            label_weights_list = [label_weights_vals]
            in_images = torch.tensor(np.concatenate(frame_list, axis=0)).to(device)
            in_images = in_images.permute(0, 3, 1, 2).float()  # [B, C, H, W]
            labels_vals = torch.tensor(
                np.concatenate(labels_list, axis=0)).to(device).long()
            label_weights_vals = torch.tensor(
                np.concatenate(label_weights_list, axis=0)).to(device).float()

            # Online optimization
            curr_updates = 0
            while curr_updates < train_cfg.max_updates:
                optimizer.zero_grad()

                logits, probs, entropy = inference(model, in_images)
                loss = criterion(logits, labels_vals)  # [B, H, W]

                # Weight foreground and background loss
                loss_weights = torch.ones_like(label_weights_vals) * train_cfg.fg_weight
                bg_mask = label_weights_vals == 0
                loss_weights.masked_fill_(bg_mask, train_cfg.bg_weight)
                loss = (loss * loss_weights).mean()

                loss.backward()
                optimizer.step()

                # Training stats
                with torch.no_grad():
                    labels_onehot = F.one_hot(
                        labels_vals, probs.shape[1]).permute(0, 3, 1, 2)  # [B, C, H, W]
                    fp = (probs * (1. - labels_onehot)).sum([0, 2, 3])  # [C]
                    tp = (probs * labels_onehot).sum([0, 2, 3])  # [C]
                    fn = ((1. - probs) * labels_onehot).sum([0, 2, 3])  # [C]
                    eps = 100.
                    cls_scores = (tp + eps) / (tp + fp + fn + eps)  # [C]
                    probs_max, preds = torch.max(probs, dim=1) # [B, H, W]

                num_updates = num_updates + 1
                curr_updates = curr_updates + 1

                # End training if min class accuracy > threshold
                min_cls_scores = torch.min(cls_scores)
                if min_cls_scores > train_cfg.accuracy_threshold:
                    break

            end = time.time()
            if min_cls_scores > train_cfg.accuracy_threshold:
                curr_stride_idx = min(curr_stride_idx + 1, len(training_strides) - 1)
            else:
                curr_stride_idx = max(curr_stride_idx - 1, 0)

            training_str = f"Fscore: {min_cls_scores:.5f}"
            stride_str = (f"num_teacher_samples: {num_teacher_samples} "
                          f"num_updates: {num_updates} "
                          f"stride: {training_strides[curr_stride_idx]}")

        elif curr_frame % train_cfg.inference_stride == 0:
            start = time.time()
            with torch.no_grad():
                logits, probs, entropy = inference(model, in_images)
            end = time.time()
            training_str = ""
            stride_str = ""

        log.info(f'frame: {curr_frame:05d} time: {end - start:.5f}s {training_str} {stride_str}')


@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    print(cfg.pretty())
    train(cfg)

if __name__=='__main__':
    main()
