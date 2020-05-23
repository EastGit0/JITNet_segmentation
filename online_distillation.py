import os
import time

import cv2
import hydra
import numpy as np
import logging
import torch
import torch.nn.functional as F
from torchvision import transforms

from models import JITNet
from dataloaders.maskrcnn_stream import (batch_segmentation_masks,
                                         visualize_masks,
                                         MaskRCNNSequenceStream)
import dataloaders.lvs_dataset as lvs_dataset


log = logging.getLogger(__name__)

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
    sequence_to_video_list = lvs_dataset.get_sequence_to_video_list(
        dataset_cfg.data_dir,
        dataset_cfg.data_dir,
        lvs_dataset.video_sequences_stable
    )
    assert dataset_cfg.sequence in sequence_to_video_list

    video_files = []
    detecttion_files = []
    for s in sequence_to_video_list[dataset_cfg.sequence]:
        video_files.append(os.path.join(
            dataset_cfg.data_dir, dataset_cfg.sequence, s[0]))
        detecttion_files.append(os.path.join(
            dataset_cfg.data_dir, dataset_cfg.sequence, s[1]))

    class_groups = lvs_dataset.sequence_to_class_groups_stable[dataset_cfg.sequence]
    log.info(video_files)
    log.info(detecttion_files)
    log.info(class_groups)

    class_groups = [ [lvs_dataset.detectron_classes.index(c) for c in g] \
                     for g in class_groups]

    stream = MaskRCNNSequenceStream(video_files, detecttion_files)

    return stream, class_groups

def inference(model, images):
    logits = model(images)  # [B, C, H, W]
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        log_probs = (probs + 1e-9).log()
        entropy = -(probs * log_probs).sum(1).mean()

    return logits, probs, entropy


def visualize_result_frame(vid_out, frame, probs, preds, labels, label_weights, num_classes, train_cfg):
    # frame: [B, C, H, W]
    # probs, preds, labels, label_weights: [B, H, W]
    vis_preds = visualize_masks(preds, preds.shape[0],
                                (preds.shape[1], preds.shape[2], 3),
                                num_classes=num_classes)
    vis_labels = visualize_masks(labels, labels.shape[0],
                                 (labels.shape[1], labels.shape[2], 3),
                                 num_classes=num_classes)

    vis_preds = vis_preds[0]
    vis_labels = vis_labels[0]
    vis_frame = np.transpose(frame[0], (1, 2, 0))
    vis_frame = (vis_frame * np.array(train_cfg.image_std) + np.array(train_cfg.image_mean)) * 255.
    vis_frame = vis_frame.astype(np.uint8)

    probs_image = np.full(vis_frame.shape, 255) * np.expand_dims(1 - probs[0], axis=2)
    probs_image = probs_image.astype(np.uint8)

    weights_image = np.full(vis_frame.shape, 255) * \
            np.expand_dims(label_weights[0] > 0, axis=2)
    weights_image = weights_image.astype(np.uint8)

    preds_image = cv2.addWeighted(vis_frame, 0.5, vis_preds, 0.5, 0)
    labels_image = cv2.addWeighted(vis_frame, 0.5, vis_labels, 0.5, 0)

    vis_image = np.concatenate((probs_image, labels_image, preds_image), axis=1)
    #vis_image = np.concatenate((weights_image, labels_image, preds_image), axis=1)

    vis_image = vis_image[::2, ::2, :]

    ret = vid_out.write(vis_image)


def update_stats(labels, pred_vals, class_tp, class_fp, class_fn,
                 class_total, class_correct, weight_mask,
                 entropy_vals, frame_id, ran_teacher,
                 num_updates, frame_stats):
    eps = 1e-06
    num_classes = len(class_total)
    curr_tp = np.zeros(num_classes, np.float32)
    curr_fp = np.zeros(num_classes, np.float32)
    curr_fn = np.zeros(num_classes, np.float32)
    curr_iou = np.zeros(num_classes, np.float32)
    curr_correct = np.zeros(num_classes, np.float32)
    curr_total = np.zeros(num_classes, np.float32)
    correct_mask = (pred_vals == labels)

    for g in range(num_classes):
        cls_mask = np.logical_and((labels == g), weight_mask)
        cls_tp_mask = np.logical_and(cls_mask, correct_mask)
        cls_tp = np.sum(cls_tp_mask)
        curr_tp[g] = cls_tp
        class_tp[g] = class_tp[g] + cls_tp

        cls_total = np.sum(cls_mask)
        curr_total[g] = cls_total
        curr_correct[g] = cls_tp
        class_total[g] = class_total[g] + cls_total
        class_correct[g] = class_correct[g] + cls_tp

        pred_mask = np.logical_and((pred_vals == g), weight_mask)
        cls_fp_mask = np.logical_and(np.logical_not(cls_mask), pred_mask)
        cls_fn_mask = np.logical_and(cls_mask, np.logical_not(pred_mask))

        cls_fp = np.sum(cls_fp_mask)
        cls_fn = np.sum(cls_fn_mask)
        curr_fp[g] = cls_fp
        curr_fn[g] = cls_fn
        class_fp[g] = class_fp[g] + cls_fp
        class_fn[g] = class_fn[g] + cls_fn

        cls_iou = (cls_tp + eps) / (cls_tp + cls_fp + cls_fn + eps)
        curr_iou[g] = cls_iou

    frame_stats[frame_id] = { 'tp': curr_tp,
                              'fp': curr_fp,
                              'fn': curr_fn,
                              'iou': curr_iou,
                              'correct': curr_correct,
                              'total': curr_total,
                              'average_entropy': entropy_vals,
                              'ran_teacher': ran_teacher,
                              'num_updates': num_updates }

def train(cfg):
    # Init model, optimizer, loss, video stream
    stream, class_groups = load_video_stream(cfg.dataset)
    num_classes = len(class_groups) + 1
    log.info(f'Number of class {num_classes}')
    device = torch.device('cuda')
    model = load_model(cfg.model, num_classes)
    model.to(device)
    optimizer = configure_optimizer(cfg.online_train.optimizer, model)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Online training stats
    train_cfg = cfg.online_train
    training_strides = train_cfg.training_strides
    curr_stride_idx = 0
    num_teacher_samples = 0
    num_updates = 0
    per_frame_stats = {}
    class_correct = np.zeros(num_classes, np.float32)
    class_total = np.zeros(num_classes, np.float32)
    class_tp = np.zeros(num_classes, np.float32)
    class_fp = np.zeros(num_classes, np.float32)
    class_fn = np.zeros(num_classes, np.float32)
    class_iou = np.zeros(num_classes, np.float32)

    vid_out = None
    if train_cfg.video_output_path:
        vid_out = cv2.VideoWriter(train_cfg.video_output_path,
                                  cv2.VideoWriter_fourcc(*'JPEG'),
                                  stream.rate,
                                  (3 * int(train_cfg.image_width / 2),
                                   int(train_cfg.image_height / 2)))
        assert vid_out

    # Online training
    for curr_frame, (frame, boxes, classes, scores, masks, num_objects, frame_id) in enumerate(stream):
        if curr_frame > train_cfg.max_frames:
            break

        # Video frame and maskrcnn outputs
        frame = cv2.resize(frame, (train_cfg.image_width, train_cfg.image_height))
        frame = frame.astype(np.float) / 255.
        frame = (frame - np.array(train_cfg.image_mean)) / np.array(train_cfg.image_std)
        frame = np.expand_dims(frame, axis=0)
        boxes = np.expand_dims(boxes, axis=0)
        classes = np.expand_dims(classes, axis=0)
        scores = np.expand_dims(scores, axis=0)
        masks = np.expand_dims(masks, axis=0)
        num_objects = np.expand_dims(num_objects, axis=0)

        train_stride = training_strides[curr_stride_idx]

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

        curr_updates = 0
        if curr_frame % train_stride == 0 and train_cfg.online_train:
            num_teacher_samples += 1
            start = time.time()

             # Online optimization
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
                    probs_max, preds = torch.max(probs, dim=1) # [B, H, W]
                    labels_onehot = F.one_hot(
                        labels_vals, probs.shape[1]).permute(0, 3, 1, 2)  # [B, C, H, W]
                    preds_onehot = F.one_hot(preds, probs.shape[1]).permute(0, 3, 1, 2)  # [B, C, H, W]
                    fp = (preds_onehot * (1. - labels_onehot)).sum([0, 2, 3])  # [C]
                    tp = (preds_onehot * labels_onehot).sum([0, 2, 3])  # [C]
                    fn = ((1. - preds_onehot) * labels_onehot).sum([0, 2, 3])  # [C]
                    eps = 100.
                    cls_scores = (tp + eps) / (tp + fp + fn + eps)  # [C]

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
                probs_max, preds = torch.max(probs, dim=1)
            end = time.time()
            training_str = ""
            stride_str = ""

        if train_cfg.stats_path:
            update_stats(labels_vals[0].cpu().numpy(),
                         preds[0].cpu().numpy(),
                         class_tp,
                         class_fp,
                         class_fn,
                         class_total,
                         class_correct,
                         np.ones(labels_vals.shape, dtype=np.bool),
                         entropy.cpu().numpy(),
                         curr_frame,
                         len(training_str) > 0,
                         curr_updates,
                         per_frame_stats)

        if vid_out:
            visualize_result_frame(vid_out,
                                   in_images.cpu().numpy(),
                                   probs_max.cpu().numpy(),
                                   preds.cpu().numpy(),
                                   labels_vals.cpu().numpy(),
                                   label_weights_vals.cpu().numpy(),
                                   len(class_groups),
                                   train_cfg)

        log.info(f'frame: {curr_frame:05d} time: {end - start:.5f}s {training_str} {stride_str}')

    if train_cfg.stats_path:
        np.save(train_cfg.stats_path, [per_frame_stats])

    if vid_out:
        vid_out.release()


@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    print(cfg.pretty())
    train(cfg)

if __name__=='__main__':
    main()
