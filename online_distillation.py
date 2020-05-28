import os
import time
import pdb

import cv2
import hydra
import numpy as np
import logging
import torch
import torch.nn.functional as F
from torchvision import transforms

from analytics import full_segment_iou
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
    elif optimizer_cfg.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=optimizer_cfg.lr,
                                    momentum=optimizer_cfg.momentum,
                                    nesterov=optimizer_cfg.nesterov,
                                    weight_decay=optimizer_cfg.weight_decay)
    return optimizer

def load_model(model_cfg, num_classes):
    model = JITNet(num_classes, **model_cfg.jitnet_params)

    states = torch.load(model_cfg.pretrained_ckpt)
    model_key = 'state_dict' if 'state_dict' in states else 'model'
    model_states = {k.replace('module.', ''): v for k,
                    v in states[model_key].items()}
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

    def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            print(module)
            module.eval()

    model.train()
    #model.apply(set_bn_eval)

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
    video_files = video_files[:dataset_cfg.sequence_limit]
    detecttion_files = detecttion_files[:dataset_cfg.sequence_limit]

    class_groups = lvs_dataset.sequence_to_class_groups_stable[dataset_cfg.sequence]
    log.info(video_files)
    log.info(detecttion_files)
    log.info(class_groups)

    class_groups = [ [lvs_dataset.detectron_classes.index(c) for c in g] \
                     for g in class_groups]

    stream = MaskRCNNSequenceStream(video_files, detecttion_files,
                                    start_frame=dataset_cfg.start_frame)

    return stream, class_groups

def inference(model, images):
    logits = model(images)  # [B, C, H, W]
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        log_probs = (probs + 1e-9).log()
        entropy = -(probs * log_probs).sum(1).mean()
        probs_max, preds = torch.max(probs, dim=1) # [B, H, W]

    return logits, probs, entropy, probs_max, preds

def calculate_class_iou(preds, labels, num_classes):
    with torch.no_grad():
        labels_onehot = F.one_hot(labels, num_classes).bool()  # [B, H, W, C]
        preds_onehot = F.one_hot(preds, num_classes).bool()  # [B, H, W, C]
        fp = (preds_onehot & ~labels_onehot).float().sum([0, 1, 2])  # [C]
        tp = (preds_onehot & labels_onehot).float().sum([0, 1, 2])  # [C]
        fn = (~preds_onehot & labels_onehot).float().sum([0, 1, 2])  # [C]
        eps = 1e-6
        cls_ious = (tp + eps) / (tp + fp + fn + eps)  # [C]

        return tp, fp, fn, cls_ious

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


def update_stats(tp, fp, fn, iou,
                 entropy_vals, frame_id, ran_teacher,
                 num_updates, frame_stats):
    frame_stats[frame_id] = { 'tp': tp,
                              'fp': fp,
                              'fn': fn,
                              'iou': iou,
                              'average_entropy': entropy_vals,
                              'ran_teacher': ran_teacher,
                              'num_updates': num_updates }

def train(cfg):
    torch.manual_seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)

    # Init model, optimizer, loss, video stream
    stream, class_groups = load_video_stream(cfg.dataset)
    num_classes = len(class_groups) + 1
    log.info(f'Number of class {num_classes}')
    device = torch.device('cuda')
    model = load_model(cfg.model, num_classes)
    model.to(device)
    optimizer = configure_optimizer(cfg.online_train.optimizer, model)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion.to(device)

    # Online training stats
    train_cfg = cfg.online_train
    training_strides = train_cfg.training_strides
    start_frame = cfg.dataset.start_frame
    curr_stride_idx = 0
    num_teacher_samples = 0
    num_updates = 0
    per_frame_stats = {}
    class_iou = np.zeros(num_classes, np.float32)

    vid_out = None
    if train_cfg.video_output_path:
        vid_out = cv2.VideoWriter(train_cfg.video_output_path,
                                  cv2.VideoWriter_fourcc(*'JPEG'),
                                  stream.rate,
                                  (3 * int(train_cfg.image_width / 2),
                                   int(train_cfg.image_height / 2)))
        assert vid_out

    if not train_cfg.online_train:
        model.eval()

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
            min_cls_scores = []
            while curr_updates < train_cfg.max_updates:
                optimizer.zero_grad()

                logits, probs, entropy, probs_max, preds = \
                    inference(model, in_images)
                tp, fp, fn, cls_scores = calculate_class_iou(preds, labels_vals, num_classes)
                loss = criterion(logits, labels_vals)  # [B, H, W]

                # Weight foreground and background loss
                loss_weights = torch.ones_like(label_weights_vals) * train_cfg.fg_weight
                bg_mask = label_weights_vals == 0
                loss_weights.masked_fill_(bg_mask, train_cfg.bg_weight)
                loss = (loss * loss_weights).mean()

                loss.backward()
                optimizer.step()

                num_updates = num_updates + 1
                curr_updates = curr_updates + 1

                min_cls_score = torch.min(cls_scores)
                min_cls_scores.append(min_cls_score)

                # Checkpoint
                if min_cls_score > train_cfg.checkpoint_threshold:
                    log.info(f'Checkpoint frame_{curr_frame + start_frame}.pth')
                    states = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'frame_id': curr_frame + start_frame,
                        'label': labels_vals[0].cpu()
                    }
                    torch.save(states, f'frame_{curr_frame + start_frame}.pth')

                # End training if min class accuracy > threshold
                if min_cls_score > train_cfg.accuracy_threshold:
                   break

            end = time.time()
            if min_cls_scores[-1] > train_cfg.accuracy_threshold:
                curr_stride_idx = min(curr_stride_idx + 1, len(training_strides) - 1)
            else:
                curr_stride_idx = max(curr_stride_idx - 1, 0)

            min_cls_scores = [f'{c:.3f}' for c in min_cls_scores]
            training_str = f"Fscore: {min_cls_scores}"
            stride_str = (f"num_teacher_samples: {num_teacher_samples} "
                          f"num_updates: {num_updates} "
                          f"stride: {training_strides[curr_stride_idx]}")

        elif curr_frame % train_cfg.inference_stride == 0:
            start = time.time()
            with torch.no_grad():
                logits, probs, entropy, probs_max, preds = \
                        inference(model, in_images)
                tp, fp, fn, cls_scores = calculate_class_iou(preds,
                                                             labels_vals,
                                                             num_classes)

            end = time.time()
            training_str = f"Fscore: {torch.min(cls_scores):.3f}"
            stride_str = ""

        if train_cfg.stats_path:
            update_stats(tp.cpu().numpy(),
                         fp.cpu().numpy(),
                         fn.cpu().numpy(),
                         cls_scores.cpu().numpy(),
                         entropy.cpu().numpy(),
                         curr_frame + start_frame,
                         len(stride_str) > 0,
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

        log.info(f'frame: {curr_frame + start_frame:05d} time: {end - start:.5f}s {training_str} {stride_str}')

    if train_cfg.stats_path:
        np.save(train_cfg.stats_path, [per_frame_stats])
        class_names = lvs_dataset.sequence_to_class_groups_stable[cfg.dataset.sequence]
        class_names = ['background'] + ['_'.join(g) for g in class_names]
        full_segment_iou.make_table(class_names,
                                    [[f'{train_cfg.stats_path}.npy', 'jitnet']],
                                    start_frame + train_cfg.max_frames,
                                    'result.csv',
                                    [],
                                    0
        )

    if vid_out:
        vid_out.release()


@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    print(cfg.pretty())
    train(cfg)

if __name__=='__main__':
    main()
