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

def configure_optimizer(optimizer_cfg, model, ckpt_states=None):
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
    elif optimizer_cfg.name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=optimizer_cfg.lr,
                                        alpha=optimizer_cfg.alpha,
                                        weight_decay=optimizer_cfg.weight_decay)

    if ckpt_states:
        if 'optimizer' in ckpt_states:
            optimizer.load_state_dict(ckpt_states['optimizer'])

    return optimizer

def load_model(model_cfg, num_classes, num_models=1):
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

    models = []
    for i in range(num_models):
        model = JITNet(num_classes, **model_cfg.jitnet_params)

        load_ret = model.load_state_dict(filtered_model_states, strict=False)
        log.info(f"Vars not loaded {load_ret[0]}")

        def set_bn_eval(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                print(module)
                module.eval()

        model.train()
        #model.apply(set_bn_eval)

        models.append(model)

    return models, states

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
    video_files = video_files[dataset_cfg.sequence_id]
    detecttion_files = detecttion_files[dataset_cfg.sequence_id]

    class_groups = lvs_dataset.sequence_to_class_groups_stable[dataset_cfg.sequence]
    log.info(video_files)
    log.info(detecttion_files)
    log.info(class_groups)

    class_groups = [ [lvs_dataset.detectron_classes.index(c) for c in g] \
                     for g in class_groups]

    stream = MaskRCNNSequenceStream([video_files], [detecttion_files],
                                    start_frame=dataset_cfg.start_frame)

    return stream, class_groups

def inference(model, images, return_intermediate=False):
    logits, intermediate = model(images, return_intermediate=return_intermediate)  # [B, C, H, W]
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        log_probs = (probs + 1e-9).log()
        entropy = -(probs * log_probs).sum(1).mean()
        probs_max, preds = torch.max(probs, dim=1) # [B, H, W]

    return logits, probs, entropy, probs_max, preds, intermediate

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


def visualize_result_frame(vid_out, dist_map, frame, probs, preds,
                           labels, label_weights, num_classes, train_cfg):
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

    dist_map = cv2.resize(dist_map[0, 0], vis_frame.shape[:2][::-1],
                          interpolation=cv2.INTER_NEAREST)[:, :, None]
    dist_map = np.full(vis_frame.shape, 255) * dist_map / 50.
    dist_map = dist_map.astype(np.uint8)
    probs_image = np.full(vis_frame.shape, 255) * np.expand_dims(1 - probs[0], axis=2)
    probs_image = probs_image.astype(np.uint8)

    weights_image = np.full(vis_frame.shape, 255) * \
            np.expand_dims(label_weights[0] > 0, axis=2)
    weights_image = weights_image.astype(np.uint8)

    preds_image = cv2.addWeighted(vis_frame, 0.5, vis_preds, 0.5, 0)
    labels_image = cv2.addWeighted(vis_frame, 0.5, vis_labels, 0.5, 0)

    vis_image = np.concatenate((dist_map, labels_image, preds_image), axis=1)
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


def profile_models(models, model_perfs,
                   curr_frame, curr_model_idx, curr_stride_idx,
                   images, labels, num_classes, train_cfg):
    next_model_idx = curr_model_idx

    model_min_cls_scores = []
    model_mean_cls_scores = []
    for i, model in enumerate(models):
        with torch.no_grad():
            logits, probs, entropy, probs_max, preds = \
                inference(model, images)
            tp, fp, fn, cls_scores = calculate_class_iou(preds, labels, num_classes)
            model_min_cls_scores.append(torch.min(cls_scores))
            model_mean_cls_scores.append(torch.mean(cls_scores[1:]))

    best_model_idx = np.argmax(model_mean_cls_scores)
    best_model_acc = model_min_cls_scores[best_model_idx]

    for i, p in enumerate(model_perfs):
        p.append((curr_frame, model_mean_cls_scores[i].item()))

    if curr_frame < train_cfg.warmup:
        for i in np.argsort(model_mean_cls_scores):
            if i != curr_model_idx:
                next_model_idx = i
    else:
        avg_model_perf = [[] for _ in models]
        for i in range(1, 8):
            frame = model_perfs[0][-i][0]
            if curr_frame - frame > train_cfg.model_perf_win:
                break
            for j in range(len(models)):
                avg_model_perf[j].append(model_perfs[j][-i][1])
        avg_model_perf = [np.mean(p) for p in avg_model_perf]
        best_avg_perf_model = np.argmax(avg_model_perf)
        avg_model_perf_str = [f'{p:.4f}' for p in avg_model_perf]
        log.info(f'avg_model_perf={avg_model_perf_str} best_perf_model={best_avg_perf_model}')
        next_model_idx = best_avg_perf_model

    #if curr_stride_idx == 0:
    #    if curr_frame < train_cfg.warmup:
    #        # Model index from low to high
    #        for i in np.argsort(model_min_cls_scores):
    #            if i != curr_model_idx:
    #                next_model_idx = i
    #    else:
    #        next_model_idx = curr_model_idx
    #elif curr_stride_idx == 1:
    #    if model_min_cls_scores[curr_model_idx] < train_cfg.accuracy_lower_bound:
    #        # Model index from high to low
    #        for i in np.argsort(model_min_cls_scores)[::-1]:
    #            if i != curr_model_idx:
    #                next_model_idx = i
    #                break

    return next_model_idx, best_model_idx, best_model_acc


def train(cfg):
    torch.manual_seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)

    # Init model, optimizer, loss, video stream
    stream, class_groups = load_video_stream(cfg.dataset)
    num_classes = len(class_groups) + 1
    log.info(f'Number of class {num_classes}')
    device = torch.device('cuda')
    models, ckpt_states = load_model(
        cfg.model, num_classes, cfg.online_train.num_models)
    for m in models:
        m.to(device)
    optimizers = [configure_optimizer(cfg.online_train.optimizer,
                                      m,
                                      ckpt_states if cfg.online_train.resume_online else None)
                  for m in models]
    cls_weight = None
    if cfg.online_train.cls_weight:
        cls_weight = cfg.online_train.cls_weight[:num_classes]
        cls_weight = torch.tensor(cls_weight).float()
    criterion = torch.nn.CrossEntropyLoss(weight=cls_weight, reduction='none')
    criterion.to(device)

    if cfg.online_train.ema:
        model_ema, _ = load_model(cfg.model, num_classes)
        model_ema.to(device)
        model_ema.eval()

    if cfg.online_train.freeze_enc:
        for m in models:
            m.freeze_enc()

    if cfg.online_train.freeze_dec:
        for m in models:
            m.freeze_dec()

    # Online training stats
    train_cfg = cfg.online_train
    training_strides = train_cfg.training_strides
    start_frame = cfg.dataset.start_frame
    curr_stride_idx = 0
    curr_model_idx = 0
    eval_model_idx = 0
    best_model_acc = 0.0
    num_teacher_samples = 0
    num_updates = 0
    per_frame_stats = {}
    class_iou = np.zeros(num_classes, np.float32)
    model_perfs = [[] for _ in models]
    prev_intermediate = None
    force_update = False
    next_update_frame = 0

    vid_out = None
    if train_cfg.video:
        vid_out = cv2.VideoWriter(train_cfg.video,
                                  cv2.VideoWriter_fourcc(*'JPEG'),
                                  #cv2.VideoWriter_fourcc(*'X264'),
                                  stream.rate,
                                  (3 * int(train_cfg.image_width / 2),
                                   int(train_cfg.image_height / 2)))
        assert vid_out

    if not train_cfg.online_train:
        for m in models:
            m.eval()

    # Online training
    replay_buffer = {}
    for curr_frame, (frame, boxes, classes, scores, masks, num_objects, frame_id) in enumerate(stream):
        if curr_frame > train_cfg.max_frames:
            break

        # Video frame and maskrcnn outputs
        frame = frame.astype(np.float32) / 255.
        frame = (frame - np.array(train_cfg.image_mean)) / np.array(train_cfg.image_std)
        frame = np.expand_dims(frame, axis=0)
        boxes = np.expand_dims(boxes, axis=0)
        classes = np.expand_dims(classes, axis=0)
        scores = np.expand_dims(scores, axis=0)
        masks = np.expand_dims(masks, axis=0)
        num_objects = np.expand_dims(num_objects, axis=0)

        # Convert maskrcnn outputs to dense labels
        labels_vals, label_weights_vals = \
            batch_segmentation_masks(1,
                                     (train_cfg.image_height, train_cfg.image_width),
                                     boxes, classes, masks, scores,
                                     num_objects, True,
                                     class_groups,
                                     scale_boxes=train_cfg.scale_boxes)

        frame = frame.transpose(0, 3, 1, 2).astype(np.float32)  # [1, H, W, C]
        labels_vals = labels_vals.astype(np.int64)  # [1, H, W]
        label_weights_vals = label_weights_vals.astype(np.float32)  # [1, H, W]

        # Make a batch of size 1
        frame = torch.tensor(frame)
        labels_vals = torch.tensor(labels_vals)
        label_weights_vals = torch.tensor(label_weights_vals)

        curr_updates = 0
        if (curr_frame == next_update_frame and train_cfg.online_train) or force_update:
            force_update = False

            replay_buffer[curr_frame] = {
                'frame': frame,
                'label': labels_vals,
                'label_weight': label_weights_vals
            }

            num_teacher_samples += 1
            start = time.time()

            # Online optimization
            min_cls_scores = []

            # Profile models
            if len(models) > 1:
                new_model_idx, best_model_idx, best_model_acc = \
                    profile_models(models, model_perfs, curr_frame, curr_model_idx, curr_stride_idx,
                                               in_images, labels_vals, num_classes,
                                               train_cfg)
                if eval_model_idx != best_model_idx:
                    eval_model_idx = best_model_idx
                if new_model_idx != curr_model_idx:
                    curr_model_idx = new_model_idx
                    curr_stride_idx = 0
                    if curr_frame > train_cfg.warmup:
                        optimizers[curr_model_idx] = configure_optimizer(train_cfg.optimizer, models[curr_model_idx])
                log.info(f'curr_model_idx={curr_model_idx} eval_model_idx={eval_model_idx}')

            model = models[curr_model_idx]
            optimizer = optimizers[curr_model_idx]
            while curr_updates < train_cfg.max_updates // train_cfg.replay_samples:
                optimizer.zero_grad()

                # Random sample from replay buffer
                if train_cfg.replay_samples > 1 and len(replay_buffer) > train_cfg.replay_samples - 1:
                    sample_idx = np.random.choice(list(replay_buffer.keys()),
                                                  (train_cfg.replay_samples - 1,),
                                                  replace=False)
                    frame_batch = [frame] + [replay_buffer[s]['frame'] for s in sample_idx]
                    label_batch = [labels_vals] + [replay_buffer[s]['label'] for s in sample_idx]
                    label_weight_batch = [label_weights_vals] + [replay_buffer[s]['label_weight'] for s in sample_idx]
                    frame_batch = torch.cat(frame_batch, dim=0)
                    label_batch = torch.cat(label_batch, dim=0)
                    label_weight_batch = torch.cat(label_weight_batch, dim=0)
                else:
                    frame_batch = frame
                    label_batch = labels_vals
                    label_weight_batch = label_weights_vals

                frame_batch = frame_batch.to(device)
                label_batch = label_batch.to(device)
                label_weight_batch = label_weight_batch.to(device)

                logits, probs, entropy, probs_max, preds, intermediate = \
                    inference(model, frame_batch, True)
                prev_intermediate = intermediate
                tp, fp, fn, cls_scores = calculate_class_iou(preds[:1], label_batch[:1], num_classes)
                logpt = criterion(logits, label_batch)  # [B, H, W]

                # Weight foreground and background loss
                fg_weights = torch.ones_like(label_weight_batch) * train_cfg.fg_weight
                bg_mask = label_weight_batch == 0
                fg_weights.masked_fill_(bg_mask, train_cfg.bg_weight)
                if train_cfg.focal_gamma > 0:
                    pt = torch.exp(-logpt)
                    loss = (((1. - pt) ** train_cfg.focal_gamma) * logpt * fg_weights).mean()
                else:
                    loss = (logpt * fg_weights).mean()

                loss.backward()
                optimizer.step()

                if train_cfg.ema:
                    with torch.no_grad():
                        for p1, p2 in zip(model.parameters(), model_ema.parameters()):
                            p2.data = p2.data * train_cfg.ema_m + p1.data * (1. - train_cfg.ema_m)

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

            with torch.no_grad():
                for i, model in enumerate(models):
                    if i == curr_model_idx:
                        continue
                    for p1, p2 in zip(model.parameters(), models[curr_model_idx].parameters()):
                        momentum = train_cfg.model_avg_m
                        p1.data = p1.data * momentum + p2.data * (1 - momentum)

            if min_cls_scores[-1] > train_cfg.accuracy_threshold:
                curr_stride_idx = min(curr_stride_idx + 1, len(training_strides) - 1)
            else:
                curr_stride_idx = max(curr_stride_idx - 1, 0)

            next_update_frame = curr_frame + training_strides[curr_stride_idx]

            if min_cls_scores[-1] > best_model_acc and eval_model_idx != curr_model_idx:
                eval_model_idx = curr_model_idx
                log.info(f'eval model index {eval_model_idx}')

            min_cls_scores = [f'{c:.3f}' for c in min_cls_scores]
            training_str = f"Fscore: {min_cls_scores}"
            stride_str = (f"num_teacher_samples: {num_teacher_samples} "
                          f"num_updates: {num_updates} "
                          f"stride: {training_strides[curr_stride_idx]}")

        elif curr_frame % train_cfg.inference_stride == 0:
            start = time.time()
            model = models[curr_model_idx]
            optimizer = optimizers[curr_model_idx]
            with torch.no_grad():
                in_images = frame.to(device)
                labels_vals = labels_vals.to(device)

                logits, probs, entropy, probs_max, preds, intermediate = \
                        inference(model if not train_cfg.ema else model_ema, in_images, True)
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

        with torch.no_grad():
            #intermediate = F.avg_pool2d(intermediate[:1], intermediate.shape[2:]).flatten(start_dim=1)
            if prev_intermediate is not None:
                #cos_sim = F.cosine_similarity(F.avg_pool2d(intermediate, 1),
                #                              F.avg_pool2d(prev_intermediate, 1)).min()
                dist_map = ((intermediate - prev_intermediate) ** 2).sum(1) ** 0.5
                dist = dist_map.mean()
            else:
                #cos_sim = 1.0
                dist_map = None
                dist = 0.0

            if dist > train_cfg.force_update_thresh:
                force_update = True

        if vid_out:
            visualize_result_frame(vid_out,
                                   None if dist_map is None else dist_map.cpu().numpy(),
                                   frame.cpu().numpy(),
                                   probs_max.cpu().numpy(),
                                   preds.cpu().numpy(),
                                   labels_vals.cpu().numpy(),
                                   label_weights_vals.cpu().numpy(),
                                   len(class_groups),
                                   train_cfg)


        log.info(f'frame: {curr_frame + start_frame:05d} time: {end - start:.5f}s {training_str} {dist:.3f} {stride_str}')

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

    states = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if train_cfg.ema:
        states['model_ema'] = model_ema.state_dict()
    torch.save(states, f'final.pth')


@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    print(cfg.pretty())
    train(cfg)

if __name__=='__main__':
    main()
