import cv2
import argparse
import h5py
import os
import numpy as np
from tqdm import tqdm
import time

from dataloaders.maskrcnn_stream import batch_segmentation_masks
from dataloaders.maskrcnn_stream import MaskRCNNStream
import dataloaders.lvs_dataset as lvs_dataset

def main(args):
    class_groups = lvs_dataset.sequence_to_class_groups_stable[args.sequence]
    class_groups = [ [lvs_dataset.detectron_classes.index(c) for c in g] \
                     for g in class_groups]

    video_dir = os.path.join(args.input_dir, args.sequence)
    video_path = os.path.join(video_dir, f'{args.sequence}{args.video_id}.mp4')
    detection_path = os.path.join(video_dir, f'detectron_large_mask_rcnn_1_{args.sequence}{args.video_id}.npy')

    stream = MaskRCNNStream(video_path, detection_path, start_frame=args.start_frame)

    frame_detections = {}
    t0 = time.time()
    for curr_frame, (frame, boxes, classes, scores, masks, num_objects, frame_id) in tqdm(enumerate(stream)):
        if args.max_frames > 0 and curr_frame >= args.max_frames:
            break

        t1 = time.time()
        # Video frame and maskrcnn outputs
        frame = np.expand_dims(frame, axis=0)
        boxes = np.expand_dims(boxes, axis=0)
        classes = np.expand_dims(classes, axis=0)
        scores = np.expand_dims(scores, axis=0)
        masks = np.expand_dims(masks, axis=0)
        num_objects = np.expand_dims(num_objects, axis=0)
        t2 = time.time()

        # Convert maskrcnn outputs to dense labels
        labels_vals, label_weights_vals = \
            batch_segmentation_masks(1,
                                     frame.shape[1:],
                                     boxes, classes, masks, scores,
                                     num_objects, True,
                                     class_groups,
                                     scale_boxes=True)
        # Convert to uint8 to save space.
        labels_vals = labels_vals.astype(np.uint8)
        label_weights_vals = label_weights_vals.astype(np.uint8)
        t3 = time.time()

        frame_detections[frame_id] = (frame[0], labels_vals[0], label_weights_vals[0])
        #print(f'{t1 - t0:.3f} {t2 - t1:.3f} {t3 - t2:.3f}')

        t0 = time.time()

    with h5py.File(os.path.join(video_dir, f'{args.sequence}{args.video_id}.hdf5'), 'w') as f:
        frames = f.create_dataset('frames', (len(frame_detections),) + frame.shape,
                                  dtype=np.uint8)
        labels = f.create_dataset('labels', (len(frame_detections),) + labels_vals[0].shape,
                                  dtype=np.uint8)
        label_weights = f.create_dataset('label_weights', (len(frame_detections),) + label_weights_vals[0].shape,
                                         dtype=np.uint8)

        for frame in tqdm(sorted(frame_detections.keys())):
            frames[frame] = frame_detections[frame][0]
            labels[frame] = frame_detections[frame][1]
            label_weights[frame] = frame_detections[frame][2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--sequence', type=str)
    parser.add_argument('--video_id', type=str)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--max_frames', type=int, default=0)
    parser.add_argument('--image_width', type=int, default=1280)
    parser.add_argument('--image_height', type=int, default=720)
    args = parser.parse_args()

    main(args)
