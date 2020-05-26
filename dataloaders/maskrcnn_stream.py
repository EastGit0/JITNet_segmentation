import os
import cv2
import numpy as np
import time
from itertools import chain

def visualize_masks(labels, batch_size, image_shape,
                    num_classes = 5):

    masks = []
    for label in range(1, num_classes + 1):
        masks.append(labels == label)

    labels_vis = np.zeros((batch_size,
                           image_shape[0],
                           image_shape[1],
                           image_shape[2]), np.uint8)

    cmap = [[166, 206, 227],
            [178, 223, 138],
            [31,  120, 180],
            [51,  160,  44],
            [251, 154, 153],
            [227,  26,  28],
            [253, 191, 111],
            [255, 127,   0],
            [202, 178, 214],
            [106,  61, 154],
            [255, 255, 153],
            [177, 89,   40],
            [125, 125, 125]] # added a gray one. might not be perfect
    for i in range(num_classes):
        labels_vis[masks[i]] = cmap[i]

    return labels_vis

def mask_rcnn_unmold_cls_mask(mask, bbox, image_shape, idx, full_masks,
                              box_masks, cls, compute_box_mask=False,
                              dialate=True, threshold = 0.5):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a binary or weighted mask with the same size as the original image.
    """
    y1, x1, y2, x2 = bbox
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return

    mask = cv2.resize(mask, (x2 - x1, y2 - y1)).astype(np.float32)

    thresh_mask = np.where(np.logical_and(mask >= threshold,
                                          cls > full_masks[y1:y2, x1:x2]),
                                          cls, full_masks[y1:y2, x1:x2]).astype(np.uint8)
    # Put the mask in the right location.
    full_masks[y1:y2, x1:x2] = thresh_mask

    if box_masks is not None:
        if dialate:
            dialate_frac = 0.15
            dy1 = max(int(y1 - dialate_frac * (y2 - y1)), 0)
            dx1 = max(int(x1 - dialate_frac * (x2 - x1)), 0)

            dy2 = min(int(y2 + dialate_frac * (y2 - y1)), image_shape[0])
            dx2 = min(int(x2 + dialate_frac * (x2 - x1)), image_shape[1])

            mask = cv2.resize(mask, (dx2 - dx1, dy2 - dy1)).astype(np.float32)
            box_masks[dy1:dy2, dx1:dx2] = np.where(mask >= 0, 1, 0).astype(np.bool)
        else:
            box_masks[y1:y2, x1:d2] = np.where(mask >= 0, 1, 0).astype(np.bool)

def mask_rcnn_single_mask(boxes, classes, scores, masks, image_shape,
                          box_mask=False, box_threshold=0.5,
                          mask_threshold=0.5):
    N = len(boxes)
    # Resize masks to original image size and set boundary threshold.
    full_masks = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    box_masks = np.zeros((image_shape[0], image_shape[1]), dtype=np.bool)

    for i in range(N):
        if scores[i] < box_threshold:
            continue
        # Convert neural network mask to full size mask
        mask_rcnn_unmold_cls_mask(masks[i], boxes[i], image_shape,
                                  i, full_masks,
                                  box_masks, classes[i],
                                  compute_box_mask=box_mask,
                                  threshold=mask_threshold)
    return full_masks, box_masks

def batch_segmentation_masks(batch_size,
                             image_shape,
                             batch_boxes,
                             batch_classes,
                             batch_masks,
                             batch_scores,
                             batch_num_objects,
                             compute_weight_masks,
                             class_groups,
                             mask_threshold=0.5,
                             box_threshold=0.5,
                             scale_boxes=True):
    h = image_shape[0]
    w = image_shape[1]

    seg_masks = np.zeros((batch_size, h, w), np.uint8)
    weight_masks = np.zeros((batch_size, h, w), np.bool)

    class_remap = {}
    for g in range(len(class_groups)):
        for c in class_groups[g]:
            class_remap[c] = g + 1

    batch_boxes = batch_boxes.copy()

    if scale_boxes and len(batch_boxes.shape) == 3:
        batch_boxes[:, :, 0] = batch_boxes[:, :, 0] * h
        batch_boxes[:, :, 2] = batch_boxes[:, :, 2] * h
        batch_boxes[:, :, 1] = batch_boxes[:, :, 1] * w
        batch_boxes[:, :, 3] = batch_boxes[:, :, 3] * w

    batch_boxes = batch_boxes.astype(np.int32)

    for b in range(batch_size):
        N = batch_num_objects[b]
        if N == 0:
            continue
        boxes = batch_boxes[b, :N, :]
        masks = batch_masks[b, :N, :, :]
        scores = batch_scores[b, :N]
        classes = batch_classes[b, :N]

        for i in range(classes.shape[0]):
            if classes[i] in class_remap:
                classes[i] = class_remap[classes[i]]
            else:
                classes[i] = 0

        idx = classes > 0
        boxes = boxes[idx]
        masks = masks[idx]
        classes = classes[idx]
        scores = scores[idx]

        full_masks, box_masks = mask_rcnn_single_mask(boxes, classes,
                                                      scores, masks,
                                                      image_shape,
                                                      box_mask=compute_weight_masks,
                                                      box_threshold=box_threshold,
                                                      mask_threshold=mask_threshold)
        seg_masks[b] = full_masks
        weight_masks[b] = box_masks

    return seg_masks, weight_masks


class MaskRCNNStream:
    def __init__(self, video_stream_path, detections_path,
                 start_frame=0, num_frames=None, stride=1,
                 loop=False):
        assert(os.path.isfile(video_stream_path))
        assert(os.path.isfile(detections_path))
        self.cap = cv2.VideoCapture(video_stream_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.num_frames = num_frames

        self.detections_path = detections_path
        self.detections = None
        self.stride = stride
        self.loop = loop

        assert(start_frame >= 0)
        self.start_frame = start_frame
        self.end_frame = self.length

        # Seek to the start frame
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def __next__(self):
        if self.detections is None:
            self.detections = np.load(self.detections_path, allow_pickle=True)[()]
            self.labeled_frames = list(self.detections.keys())
            self.num_labeled_frames = len(self.labeled_frames)
            if self.num_frames is not None:
                assert(self.start_frame + self.num_frames <= self.length)
                self.end_frame = (self.start_frame + self.num_frames) - 1

        frame = None
        boxes = None
        classes = None
        scores = None
        masks = None
        labels_not_found = True
        while labels_not_found:
            frame_id = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if (not ret) or (frame_id >= self.end_frame - 1):
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                    frame_id = self.start_frame
                else:
                    self.detections = None
                    raise StopIteration

            if frame_id in self.detections and frame_id % self.stride==0:
                boxes, classes, scores, masks = self.detections[frame_id]
                labels_not_found = False

        return frame, boxes, classes, scores, masks, scores.shape[0], frame_id

    def __iter__(self):
        return self

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    next = __next__


class MaskRCNNMultiStream:
    def __init__(self, video_paths, detections_paths,
                 start_frame=0, stride=1):
        self.streams = []
        self.stream_idx = 0
        self.num_streams = len(video_paths)
        print(video_paths)
        for d in range(len(video_paths)):
            input_stream = MaskRCNNStream(video_paths[d], detections_paths[d],
                                          start_frame=start_frame, stride=stride)
            self.streams.append(input_stream)

    def __next__(self):
        self.stream_idx = (self.stream_idx + 1) % self.num_streams
        frame, boxes, classes, scores, masks, scores.shape[0], frame_id

        return self.streams[self.stream_idx].__next__()

    def __iter__(self):
        return self

    next = __next__

class MaskRCNNSequenceStream:
    def __init__(self, video_paths, detections_paths,
                 start_frame=0, stride=1):
        self.streams = []
        self.stream_idx = 0
        self.num_streams = len(video_paths)
        self.rate = 0
        for d in range(len(video_paths)):
            input_stream = MaskRCNNStream(video_paths[d], detections_paths[d],
                                          start_frame=start_frame, stride=stride,
                                          loop=False)
            self.streams.append(input_stream)
            #print(self.rate, input_stream.rate)
            if self.rate == 0:
                self.rate = input_stream.rate
            #else:
            #    assert(self.rate == input_stream.rate)
        self.seq_stream = chain(*(self.streams))

    def __next__(self):
        return next(self.seq_stream)

    def __iter__(self):
        return self

    next = __next__


if __name__ == "__main__":
    mask_rcnn_stream = MaskRCNNStream('/home/cfan/lvsdataset/driving1/driving1000.mp4',
            '/home/cfan/lvsdataset/driving1/detectron_large_mask_rcnn_1_driving1000.npy', num_frames=100)
    count = 0
    start = time.time()
    for s in mask_rcnn_stream:
        frame, boxes, classes, scores, masks, num_objects, frame_id = s
        count = count + 1
    end = time.time()
    print(count, (end - start)/count)