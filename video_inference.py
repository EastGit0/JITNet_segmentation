from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
from collections import defaultdict
from six.moves import xrange
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all, im_detect_raw_masks
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

sys.path.append(os.path.realpath('./Detectron.pytorch'))
sys.path.append(os.path.realpath('./utils'))
from stream import VideoInputStream

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def worker_stream():
    """main function"""





    s = VideoInputStream(0)
    frame_id = 0

    frame_detections ={}

    while True:
        for im in s:
            assert im is not None

            timers = defaultdict(Timer)

            # cls_boxes, cls_segms, _ = im_detect_raw_masks(maskRCNN, im, timers=timers)


            # Window name in which image is displayed
            window_name = 'image'
              
            # Using cv2.imshow() method 
            # Displaying the image 
            cv2.imshow(window_name, im)

            # frame_detections[frame_id] = [boxes, class_ids, scores, masks]
            frame_id = frame_id + 1

    #closing all open windows 
    cv2.destroyAllWindows()

