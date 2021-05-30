from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
from collections import defaultdict
import time
import cv2
import argparse
import scipy
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
import matplotlib.pyplot as plt

from stream import VideoInputStream

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
# cv2.ocl.setUseOpenCL(False)






def worker_stream():
    """main function"""


    if torch.cuda.is_available():
        print("Running JITNet on GPU!")
    else:
        print("Running JITNet on CPU!")


     # Dataset used for training the model
    # dataset_type = config['train_loader']['type']
    # assert dataset_type in ['DETECTRON', 'COCO']
    # scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    # loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    # to_tensor = transforms.ToTensor()
    # normalize = transforms.Normalize(loader.MEAN, loader.STD)
    # num_classes = loader.dataset.num_classes
    # palette = loader.dataset.palette

    # # Model
    # model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    # availble_gpus = list(range(torch.cuda.device_count()))
    # device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # checkpoint = torch.load(args.model, map_location=device)
    # #print("Checkpoint: ", checkpoint)
    # if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
    #     checkpoint = checkpoint['state_dict']
    #     #print("Update state dict: ", checkpoint)
    # if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
    #     model = torch.nn.DataParallel(model)
    #     #print("Update parallel: ", model)
    # model.load_state_dict(checkpoint, strict=False)
    # model.to(device)
    # model.eval()

    # if not os.path.exists('outputs'):
    #     os.makedirs('outputs')

    # image_files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
    # with torch.no_grad():
    #     tbar = tqdm(image_files, ncols=100)
    #     for img_file in tbar:
    #         image = Image.open(img_file).convert('RGB')
    #         input = normalize(to_tensor(image)).unsqueeze(0)

    #         prediction = model(input.to(device))
    #         prediction = prediction[0].squeeze(0).cpu().numpy()

    #         prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
    #         save_images(image, prediction, args.output, img_file, palette)



    ## Loop Over Video Stream
    s = VideoInputStream(0)
    frame_id = 0

    ax1 = plt.subplot(1,1,1)

    # Window name in which image is displayed
    window_name = "Steam"

    # cv2.startWindowThread()
    # cv2.namedWindow(window_name)
    # plt.show()

    frame_detections ={}

    while True:
        for im in s:
            assert im is not None
            if frame_id == 0:
                window = ax1.imshow(im)
                plt.ion()
            # Using cv2.imshow() method 
            # Displaying the image 
            # cv2.imshow(window_name, im)
            window.set_data(im)
            plt.pause(0.04)
            # plt.show()

            # frame_detections[frame_id] = [boxes, class_ids, scores, masks]
            frame_id = frame_id + 1

    plt.ioff()
    plt.show()

    #closing all open windows 
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    worker_stream()