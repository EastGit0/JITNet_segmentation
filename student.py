import cv2
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from utils.helpers import colorize_mask
import matplotlib.pyplot as plt
# from models.jitnet import JITNet
# from models.jitnetlight import JITNetLight
from paramiko import SSHClient
from scp import SCPClient, SCPException
from stream import VideoInputStream
import json
import models
import time
# from scipy.signal import medfilt
from scipy.ndimage import median_filter

import multiprocessing

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
# cv2.ocl.setUseOpenCL(False)

class MailBox_Process(multiprocessing.Process):
    def __init__(self, id, queue):
        super(MailBox_Process, self).__init__()
        self.id = id
        self.queue = queue
                
    def run(self):
        self.queue.put(True)
        while True:
            image, frame_id = self.queue.get()
            t1 = time.time()
            Student.turn_in_homework(image, frame_id)

            t2 = time.time()
            print("\nProcess 1: Transfer Frame")

class Student():
    def __init__(self, model_path, config):
        
        if torch.cuda.is_available():
            print("Running JITNet on GPU!")
        else:
            print("Running JITNet on CPU!")
        
        self.config = config
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.43931922, 0.41310471, 0.37480941], [0.24272706, 0.23649098, 0.23429529])

        num_classes = 81
        if self.config['arch']['type'] == "JITNet":
          print("Using JITNet Model")
          # self.model = JITNet(num_classes)
        else:
          print("Using JITNetLight Model")
          # self.model = JITNetLight(num_classes)
        self.model = getattr(models, self.config['arch']['type'])(num_classes, **config['arch']['args'])

        print(f'\n{self.model}\n')

        self.availble_gpus = list(range(torch.cuda.device_count()))
        self.device = torch.device('cuda:0' if len(self.availble_gpus) > 0 else 'cpu')

        self.load_weights(model_path)

        # Set up SSH
        self.ssh = SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.connect('35.233.229.168')
        self.scp = SCPClient(self.ssh.get_transport())

        self.frame_id = 1
        self.window_name = "Webcam"

        self.next_weight_id = 1
        self.next_weight_path = "saved/teacher_weights/weights_{}".format(str(self.next_weight_id))


    def load_weights(self, path):
        print("---------- LOADING WEIGHTS: {} ---------".format(path))
        self.checkpoint = torch.load(path, map_location=self.device)

        if isinstance(self.checkpoint, dict) and 'state_dict' in self.checkpoint.keys():
            self.checkpoint = self.checkpoint['state_dict']
        if 'module' in list(self.checkpoint.keys())[0] and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        self.model.load_state_dict(self.checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()


    def turn_in_homework(self, image, mask):
        frame_name = "saved/stream_outputs/{}.jpg".format("frame_" + str(self.frame_id))
        mask_name = "saved/stream_outputs/{}.png".format("prediction_" + str(self.frame_id))

        # Save Frame and Mask
        cv2.imwrite(frame_name, image) # frame
        cv2.imwrite(mask_name, mask) #mask
        
        # Send Frame and Mask
        t1 = time.time()
        self.scp.put(frame_name, remote_path='/home/cs348k/data/student/frames')
        t2 = time.time()
        self.scp.put(mask_name, remote_path='/home/cs348k/data/student/predictions')
        t3 = time.time()

        # print("\nTransfer Time Frame Color: ", t2-t1)
        # print("Transfer Time Mask: ", t3-t2)
        
        # delete frame and mask (no need to accumulate masks and frames)
        os.system("rm {}".format(frame_name))
        os.system("rm {}".format(mask_name))

    def video_stream(self):
        """main function"""

        multiprocessing.set_start_method("spawn", force=True)

        ## Loop Over Video Stream
        s = VideoInputStream(0)

        queue = multiprocessing.Queue()
        mailbox = MailBox_Process(0, queue)
        mailbox.start()

        # Wait till transfer process ready
        queue.get()

        cv2.startWindowThread()
        cv2.namedWindow(self.window_name)

        # while True:
        try:
            for im in s:
                # Timing Data
                start_time = time.time()
                assert im is not None
                self.frame_id = self.frame_id + 1

                ##### Send Frame and Mask #####
                queue.put(im, self.frame_id)

                ##### Make Prediction #####
                input = self.normalize(self.to_tensor(im)).unsqueeze(0)
                prediction = self.model(input.to(self.device))
                with torch.no_grad():
                    prediction = prediction[0].squeeze(0).numpy()
                
                ## Apply Softmax
                # prediction = F.softmax(torch.from_numpy(prediction), dim=0)
                prediction = np.exp(prediction)
                prediction = prediction / np.sum(prediction, axis=0, keepdims=False)
                
                ## Thresholding post processing
                prediction = (prediction[0,:,:])
                # prediction = median_filter(prediction, (15, 15), mode='constant', cval=0)
                threshold = 0.3
                low = np.where(prediction < threshold)
                high = np.where(prediction >= threshold)
                prediction[low] = 0
                prediction[high] = 255

                ##### Display new Frame #####
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im[low] = 0
                cv2.imshow(self.window_name, im) #prediction.astype(np.uint8)
                

                ##### Check for New Weights #####
                # try:
                #     # scp_time_1 = time.time()
                #     # self.scp.get(local_path=self.next_weight_path, remote_path=("/home/cs348k/data/student/weights/{}/weights_{}.pth".format(self.config['arch']['type'], str(self.next_weight_id))))
                #     # scp_time_2 = time.time()
                # except SCPException:
                #     pass
                    # scp_time_2 = time.time()

                if os.path.exists(self.next_weight_path):
                  weights_time_1 = time.time()
                  self.load_weights(self.next_weight_path)
                  os.system("rm {}".format(self.next_weight_path))
                  self.next_weight_id = self.next_weight_id + 1
                  self.next_weight_path = "saved/teacher_weights/weights_{}".format(str(self.next_weight_id))
                  weights_time_2 = time.time()
                  print("Load Weights Time: ", weights_time_2 - weights_time_1)
                else:
                  cv2.waitKey(5)


                end_time = time.time()

                print("Process 0: Seconds Per Frame = ", end_time-start_time)


        except KeyboardInterrupt:
            print("\nExiting...")
            self.scp.close()
            sys.exit(0)

             

        #### Never Reach Here ####
        # plt.ioff()
        # plt.show()

        #closing all open windows 
        # cv2.destroyAllWindows()

def main(config):
    # Set Up Student
    config = json.load(open(args.config))
    student = Student(model_path=args.model_path, config=config)

    torch.cuda.empty_cache()

    student.video_stream()


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-mp', '--model_path', default=None,type=str,
                        help='Path to the starting Model')
    parser.add_argument('-c','--config', default='config_student.json',type=str,
                        help='Use a custom model, eg JITNetLight')

    args = parser.parse_args()

    main(args)
