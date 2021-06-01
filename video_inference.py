import cv2
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from utils.helpers import colorize_mask
import matplotlib.pyplot as plt
from models.jitnet import JITNet
from models.jitnetlight import JITNetLight
from paramiko import SSHClient
from scp import SCPClient
from stream import VideoInputStream

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
# cv2.ocl.setUseOpenCL(False)

class Student():
    def __init__(self, model_path, model_JITNet):
        
        if torch.cuda.is_available():
            print("Running JITNet on GPU!")
        else:
            print("Running JITNet on CPU!")
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.43931922, 0.41310471, 0.37480941], [0.24272706, 0.23649098, 0.23429529])

        num_classes = 81
        if model_JITNet:
          print("Using JITNet Model")
          self.model = JITNet(num_classes)
        else:
          print("Using JITNetLight Model")
          self.model = JITNetLight(num_classes)

        self.availble_gpus = list(range(torch.cuda.device_count()))
        self.device = torch.device('cuda:0' if len(self.availble_gpus) > 0 else 'cpu')

        self.load_weights(model_path)

        # Set up SSH
        self.ssh_frame = SSHClient()
        self.ssh_frame.load_system_host_keys()
        self.ssh_frame.connect('35.233.229.168')

        self.ssh_mask = SSHClient()
        self.ssh_mask.load_system_host_keys()
        self.ssh_mask.connect('35.233.229.168')

        self.frame_id = 0
        self.window_name = "Steam"

        self.next_weight_id = 1
        self.next_weight_path = "./teacher_weights/weights_{}".format(str(self.next_weight_id))

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

        # Save Frame and Mask

        # Send Frame and Mask
        with SCPClient(self.ssh_frame.get_transport()) as scp:
            scp.put("{}.jpg".format(self.frame_id))

        with SCPClient(self.ssh_mask.get_transport()) as scp:
            scp.put("{}.png".format(self.frame_id))


    def video_stream(self):
        """main function"""

        ## Loop Over Video Stream
        s = VideoInputStream(0)

        # ax1 = plt.subplot(1,1,1)
        # plt.show()

        # Window name in which image is displayed

        cv2.startWindowThread()
        cv2.namedWindow(self.window_name)

        while True:
            for im in s:
                assert im is not None
                self.frame_id = self.frame_id + 1

                ##### Make Prediction #####
                # input = normalize(to_tensor(im.convert('RGB'))).unsqueeze(0)
                input = self.normalize(self.to_tensor(im)).unsqueeze(0)
                prediction = self.model(input.to(self.device))
                prediction = prediction[0].squeeze(0).cpu().detach().numpy()
                prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

                prediction[np.where(prediction == 0)] = 1
                prediction[np.where(prediction >= 1)] = 0
                ##### Send Frame and Mask #####
#                self.turn_in_homework(im, prediction)

                ##### Display new Frame #####
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                print(prediction)
                cv2.imshow(self.window_name, output_im) #prediction.astype(np.uint8)
                

                ##### Check for New Weights #####
                if os.path.exists(self.next_weight_path):
                  self.load_weights(self.next_weight_path)
                  self.next_weight_id = self.next_weight_id + 1
                  self.next_weight_path = "./teacher_weights/weights_{}".format(str(self.next_weight_id))
                else:
                  cv2.waitKey(100)


                # if frame_id == 0:
                #     window = ax1.imshow(im)
                #     plt.ion()
                # Using cv2.imshow() method 
                # Displaying the image 


                # window.set_data(im)
                # plt.pause(0.04)
                # plt.show()

             

        #### Never Reach Here ####
        # plt.ioff()
        # plt.show()

        #closing all open windows 
        # cv2.destroyAllWindows()

def main(config):
    # Set Up Student
    student = Student(model_path=args.model_path, model_JITNet = (not args.not_JITNet))

    torch.cuda.empty_cache()

    student.video_stream()


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-mp', '--model_path', default=None,type=str,
                        help='Path to the starting Model')
    parser.add_argument('--not_JITNet', action="store_true",
                        help='Use a custom model, eg JITNetLight')

    args = parser.parse_args()

    main(args)
