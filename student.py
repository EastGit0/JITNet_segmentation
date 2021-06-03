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


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
# cv2.ocl.setUseOpenCL(False)

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
        # if self.config['arch']['type'] == "JITNet":
        #   print("Using JITNet Model")
        #   self.model = JITNet(num_classes)
        # else:
        #   print("Using JITNetLight Model")
        #   self.model = JITNetLight(num_classes)
        self.model = getattr(models, self.config['arch']['type'])(num_classes, **config['arch']['args'])

        self.availble_gpus = list(range(torch.cuda.device_count()))
        self.device = torch.device('cuda:0' if len(self.availble_gpus) > 0 else 'cpu')

        self.load_weights(model_path)

        # Set up SSH
        self.ssh = SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.connect('35.233.229.168')
        self.scp = SCPClient(self.ssh.get_transport())

        # self.ssh_mask = SSHClient()
        # self.ssh_mask.load_system_host_keys()
        # self.ssh_mask.connect('35.233.229.168')
        # self.scp_mask = SCPClient(self.ssh_mask.get_transport())

        self.frame_id = 0
        self.window_name = "Steam"

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
        # tensor_name = "saved/stream_outputs/{}.pt".format("prediction_" + str(self.frame_id))

        # Save Frame and Mask
        cv2.imwrite(frame_name, image) # frame
        cv2.imwrite(mask_name, mask) #mask
        # tensor_mask = self.to_tensor(mask)
        # torch.save(tensor_mask, tensor_name)
        
        # Send Frame and Mask
        self.scp.put(frame_name, remote_path='/home/cs348k/data/student/frames')
        self.scp.put(mask_name, remote_path='/home/cs348k/data/student/predictions')
        # self.scp.put(tensor_name, remote_path='/home/cs348k/data/student/predictions')
        
        # delete frame and mask (no need to accumulate masks and frames)
        os.system("rm {}".format(frame_name))
        os.system("rm {}".format(mask_name))
        # os.system("rm {}".format(tensor_name))

    def video_stream(self):
        """main function"""

        ## Loop Over Video Stream
        s = VideoInputStream(0)

        # ax1 = plt.subplot(1,1,1)
        # plt.show()

        # Window name in which image is displayed

        cv2.startWindowThread()
        cv2.namedWindow(self.window_name)

        # while True:
        try:
            for im in s:
                assert im is not None
                self.frame_id = self.frame_id + 1

                ##### Make Prediction #####
                start_time = time.time()
                # input = normalize(to_tensor(im.convert('RGB'))).unsqueeze(0)
                input = self.normalize(self.to_tensor(im)).unsqueeze(0)
                prediction = self.model(input.to(self.device))


                if 0:
                    with torch.no_grad():
                        probs = F.softmax(torch.from_numpy(prediction), dim=1)
                        print(probs.shape)
                        probs_max, preds = torch.max(probs, dim=1) # [B, H, W]
                        print(probs_max.shape)
                        print(preds.shape)








                else:
                    end_time_1 = time.time()
                    prediction = prediction[0].squeeze(0).cpu().detach().numpy()
                    # print(prediction)
                    end_time_2 = time.time()

                    prediction = F.softmax(torch.from_numpy(prediction), dim=0)

                    end_time_3 = time.time()
                    print("TIME: ", end_time_3 - end_time_2)
                    # print(prediction[0,:,:])
                    # print(prediction[1,:,:])

                    person = (prediction[1,:,:] * 128).numpy()
                    item_12 = (prediction[12,:,:] * 255).numpy()
                    background = (prediction[0,:,:]).numpy()
                    super_background = background.copy()
                    summed = item_12 + background
                    print("Max SUPER BACKGROUND: ", np.amax(super_background))
                    super_background[np.where(super_background < .92)] = 0
                    cv2.imshow("Person", person)
                    cv2.imshow("Item 12", item_12)
                    cv2.imshow("Background", background)
                    cv2.imshow("Super_Background", super_background)
                    cv2.imshow("Summed", summed)
                    prediction = prediction.argmax(0)
                    # prediction = torch.max(prediction, dim=0)
                    print(prediction)
                    prediction = prediction.numpy()
                    cv2.imshow("argmax", prediction.astype(np.uint8))
                    print(np.unique(prediction))
                
                
                cv2.waitKey(10000)
                continue
                exit()
                # end_time_3 = time.time()

                

                ##### Send Frame and Mask #####
                self.turn_in_homework(im, prediction)
                end_time_4 = time.time()

                ##### Display new Frame #####
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im[np.where(prediction >= 1)] = 0
                cv2.imshow(self.window_name, im) #prediction.astype(np.uint8)
                end_time_5 = time.time()

                ##### Check for New Weights #####
                try:
                    scp_time_1 = time.time()
                    self.scp.get(local_path=self.next_weight_path, remote_path=("/home/cs348k/data/student/weights/{}/weights_{}.pth".format(self.config['arch']['type'], str(self.next_weight_id))))
                    scp_time_2 = time.time()
                except SCPException:
                    # print("No Weights!")
                    pass

                if os.path.exists(self.next_weight_path):
                  self.load_weights(self.next_weight_path)
                  os.system("rm {}".format(self.next_weight_path))
                  self.next_weight_id = self.next_weight_id + 1
                  self.next_weight_path = "saved/teacher_weights/weights_{}".format(str(self.next_weight_id))
                else:
                  cv2.waitKey(100)


                print("Prediction Time 1: ", end_time_1 - start_time)
                print("Prediction Time 2: ", end_time_2 - end_time_1)
                print("Prediction Time 3: ", end_time_3 - end_time_2)
                print("Prediction Time 4: ", end_time_4 - end_time_3)
                print("Prediction Time 5: ", end_time_5 - end_time_4)
                print("Prediction Time 6: ", scp_time_2 - scp_time_1)
                print("\n")

                # if frame_id == 0:
                #     window = ax1.imshow(im)
                #     plt.ion()
                # Using cv2.imshow() method 
                # Displaying the image 


                # window.set_data(im)
                # plt.pause(0.04)
                # plt.show()
        except KeyboardInterrupt:
            print("\nExiting...")
            self.scp.close()
            # self.scp_mask.close()
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
    # parser.add_argument('--not_JITNet', action="store_true",
    #                     help='Use a custom model, eg JITNetLight')

    args = parser.parse_args()

    main(args)
