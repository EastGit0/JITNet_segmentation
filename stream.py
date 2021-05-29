import os
import cv2
import time
from random import randint

class VideoInputStream:
    def __init__(self, stream_path, start_frame = 0,
                 loop = False, reset_frame = 0):

        self.cap = cv2.VideoCapture(stream_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.loop = loop
        self.reset_frame = reset_frame
        self.start_frame = start_frame

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            if not self.loop:
                raise StopIteration()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.reset_frame)
                ret, frame = self.cap.read()

        return frame

    def __iter__(self):
        return self

    next = __next__