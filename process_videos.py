import argparse
import json
import os
import subprocess
import time
from multiprocessing import Pool

import pandas as pd
import tqdm
from cv2 import cv2


class ProcessVideos:
    def __init__(self) -> None:
        self.clip_root = r"/home/siplab/桌面/yoloNhit_calvin/HIT/data/table_tennis/clips/test/"
        self.midframe_root = r"/home/siplab/桌面/yoloNhit_calvin/HIT/data/table_tennis/keyframes/test/"

    def max_width_n_max_height(self, width, height, targ_size=360):
        if min(width, height) <= targ_size:
            new_width, new_height = width, height
        else:
            if height > width:
                new_width = targ_size
                new_height = int(round(new_width * height / width / 2) * 2)
            else:
                new_height = targ_size
                new_width = int(round(new_height * width / height / 2) * 2)
        return new_width, new_height

    def resize_image(
        self,
        frame,
        timestamp: int,
        targ_fps=30,
        targ_size=360,
    ):
        targ_dir = os.path.join(self.clip_root, "M-4")
        frame_targ_dir = os.path.join(self.midframe_root, "M-4")

        width, height, channels = frame.shape
        new_width, new_height = self.max_width_n_max_height(width, height, targ_size)
        frame = cv2.resize(frame, (new_width, new_height))

        clip_filename = f'{os.path.join(targ_dir, str(timestamp)+".jpg")}'
        cv2.imwrite(clip_filename, frame)
        if self.is_keyframe(timestamp):
            keyframe_filename = f'{os.path.join(frame_targ_dir, str(timestamp)+".jpg")}'
            cv2.imwrite(keyframe_filename, frame)
        return frame

    def is_keyframe(self, timestamp, targ_fps=5):
        return (timestamp + 1 + targ_fps // 2) % targ_fps == 0
