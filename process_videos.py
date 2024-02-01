import argparse
import json
import os
import subprocess
import time
from multiprocessing import Pool

import cv2
import pandas as pd
import tqdm


class ProcessVideos:
    def __init__(self) -> None:
        self.clip_root_origin = r"/home/siplab4/chaoen/yoloNhit_calvin/HIT/data/table_tennis/clips_ori/test/"
        self.clip_root = r"/home/siplab4/chaoen/yoloNhit_calvin/HIT/data/table_tennis/clips/test/"
        self.midframe_root = r"/home/siplab4/chaoen/yoloNhit_calvin/HIT/data/table_tennis/keyframes/test/"

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
        targ_fps=60,
        targ_size=360,
    ):
        ori_dir = os.path.join(self.clip_root_origin, "M-4")
        targ_dir = os.path.join(self.clip_root, "M-4")
        frame_targ_dir = os.path.join(self.midframe_root, "M-4")

        width, height, channels = frame.shape
        new_width, new_height = self.max_width_n_max_height(width, height, targ_size)

        # save origin image
        clip_origin_filename = f'{os.path.join(targ_dir, str(timestamp)+".jpg")}'
        cv2.imwrite(clip_origin_filename, frame)

        # resize image
        clip_filename = f'{os.path.join(targ_dir, str(timestamp)+".jpg")}'
        frame = cv2.resize(frame, (new_width, new_height))
        cv2.imwrite(clip_filename, frame)
        if self.is_keyframe(timestamp):
            keyframe_filename = f'{os.path.join(frame_targ_dir, str(timestamp)+".jpg")}'
            cv2.imwrite(keyframe_filename, frame)
        return frame

    def is_keyframe(self, timestamp, targ_fps=1):
        return (timestamp + 1 + targ_fps // 2) % targ_fps == 0
