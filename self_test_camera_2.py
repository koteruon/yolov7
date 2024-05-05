import subprocess as sp
import time

import cv2
import ffmpeg
import numpy as np

# ffmpeg -re -stream_loop -1 -i /home/chaoen/yoloNhit_calvin/yolov7/inference/videos_long/0010.mp4 -map 0:v -f v4l2 /dev/video0


in_filename = "/home/siplab4/chaoen/yoloNhit_calvin/yolov7/inference/videos_long/6.mp4"

decklink_name = "DeckLink Quad HDMI Recorder (3)"
width = 1920
height = 1080

# process1 = (
#     ffmpeg
#     .input(decklink_name, format="decklink")
#     .output('pipe:', format='rawvideo', pix_fmt='rgb24')
#     .run_async(pipe_stdout=True)
# )


process1 = (
    ffmpeg
    .input(in_filename)
    .filter('fps', fps=60, round='up')
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)


avg_time = 0.0
t = time.time()
while(True):
    in_bytes = process1.stdout.read(width * height * 3)
    if not in_bytes:
        break
    in_frame = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([height, width, 3])
    )
    avg_time = avg_time * 0.5 + (time.time() - t) * 0.5
    t = time.time()
    # print(f"{1.0 / (avg_time):.1f} FPS")
