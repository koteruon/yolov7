import subprocess as sp
import time

import cv2
import ffmpeg
import numpy as np

# ffmpeg -re -stream_loop -1 -i /home/chaoen/yoloNhit_calvin/yolov7/inference/videos_long/0010.mp4 -map 0:v -f v4l2 /dev/video0


in_filename = "./inference/videos/0005.mp4"

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
    ffmpeg.input(in_filename)
    .filter("fps", fps=60, round="up")
    .output("pipe:", format="rawvideo", pix_fmt="rgb24")
    .run_async(pipe_stdout=True)
)

# cap = cv2.VideoCapture("/dev/video0")

# if not cap.isOpened():
#     print("Error: Could not open video file.")
#     exit()

avg_time = 0.0
t = time.time()
while True:
    in_bytes = process1.stdout.read(width * height * 3)
    if not in_bytes:
        break
    in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

    # ret, in_frame = cap.read()
    # if not ret:
    #     break

    in_frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("test", in_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
print(f"{time.time() - t} seconds")
