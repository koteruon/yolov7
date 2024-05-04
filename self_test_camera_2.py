import subprocess as sp
import time

import cv2
import ffmpeg
import numpy as np

# ffmpeg -re -stream_loop -1 -i /home/chaoen/yoloNhit_calvin/yolov7/inference/videos_long/0010.mp4 -map 0:v -f v4l2 /dev/video0

# FFmpeg record=================
def videoFFmpeg(fileOut,framerateVideo):
    video_ = ffmpeg.input('video=screen-capture-recorder',
                          thread_queue_size=2048,
                          rtbufsize='2048M',
                          pixel_format='bgr24',
                          framerate=framerateVideo,
                          f='dshow'
                          )

    v2 = ffmpeg.filter(video_, 'scale',in_range='full',out_range='full',eval='init',interl='false',flags='bitexact+accurate_rnd+full_chroma_int').filter('fps', fps=framerateVideo).filter('pp', 'fa').filter('crop', 1920,1040,0,0)
    out = ffmpeg.output(v2, fileOut, acodec='copy', vcodec="libx264", preset='ultrafast',
                        tune='film', crf=17, r=framerateVideo, force_key_frames='expr:gte(t,n_forced*1)',
                        sc_threshold=0, pix_fmt='yuv420p', max_muxing_queue_size=2048,
                        start_at_zero=None)


    out = out.global_args('-hide_banner')
    out = out.global_args('-hwaccel_device', 'auto')
    out = out.global_args('-hwaccel', 'auto')
    # out = out.global_args('-report')
    out = out.overwrite_output()

    process = out.run_async(pipe_stdin=True)
    return process

# Quit  FFmpeg ==================
def QuitFFmpeg(process):
    # print(1)
    process.communicate(str.encode("q"))

    time.sleep(3)
    process.terminate()
    return 'Exit'


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
