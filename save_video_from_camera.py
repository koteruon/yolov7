import os
from datetime import datetime

import cv2

cap = cv2.VideoCapture("/dev/video0")

if not cap.isOpened():
    print("streaming is not supported")
    exit()

recording = False
cap.set(cv2.CAP_PROP_FPS, 60)


file_list = os.listdir("./inference/videos_tmp")
filename = []
if len(file_list) != 0:
    for file in file_list:
        filename.append(int(file.split(".")[0]))
    count = sorted(filename)[-1] + 1
else:
    count = 0
print(count)

while True:
    ret, frame = cap.read()

    if not ret:
        print("can't read frame")
        break

    framerate = int(cap.get(cv2.CAP_PROP_FPS))

    frame_s = cv2.resize(frame, (640, 320))
    cv2.imshow("Video Stream", frame_s)

    key = cv2.waitKey(1) & 0xFF
    now = datetime.now().strftime("%H:%M:%S")

    if key == ord(" "):  # 按下空格键开始/停止录制
        if not recording:
            recording = True
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            out = cv2.VideoWriter(
                f"./inference/videos_tmp/{count}.mp4", fourcc, framerate, (frame.shape[1], frame.shape[0])
            )
            print("start recording")
        else:
            recording = False
            out.release()
            os.system(f"cp ./inference/videos_tmp/{count}.mp4 ./inference/videos/{count}.mp4")
            count += 1
            print("stop recording")

    if now == "09:55:00":
        if not recording:
            recording = True
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            out = cv2.VideoWriter(
                f"./inference/videos_tmp/{count}.mp4", fourcc, framerate, (frame.shape[1], frame.shape[0])
            )
            print("start recording")

    if recording:
        out.write(frame)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
