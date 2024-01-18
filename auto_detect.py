import os
import time

now_count = 0
filename = []
while True:
    file_list = os.listdir("./inference/videos")
    if len(file_list) != 0:
        for file in file_list:
            filename.append(int(file.split(".")[0]))
        count = sorted(filename)[-1]
    else:
        count = 0

    time.sleep(1)
    print(now_count, count, time.strftime("%Y-%m-%d %H:%M:%S"))

    if now_count < count:
        os.system(
            f"python detect.py --weights ./weights/yolov7x_3classes.pt --conf 0.5 --img-size 960 --source ./inference/videos/{count}.mp4 --save-txt --save-conf --project runs/detect/ --name yolov7_20231023_{count} --onlyball"
        )
        os.system(
            f"python predict_as_signal_only_ball_add_param.py --root_path ./runs/detect/yolov7_20231023_{count} --video_name {count}.mp4"
        )
        now_count += 1
