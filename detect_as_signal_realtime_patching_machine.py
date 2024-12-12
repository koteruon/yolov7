import argparse
import re
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from matplotlib.path import Path as matplotlib_path
from tqdm import tqdm

from models.experimental import attempt_load
from process_videos import ProcessVideos
from utils.datasets import LoadCamera, LoadImages, LoadStreams
from utils.general import (
    apply_classifier,
    check_img_size,
    check_imshow,
    increment_path,
    non_max_suppression,
    scale_coords,
    set_logging,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import plot_one_box
from utils.torch_utils import TracedModel, load_classifier, select_device, time_synchronized


class YoloV7:
    def __init__(self):
        self.mark_no_count = False
        self.has_mark_no_count_points = False

    def Draw_Circle(self, event, x, y, flags, param):
        # 用於透視變形取點
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(param["img"], (x, y), 3, (0, 255, 255), -1)
            param["point_x"].append(x)
            param["point_y"].append(y)

    def Mark_No_Count_Point(self, image, frame_width, frame_height):
        # 點選透視變形位置, 順序為:左上,左下,右下,右上
        PT_data = {"img": image.copy(), "point_x": [], "point_y": []}
        # TODO 測試用
        cv2.namedWindow("PIC2 (press Q to quit)", 0)
        cv2.resizeWindow("PIC2 (press Q to quit)", frame_width, frame_height)
        cv2.setMouseCallback("PIC2 (press Q to quit)", self.Draw_Circle, PT_data)
        while True:
            cv2.imshow("PIC2 (press Q to quit)", PT_data["img"])
            if cv2.waitKey(2) == ord("q"):
                print(PT_data)
                cv2.destroyWindow("PIC2 (press Q to quit)")
                break

        # PerspectiveTransform
        upper_left = [PT_data["point_x"][0] / frame_width, PT_data["point_y"][0] / frame_height]
        lower_left = [PT_data["point_x"][1] / frame_width, PT_data["point_y"][1] / frame_height]
        lower_right = [PT_data["point_x"][2] / frame_width, PT_data["point_y"][2] / frame_height]
        upper_right = [PT_data["point_x"][3] / frame_width, PT_data["point_y"][3] / frame_height]
        no_count_point = np.float32([upper_left, lower_left, lower_right, upper_right])
        self.polygon_path = matplotlib_path(no_count_point)

    def detect(self, only_ball=False):
        source, weights, view_img, save_txt, imgsz, trace = (
            opt.source,
            opt.weights,
            opt.view_img,
            opt.save_txt,
            opt.img_size,
            not opt.no_trace,
        )

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != "cpu"  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if trace:
            model = TracedModel(model, device, opt.img_size)
        if half:
            model.half()  # to FP16
        frame_height = 1080
        frame_width = 1920

        # Set Dataloader
        # view_img = check_imshow()
        cudnn.enabled = True  # set True to speed up constant image size inference
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadCamera(
            device,
            half,
            source,
            img_size=imgsz,
            stride=stride,
            model_choices=opt.model_choices,
            fps=int(opt.fps),
            height=frame_height,
            width=frame_width,
            opencv_or_ffmpeg=opt.opencv_or_ffmpeg,
            trajectory=False,
        )
        process_video = ProcessVideos()

        # Get names and colors
        names = model.module.names if hasattr(model, "module") else model.names
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        colors = [[158, 66, 3], [221, 47, 113], [86, 104, 193]]  # 新聞記者的顏色

        # Run inference
        if device.type != "cpu":
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        if view_img:
            cv2.namedWindow("Realtime Trajectory", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("Realtime Trajectory", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # recording
        is_recording = False
        vid_writer = None

        t4 = time_synchronized()
        # yolo detect
        for img, im0s, trajectory in dataset:
            if not self.has_mark_no_count_points and self.mark_no_count:
                self.Mark_No_Count_Point(im0s, im0s.shape[1], im0s.shape[0])
                self.has_mark_no_count_points = True

            # Warmup
            if device.type != "cpu" and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]
            ):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms
            )
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = "Realtime", "", im0s.copy(), dataset.count
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)
                lines = []
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    for *xyxy, conf, cls in reversed(det):
                        # 判斷boundaries
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        if opt.ball_top_boundary != "":
                            numerator, denominator = map(int, opt.ball_top_boundary.split("/"))
                            if xywh[1] < (numerator / denominator):  # y軸在界線之上
                                continue
                        if opt.ball_botton_boundary != "":
                            numerator, denominator = map(int, opt.ball_botton_boundary.split("/"))
                            if xywh[1] > (numerator / denominator):  # y軸在界線之下
                                continue
                        if self.mark_no_count:
                            ball_center = [xywh[0], xywh[1]]
                            if self.has_mark_no_count_points and self.polygon_path.contains_point(ball_center):
                                continue

                        lines.append((cls, *xywh, conf) if opt.save_conf else (cls, *xywh))  # label format
                        label = f"{names[int(0)]} {conf.item():.2f}"
                        plot_one_box(xyxy, im0, label=label, color=colors[int(0)], line_thickness=1)

                if opt.save_video and view_img:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("s"):
                        if not is_recording:
                            video_buffer = deque()
                            text_buffer = deque()
                            sub_save_path = Path(increment_path(save_path, exist_ok=opt.exist_ok))
                            sub_save_path.mkdir(parents=True, exist_ok=True)
                            video_path = str(sub_save_path / p.name) + ".mp4"
                            text_dir = Path(sub_save_path / "labels")
                            text_dir.mkdir(parents=True, exist_ok=True)
                            is_recording = True
                            record_frame = 1
                            fps, w, h = 60, im0.shape[1], im0.shape[0]
                            vid_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                            print("開始錄影...")

                    if key == ord("t"):
                        if is_recording:
                            for text in tqdm(list(text_buffer), desc="處理text資料"):
                                for txt_path, lines in text.items():
                                    with open(txt_path + ".txt", "a") as f:
                                        for line in lines:
                                            f.write(("%g " * len(line)).rstrip() % line + "\n")
                            for video in tqdm(list(video_buffer), desc="處理video資料"):
                                vid_writer.write(video)
                            vid_writer.release()
                            vid_writer = None
                            is_recording = False
                            print(f"錄影已儲存至 {video_path}")

                    if key == ord("k"):  # 按下 'k' 模擬 Ctrl+C
                        if vid_writer is not None:
                            vid_writer.release()
                        cv2.destroyAllWindows()
                        print("收到終止信號，結束程序...")
                        sys.exit(0)

                    if key == 32:
                        cv2.putText(im0, "PAUSE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
                        cv2.imshow("Realtime Trajectory", im0)
                        paused = True
                        while paused:
                            key = cv2.waitKey(1) & 0xFF
                            if key == 32:  # 再次按下空白鍵時恢復
                                paused = False
                                print("錄影已恢復...")
                            time.sleep(0.1)
                        time.sleep(1)

                    if is_recording:
                        cv2.putText(
                            im0, "RECORDING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                        )
                        video_buffer.append(im0s)
                        if lines:
                            txt_path = str(text_dir / p.stem) + f"_{record_frame}"
                            text_buffer.append({f"{txt_path}": lines})
                        record_frame += 1
                    else:
                        cv2.putText(
                            im0, "STAND BY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (34, 139, 34), 2, cv2.LINE_AA
                        )
                    cv2.imshow("Realtime Trajectory", im0)

                # Print time (inference + NMS)
                print(
                    f"{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, ({(1E3 * (time_synchronized() - t4)):.1f}ms) Total time, ({1.0 / (time_synchronized() - t4):.1f}) FPS"
                )

                t4 = time_synchronized()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov7.pt", help="model.pt path(s)")
    parser.add_argument("--source", type=str, default="inference/images", help="source")  # file/folder, 0 for webcam
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-video", action="store_true", help="display results")
    parser.add_argument(
        "--save-dir", default="/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis", help="save results directory"
    )
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --class 0, or --class 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default="runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--no-trace", action="store_true", help="don`t trace model")
    parser.add_argument("--onlyball", action="store_true", help="plot only ball")
    parser.add_argument("--ball-top-boundary", default="", help="ball boundary")
    parser.add_argument("--ball-botton-boundary", default="", help="ball boundary")
    parser.add_argument("--person-left-boundary", default="", help="person boundary")
    parser.add_argument("--person-right-boundary", default="", help="person boundary")
    parser.add_argument("--person-top-boundary", default="", help="person boundary")
    parser.add_argument("--person-botton-boundary", default="", help="person boundary")
    parser.add_argument("--table-top-boundary", default="", help="table boundary")
    parser.add_argument("--table-botton-boundary", default="", help="table boundary")
    parser.add_argument("--model-choices", default="yolo", help="yolo or tracknet")
    parser.add_argument(
        "--tracknet-weights",
        default="../12_in_12_out_pytorch/weight/model_12_42/TN12model_best_acc",
        help="tracknet weights",
    )
    parser.add_argument("--fps", default="60", help="fps")
    parser.add_argument("--opencv-or-ffmpeg", default="opencv", help="opencv or ffmpeg")
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        yoloV7 = YoloV7()
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov7.pt"]:
                yoloV7.detect(only_ball=opt.onlyball)
                strip_optimizer(opt.weights)
        else:
            yoloV7.detect(only_ball=opt.onlyball)
