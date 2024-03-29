import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow.keras.backend as K
import torch
import torch.backends.cudnn as cudnn
from tensorflow.keras.models import load_model

from models.experimental import attempt_load
from process_videos import ProcessVideos
from utils.datasets import LoadCamera, LoadImages, LoadStreams
from utils.general import (apply_classifier, check_img_size, check_imshow,
                           increment_path, non_max_suppression, scale_coords,
                           set_logging, strip_optimizer, xyxy2xywh)
from utils.plots import plot_one_box
from utils.torch_utils import (TracedModel, load_classifier, select_device,
                               time_synchronized)


class YoloV7:
    def TrackNet_Custom_Loss(self, y_true, y_pred):
        # Loss function
        loss = (-1) * (
            K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1))
            + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1))
        )
        return K.mean(loss)

    def TrackNet_Predict_Ball_Center(self, ratio, pred):
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        h_pred = pred[0] * 255
        h_pred = h_pred.astype("uint8")
        x_c_pred, y_c_pred = None, None
        if np.amax(h_pred[-1]) > 0:
            (cnts, _) = cv2.findContours(h_pred[-1].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            # 找到最大框選區域
            max_area = -1
            max_area_idx = None
            for j in range(len(rects)):
                area = rects[j][2] * rects[j][3]
                if area > max_area:
                    max_area_idx = j
                    max_area = area
            target = rects[max_area_idx]
            # 計算框選中心點
            (x_c_pred, y_c_pred) = (
                int(ratio * (target[0] + target[2] / 2)),
                int(ratio * (target[1] + target[3] / 2)),
            )
        return x_c_pred, y_c_pred

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

    def detect(self):
        source, weights, view_img, save_txt, imgsz, trace = (
            opt.source,
            opt.weights,
            opt.view_img,
            opt.save_txt,
            opt.img_size,
            not opt.no_trace,
        )

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != "cpu"  # half precision only supported on CUDA

        # Load model
        if opt.model_choices == 'yolo':
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size
            if trace:
                model = TracedModel(model, device, opt.img_size)
            if half:
                model.half()  # to FP16
        elif opt.model_choices == 'tracknet':
            sys.path.append("../12_in_12_out_pytorch")
            model = load_model(opt.tracknet_weights, custom_objects={"custom_loss": self.TrackNet_Custom_Loss})
            stride = None
            frame_height = 1080
            HEIGHT = 288  # model input size
            WIDTH = 512
            imgsz = (HEIGHT, WIDTH)
            ratio = frame_height / HEIGHT

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name="resnet101", n=2)  # initialize
            modelc.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"]).to(device).eval()

        # Set Dataloader
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadCamera(device, half, source, img_size=imgsz, stride=stride, model_choices = opt.model_choices, fps = int(opt.fps))
        process_video = ProcessVideos()

        # Get names and colors
        if opt.model_choices == 'yolo':
            names = model.module.names if hasattr(model, "module") else model.names
            # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            colors = [[158,66,3],[221,47,113],[86,104,193]] # 新聞記者的顏色

        # Run inference
        if opt.model_choices == 'yolo':
            if device.type != "cpu":
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            old_img_w = old_img_h = imgsz
            old_img_b = 1

        if view_img:
            cv2.namedWindow('Realtime Trajectory', cv2.WINDOW_NORMAL)

        # yolo detect
        for path, img, im0s, vid_cap, trajectory in dataset:
            t0 = time_synchronized()

            # Warmup
            if opt.model_choices == 'yolo':
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
                if opt.model_choices == 'yolo':
                    pred = model(img, augment=opt.augment)[0]
                elif opt.model_choices == 'tracknet':
                    pred = model.predict(img, batch_size=1)
            t2 = time_synchronized()

            # Apply NMS
            if opt.model_choices == 'yolo':
                pred = non_max_suppression(
                    pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms
                )
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = "Realtime", "", im0s.copy(), dataset.count

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                # most confidence
                most_confidence = -1
                most_confidence_ball_xyxy = None
                most_confidence_balls = []

                if opt.model_choices == 'yolo':
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                        for *xyxy, conf, cls in reversed(det):
                            if int(cls) != 0:
                                continue

                            # 判斷boundaries
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            if cls == 0: # ball boundaries
                                if opt.ball_top_boundary != "":
                                    numerator, denominator = map(int, opt.ball_top_boundary.split('/'))
                                    if xywh[1] < (numerator / denominator): # y軸在界線之上
                                        continue
                                if opt.ball_botton_boundary != "":
                                    numerator, denominator = map(int, opt.ball_botton_boundary.split('/'))
                                    if xywh[1] > (numerator / denominator): # y軸在界線之下
                                        continue

                            if conf > most_confidence:
                                most_confidence = conf
                                most_confidence_ball_xyxy = xyxy
                                most_confidence_balls.append([int(cls.item()), *xywh])

                        # 指畫出信心最高的那一顆球
                        if most_confidence != -1 and most_confidence_ball_xyxy != None:  # Add bbox to image
                            label = f"{names[int(0)]} {most_confidence:.2f}"
                            plot_one_box(most_confidence_ball_xyxy, im0, label=label, color=colors[int(0)], line_thickness=1)

                        # 取得球的位置
                        if most_confidence_balls:
                            if trajectory:
                                trajectory.Read_Yolo_Label_One_Frame(balls=most_confidence_balls)

                elif opt.model_choices == 'tracknet':
                    x_c_pred, y_c_pred = self.TrackNet_Predict_Ball_Center(ratio, pred)
                    trajectory.Read_Yolo_Label_One_Frame(x_c_pred = x_c_pred, y_c_pred = y_c_pred)

                # 畫出軌跡
                if trajectory:
                    image_CV = trajectory.Detect_Trajectory(im0)
                    trajectory.Add_Ball_In_Queue()
                    ball_direction, ball_direction_last = trajectory.Detect_Ball_Direction()
                    image_CV = trajectory.Draw_On_Image(image_CV, ball_direction)
                    trajectory.Next_Count()
                    if view_img:
                        cv2.imshow('Realtime Trajectory', image_CV)
                else:
                    if view_img:
                        cv2.imshow('Realtime Trajectory', im0)

                t4 = time_synchronized()

                # Print time (inference + NMS)
                print(f"{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, ({(1E3 * (t4 - t0)):.1f}ms) Total time, ({1.0 / (t4 - t0):.1f}) FPS")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov7.pt", help="model.pt path(s)")
    parser.add_argument("--source", type=str, default="inference/images", help="source")  # file/folder, 0 for webcam
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="display results")
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
    parser.add_argument("--ball-top-boundary", default="", help="ball boundary")
    parser.add_argument("--ball-botton-boundary", default="", help="ball boundary")
    parser.add_argument("--model-choices", default="yolo", help="yolo or tracknet")
    parser.add_argument("--tracknet-weights", default="../12_in_12_out_pytorch/weight/model_12_42/TN12model_best_acc", help="tracknet weights")
    parser.add_argument("--fps", default="60", help="fps")
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        yoloV7 = YoloV7()
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov7.pt"]:
                yoloV7.detect()
                strip_optimizer(opt.weights)
        else:
            yoloV7.detect()
