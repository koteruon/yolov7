import argparse
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from matplotlib.path import Path as matplotlib_path
from numpy import random

from keypoints_detection import KeyPointDetection
from models.experimental import attempt_load
from process_videos import ProcessVideos
from utils.datasets import LoadCamera, LoadImages, LoadStreams
from utils.general import (apply_classifier, check_img_size, check_imshow,
                           check_requirements, increment_path,
                           non_max_suppression, scale_coords, set_logging,
                           strip_optimizer, xyxy2xywh)
from utils.plots import plot_one_box
from utils.torch_utils import (TracedModel, load_classifier, select_device,
                               time_synchronized)


class YoloV7:
    def __init__(self):
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

    def detect(self, save_ori_img=False, only_ball=False, mark_no_count=False):
        source, weights, view_img, save_txt, imgsz, trace = (
            opt.source,
            opt.weights,
            opt.view_img,
            opt.save_txt,
            opt.img_size,
            not opt.no_trace,
        )
        save_img = not opt.nosave and not source.endswith(".txt")  # save inference images
        webcam = (
            source.isnumeric()
            or source.endswith(".txt")
            or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        )
        camera = bool(re.search("^/dev/video\d+", source))  # true if use camera

        # Directories
        if camera:
            save_dir = Path(opt.save_dir)  # specify directory to save
        else:
            save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name="resnet101", n=2)  # initialize
            modelc.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"]).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if camera:
            view_img = check_imshow()
            # view_img = False
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadCamera(source, img_size=imgsz, stride=stride)
            process_video = ProcessVideos()
        elif webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)


        # Get names and colors
        names = model.module.names if hasattr(model, "module") else model.names
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # colors = [[239,107,39],[125,209,71],[73,188,215]]
        colors = [[158, 66, 3], [221, 47, 113], [86, 104, 193]]  # 新聞記者的顏色

        # Run inference
        if device.type != "cpu":
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            if not self.has_mark_no_count_points and mark_no_count:
                self.Mark_No_Count_Point(im0s, im0s.shape[1], im0s.shape[0])
                self.has_mark_no_count_points = True

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

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

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if camera:
                    p, s, im0, frame = "M-4", "", im0s, dataset.count
                elif webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                if camera:
                    im0_origin = im0.copy()
                    clip_path = str(save_dir / "clips" / "test" / p.stem / f"{frame}") + ".jpg"  # 0.jpg
                    keyframe_path = str(save_dir / "keyframes" / "test" / p.stem / f"{frame}") + ".jpg"  # 0.jpg
                # Save original image
                if save_ori_img:
                    im0_origin = im0.copy()
                    clip_path = str(save_dir / "clips" / "test" / p.stem / f"{frame}") + ".jpg"  # 0.jpg
                    os.makedirs(os.path.dirname(clip_path), exist_ok=True)
                    cv2.imwrite(clip_path, im0_origin)

                txt_path = str(save_dir / "labels" / p.stem) + (
                    "" if dataset.mode == "image" else f"_{frame}"
                )  # img.txt

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    most_confidence = -1
                    most_confidence_ball_xyxy = None
                    for *xyxy, conf, cls in reversed(det):
                        if only_ball:
                            if int(cls) != 0:
                                continue

                        # 判斷boundaries
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        if cls == 0:  # ball boundaries
                            if opt.ball_top_boundary != "":
                                numerator, denominator = map(int, opt.ball_top_boundary.split("/"))
                                if xywh[1] < (numerator / denominator):  # y軸在界線之上
                                    continue
                            if opt.ball_botton_boundary != "":
                                numerator, denominator = map(int, opt.ball_botton_boundary.split("/"))
                                if xywh[1] > (numerator / denominator):  # y軸在界線之下
                                    continue
                            ball_center = [xywh[0], xywh[1]]
                            if self.has_mark_no_count_points and self.polygon_path.contains_point(ball_center):
                                continue
                        if cls == 1:  # person boundaries
                            if opt.person_left_boundary != "" and opt.person_right_boundary != "":
                                left_numerator, left_denominator = map(int, opt.person_left_boundary.split("/"))
                                right_numerator, right_denominator = map(int, opt.person_right_boundary.split("/"))
                                if xywh[0] > (left_numerator / left_denominator) and xywh[0] < (
                                    right_numerator / right_denominator
                                ):  # x軸在正中間的
                                    continue
                            if opt.person_top_boundary != "":
                                numerator, denominator = map(int, opt.person_top_boundary.split("/"))
                                if xywh[1] < (numerator / denominator):  # y軸在界線之上
                                    continue
                            if opt.person_botton_boundary != "":
                                numerator, denominator = map(int, opt.person_botton_boundary.split("/"))
                                if xywh[1] > (numerator / denominator):  # y軸在界線之下
                                    continue
                        if cls == 2:  # table boundaries
                            if opt.table_top_boundary != "":
                                numerator, denominator = map(int, opt.table_top_boundary.split("/"))
                                if xywh[1] < (numerator / denominator):  # y軸在界線之上
                                    continue
                            if opt.table_botton_boundary != "":
                                numerator, denominator = map(int, opt.table_botton_boundary.split("/"))
                                if xywh[1] > (numerator / denominator):  # y軸在界線之下
                                    continue

                        if opt.only_one_ball:
                            if int(cls) == 0:
                                if conf > most_confidence:
                                    most_confidence = conf
                                    most_confidence_ball_xyxy = xyxy

                        if save_txt:  # Write to file
                            if not opt.only_one_ball or int(cls) != 0:
                                xywh = (
                                    (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                )  # normalized xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + ".txt", "a") as f:
                                    f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or view_img:  # Add bbox to image
                            if not opt.only_one_ball or int(cls) != 0:
                                label = f"{names[int(cls)]} {conf:.2f}"
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    # 紀錄出信心最高的那一顆球
                    if (
                        opt.only_one_ball and save_txt and most_confidence != -1 and most_confidence_ball_xyxy != None
                    ):  # Add bbox to image
                        xywh = (
                            (xyxy2xywh(torch.tensor(most_confidence_ball_xyxy).view(1, 4)) / gn).view(-1).tolist()
                        )  # normalized xywh
                        line = (int(0), *xywh, most_confidence) if opt.save_conf else (int(0), *xywh)  # label format
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    # 指畫出信心最高的那一顆球
                    if (
                        opt.only_one_ball and most_confidence != -1 and most_confidence_ball_xyxy != None
                    ):  # Add bbox to image
                        label = f"{names[int(0)]} {most_confidence:.2f}"
                        plot_one_box(
                            most_confidence_ball_xyxy, im0, label=label, color=colors[int(0)], line_thickness=1
                        )

                # Print time (inference + NMS)
                print(f"{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS")

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 60, im0.shape[1], im0.shape[0]
                                save_path += ".mp4"
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer.write(im0)
                        if camera:  # save clip and keyframe frame to hit
                            frame = process_video.resize_image(im0, frame)

        if save_txt or save_img:
            s = (
                f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
                if save_txt
                else ""
            )
            # print(f"Results saved to {save_dir}{s}")

        print(f"Done. ({time.time() - t0:.3f}s)")


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
    parser.add_argument("--save-ori-img", action="store_true", help="save original image")
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
    parser.add_argument("--mark-no-count", action="store_true", help="mark no count")
    parser.add_argument("--ball-top-boundary", default="", help="ball boundary")
    parser.add_argument("--ball-botton-boundary", default="", help="ball boundary")
    parser.add_argument("--person-left-boundary", default="", help="person boundary")
    parser.add_argument("--person-right-boundary", default="", help="person boundary")
    parser.add_argument("--person-top-boundary", default="", help="person boundary")
    parser.add_argument("--person-botton-boundary", default="", help="person boundary")
    parser.add_argument("--table-top-boundary", default="", help="table boundary")
    parser.add_argument("--table-botton-boundary", default="", help="table boundary")
    parser.add_argument("--only-one-ball", action="store_true", help="predict ball only the hightest prediction")
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        yoloV7 = YoloV7()
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov7.pt"]:
                yoloV7.detect(save_ori_img=opt.save_ori_img, only_ball=opt.onlyball, mark_no_count=opt.mark_no_count)
                strip_optimizer(opt.weights)
        else:
            yoloV7.detect(save_ori_img=opt.save_ori_img, only_ball=opt.onlyball, mark_no_count=opt.mark_no_count)
