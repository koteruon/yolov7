import argparse
import heapq
import json
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from utils.datasets import letterbox
from utils.general import bbox_iou, non_max_suppression_kpt
from utils.plots import output_to_keypoint


class KeyPointDetection:
    def __init__(self, is_train=False, process_videos=None, args=None):
        if args:
            self.args = args
        else:
            self.args = SimpleNamespace()
            self.args.person_left_boundary == ""
            self.args.person_right_boundary == ""
            self.args.person_top_boundary == ""
            self.args.person_botton_boundary == ""

        # 輸出的結果
        self.all_outputs = {}

        # load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weigths = torch.load("weights/yolov7-w6-pose.pt", map_location=self.device)
        self.model = weigths["model"]
        _ = self.model.float().eval()
        if torch.cuda.is_available():
            self.model.half().to(self.device)

        # 輸出檔案位置
        if is_train:
            self.root = r"../HIT/data/table_tennis/keyframes/train/"
            self.json_path = r"../HIT/data/table_tennis/annotations/table_tennis_train_person_bbox_kpts.json"
            self.person_bbox_path = r"../HIT/data/table_tennis/boxes/table_tennis_train_det_person_bbox.json"
        else:
            self.root = r"../HIT/data/table_tennis/keyframes/test/"
            self.json_path = r"../HIT/data/table_tennis/annotations/table_tennis_test_person_bbox_kpts.json"
            self.person_bbox_path = r"../HIT/data/table_tennis/boxes/table_tennis_test_det_person_bbox.json"

        self.process_videos = process_videos

        self.pose_bbox_threshold = 0.25
        if os.path.exists(self.person_bbox_path):
            self.person_bbox = {}
            with open(self.person_bbox_path, "r") as file:
                data = json.load(file)
                for entry in data:
                    video_id = entry["video_id"]
                    image_id = entry["image_id"]
                    if video_id not in self.person_bbox:
                        self.person_bbox[video_id] = {}
                    if image_id not in self.person_bbox[video_id]:
                        self.person_bbox[video_id][image_id] = []
                    self.person_bbox[video_id][image_id].append(entry)
        else:
            self.person_bbox = None

    def detect_one(self, timestamp, root_idx=0, root_dir="M-4"):
        if int(timestamp) > 100000:
            raise Exception("timestamp greater than 100000")
        im = cv2.imread(os.path.join(self.root, root_dir, "{}.jpg".format(timestamp)))
        origin_height = im.shape[0]
        origin_width = im.shape[1]

        im = letterbox(im, 960, stride=64, auto=True)[0]
        im = transforms.ToTensor()(im)
        im = torch.tensor(np.array([im.numpy()]))

        if torch.cuda.is_available():
            im = im.half().to(self.device)

        yolo_height = im.shape[2]
        yolo_width = im.shape[3]

        output, _ = self.model(im)
        output = non_max_suppression_kpt(
            output, 0.25, 0.65, nc=self.model.yaml["nc"], nkpt=self.model.yaml["nkpt"], kpt_label=True
        )
        output = output_to_keypoint(output)

        # print(yolo_height, yolo_width)
        # print(origin_height, origin_width)
        h_ratio = origin_height / yolo_height
        w_ratio = origin_width / yolo_width


        if self.person_bbox and root_dir in self.person_bbox and int(timestamp) in self.person_bbox[root_dir]:
            bbox_frames = self.person_bbox[root_dir][int(timestamp)]
            target_bbox = torch.tensor([bbox_frame["bbox"] for bbox_frame in bbox_frames])
            number_of_person = len(target_bbox)
        else:
            target_bbox = torch.tensor([])
            number_of_person = 0

        coco_outputs = []
        for idx in range(output.shape[0]):
            # 去除x軸在中間的裁判
            if self.args.person_left_boundary != "" and self.args.person_right_boundary != "":
                left_numerator, left_denominator = map(int, self.args.person_left_boundary.split("/"))
                right_numerator, right_denominator = map(int, self.args.person_right_boundary.split("/"))
                if output[idx, 2] >= 960 * (left_numerator / left_denominator) and output[idx, 2] <= 960 * (
                    right_numerator / right_denominator
                ):
                    continue
            if self.args.person_top_boundary != "":
                numerator, denominator = map(int, self.args.person_top_boundary.split("/"))
                if output[idx, 3] < 576 * (numerator / denominator):  # y軸在界線之上
                    continue
            if self.args.person_botton_boundary != "":
                numerator, denominator = map(int, self.args.person_botton_boundary.split("/"))
                if output[idx, 3] > 576 * (numerator / denominator):  # y軸在界線之下
                    continue

            category_id = int(output[idx][1])  ## cls

            bbox = output[idx][2:6]
            bbox[0::2] = bbox[0::2] * w_ratio
            bbox[1::2] = bbox[1::2] * h_ratio

            if number_of_person > 0:
                pose_bbox = torch.tensor(bbox)
                iou = bbox_iou(pose_bbox, target_bbox, x1y1x2y2=True, CIoU=True)
                if torch.all(iou < self.pose_bbox_threshold):
                    continue
                max_iou = iou.max()
            else:
                max_iou = 0.0

            keypoints = output[idx][7:]
            keypoints[0::3] = keypoints[0::3] * w_ratio
            keypoints[1::3] = keypoints[1::3] * h_ratio

            image_id = int("{}".format(timestamp)) + (100000 * root_idx)
            coco_outputs.append({
                "image_id": image_id,
                "category_id": 1 if category_id == 0 else category_id,
                "bbox": bbox.tolist(),
                "keypoints": keypoints.reshape(-1, 3).tolist(),
                "score": float(output[idx][6]),
                "iou": max_iou
            })

        top_number_of_person_coco_output = heapq.nlargest(number_of_person, coco_outputs, key=lambda x: x.get("iou", float('-inf')))
        for d in top_number_of_person_coco_output:
            d.pop("iou", None)

        if top_number_of_person_coco_output:
            self.all_outputs[image_id] = top_number_of_person_coco_output

    def detect(self, timestamp=None):
        detect_outputs = {}
        root_path = sorted(os.listdir(self.root))
        with torch.no_grad():
            for root_idx, root_dir in enumerate(root_path):
                if timestamp is not None:
                    # camera streaming
                    frame_span = 60
                    right_span = frame_span // 2
                    left_span = frame_span - right_span
                    for x in range(int(timestamp) - left_span, int(timestamp) + right_span):
                        if self.process_videos.is_keyframe(x):
                            if os.path.exists(os.path.join(self.root, root_dir, "{}.jpg".format(x))):
                                if x not in self.all_outputs:
                                    self.detect_one(x, root_idx, root_dir)
                                detect_outputs[x] = self.all_outputs[x]
                else:
                    path = os.path.join(self.root, root_dir)
                    list_of_images = os.listdir(path)
                    for image in tqdm(list_of_images):
                        self.detect_one(image.split(".")[0], root_idx, root_dir)

    def dump(self):
        with open(self.json_path, "w") as fp:
            output = []
            for op in self.all_outputs.values():
                for line in op:
                    output.append(line)
            json.dump(output, fp)
        print(f"Write keypoints into json file {self.json_path} successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose estimation using yolov7")
    parser.add_argument(
        "--train",
        help="Build training dataset",
        action="store_true",
    )
    parser.add_argument("--person-left-boundary", default="", help="person boundary")
    parser.add_argument("--person-right-boundary", default="", help="person boundary")
    parser.add_argument("--person-top-boundary", default="", help="person boundary")
    parser.add_argument("--person-botton-boundary", default="", help="person boundary")

    args = parser.parse_args()

    t1 = time.time()
    key_points_detector = KeyPointDetection(is_train=args.train, args=args)
    key_points_detector.detect()
    key_points_detector.dump()
    t2 = time.time()
    print("Total Time :", t2 - t1)
