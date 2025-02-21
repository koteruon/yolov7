import argparse
import csv
import datetime
import json
import math
import os
import queue
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pandas as pd
from matplotlib.path import Path as matplotlib_path
from scipy.optimize import leastsq
from tqdm import tqdm


class Trajectory:
    def Monotonic(self, L, strictly=False, half=False):
        # 檢查單調函數(嚴格遞增或遞減)
        if half:
            if strictly:
                # 找出所有递增子序列的起始索引
                mask = np.concatenate(([False], L[1:] > L[:-1], [False]))
            else:
                mask = np.concatenate(([False], L[1:] >= L[:-1], [False]))

            # 使用 `diff` 函數計算相鄰元素之間的差值
            diff = np.diff(mask.astype(np.int32))
            # 查找連續 True 子字串的起始索引
            start_indices = np.where(diff == 1)[0] + 1
            # 查找連續 True 子字串的結束索引
            end_indices = np.where(diff == -1)[0] + 1

            # 計算子字串長度
            subsequence_lengths = end_indices - start_indices
            # 計算連續 True 子字串的長度
            if len(end_indices) > 0:
                if np.max(subsequence_lengths) > (len(L) // 2):
                    half_strictly_increasing = True
                else:
                    half_strictly_increasing = False
            else:
                half_strictly_increasing = False

            if strictly:
                # 找出所有遞減子序列的起始索引
                mask = np.concatenate(([False], L[1:] < L[:-1], [False]))
            else:
                mask = np.concatenate(([False], L[1:] <= L[:-1], [False]))

            # 使用 `diff` 函數計算相鄰元素之間的差值
            diff = np.diff(mask.astype(np.int32))
            # 查找連續 True 子字串的起始索引
            start_indices = np.where(diff == 1)[0] + 1
            # 查找連續 True 子字串的結束索引
            end_indices = np.where(diff == -1)[0] + 1

            # 計算子字串長度
            subsequence_lengths = end_indices - start_indices
            # 計算連續 True 子字串的長度
            if len(end_indices) > 0:
                if np.max(subsequence_lengths) > (len(L) // 2):
                    half_strictly_decreasing = True
                else:
                    half_strictly_decreasing = False
            else:
                half_strictly_decreasing = False

            # 回傳結果
            if half_strictly_increasing:
                return "right"
            elif half_strictly_decreasing:
                return "left"
            else:
                return "unknown"
        else:
            if strictly:
                strictly_increasing = np.all(L[1:] > L[:-1])
                strictly_decreasing = np.all(L[1:] < L[:-1])
                # 回傳結果
                if strictly_increasing:
                    return "right"
                elif strictly_decreasing:
                    return "left"
                else:
                    return "unknown"
            else:
                non_strictly_increasing = np.all(L[1:] >= L[:-1])
                non_strictly_decreasing = np.all(L[1:] <= L[:-1])
                # 回傳結果
                if non_strictly_increasing:
                    return "right"
                elif non_strictly_decreasing:
                    return "left"
                else:
                    return "unknown"

    def Euclidean_Distance(self, x, y, x1, y1):
        # 計算歐式距離
        ed = math.sqrt(pow(x - x1, 2) + pow(y - y1, 2))
        return ed

    def Create_Output_Dir(self, output_path):
        # 建立輸出檔案夾
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        # 建立影片預測結果資料夾
        video_path = Path(output_path.joinpath("video"))
        video_path.mkdir(parents=True, exist_ok=True)
        video_path = video_path.as_posix()
        # 建立影片預測結果資料夾
        speed_path = Path(output_path.joinpath("speed"))
        speed_path.mkdir(parents=True, exist_ok=True)
        speed_path = speed_path.as_posix()

        return video_path, speed_path

    def Read_Video(self, input_path):
        # 讀取影片
        # 選擇影片編碼
        if self.video_suffix in [".avi"]:
            fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        elif self.video_suffix in [".mp4", ".MP4", ".mov", ".MOV"]:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            print("usage: video type can only be .avi or .mp4 or .MOV")
            exit(1)

        # 讀取影片
        cap = cv2.VideoCapture(input_path)
        success, image = cap.read()
        if not success:
            raise Exception("Could not read")

        framerate = int(round(cap.get(cv2.CAP_PROP_FPS)))
        frame_height, frame_width = int(cap.get(4)), int(cap.get(3))
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        return success, image, cap, framerate, frame_height, frame_width, total_frames

    def Write_Video(self, video_path, size):
        # 選擇影片編碼
        if self.video_suffix in [".avi"]:
            fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        elif self.video_suffix in [".mp4", ".MP4", ".mov", ".MOV"]:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            print("usage: video type can only be .avi or .mp4 or .MOV")
            exit(1)

        # 寫 預測結果
        output = cv2.VideoWriter(
            video_path,
            fourcc,
            self.framerate,
            size,
        )

        return output

    def draw_circle(self, event, x, y, flags, param):
        colors, current_color_index = param
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if len(self.PT_dict["points"]) % 4 == 0 and len(self.PT_dict["points"]) > 0:
                current_color_index[0] = (current_color_index[0] + 1) % len(colors)  # 換顏色

            self.PT_dict["points"].append((x, y))
            cv2.circle(self.PT_dict["img"], (x, y), 5, colors[current_color_index[0]], -1)

    def get_grouped_perspective_points(self, image, frame_width, frame_height):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # 輪流使用的顏色
        current_color_index = [0]  # 使用列表來保持可變性

        # 觀測區域
        if self.mark:
            self.PT_dict = {"img": image.copy(), "points": []}
            cv2.namedWindow("PIC2 (press Q to quit)", 0)
            cv2.resizeWindow("PIC2 (press Q to quit)", frame_width, frame_height)
            cv2.setMouseCallback("PIC2 (press Q to quit)", self.draw_circle, (colors, current_color_index))

            while True:
                cv2.imshow("PIC2 (press Q to quit)", self.PT_dict["img"])
                if cv2.waitKey(2) == ord("q"):
                    print(self.PT_dict)
                    cv2.destroyWindow("PIC2 (press Q to quit)")
                    break
            self.Write_focus_bbox(self.PT_dict["points"])
        else:
            self.PT_dict = {"img": image.copy(), "points": self.Read_focus_bbox()}

        grouped_points = [
            self.PT_dict["points"][i : i + 4]
            for i in range(0, len(self.PT_dict["points"]), 4)
            if len(self.PT_dict["points"][i : i + 4]) == 4
        ]

        self.polygon_paths = [matplotlib_path(np.float32(group)) for group in grouped_points]

        # Switch 區域
        # self.PT_dict = {"img": image.copy(), "points": []}
        # cv2.namedWindow("PIC2 (press Q to quit)", 0)
        # cv2.resizeWindow("PIC2 (press Q to quit)", frame_width, frame_height)
        # cv2.setMouseCallback("PIC2 (press Q to quit)", self.draw_circle, (colors, current_color_index))

        # while True:
        #     cv2.imshow("PIC2 (press Q to quit)", self.PT_dict["img"])
        #     if cv2.waitKey(2) == ord("q"):
        #         print(self.PT_dict)
        #         cv2.destroyWindow("PIC2 (press Q to quit)")
        #         break

        # self.PT_dict = {"img": image.copy(), "points": [(777, 224), (772, 622), (1182, 616), (1161, 217)]}  # C0012

        # grouped_points = [
        #     self.PT_dict["points"][i : i + 4]
        #     for i in range(0, len(self.PT_dict["points"]), 4)
        #     if len(self.PT_dict["points"][i : i + 4]) == 4
        # ]

        # self.swtich_polygon_paths = [matplotlib_path(np.float32(group)) for group in grouped_points]

    def Read_Yolo_Label_One_Frame(self, label_file=None, balls=None, x_c_pred=None, y_c_pred=None):
        if x_c_pred != None and y_c_pred != None:
            self.x_c_pred, self.y_c_pred = x_c_pred, y_c_pred
            return

        if balls == None:
            balls = []
        # 取得Yolo預測球的位置
        if label_file:
            with open(label_file, "r") as f:
                for line in f:
                    l = line.split()
                    if len(l) > 0:
                        if int(l[0]) == 0:
                            balls.append(l)

        # 找尋最接近上次的球
        distance = sys.float_info.max
        for ball in balls:
            ball_x_c_pred, ball_y_c_pred = int(float(ball[1]) * self.frame_width), int(
                float(ball[2]) * self.frame_height
            )
            ball_point = (ball_x_c_pred, ball_y_c_pred)

            if any(polygon.contains_point(ball_point) for polygon in self.polygon_paths):
                self.ball_update = True
                self.x_pre_c_pred, self.y_pre_c_pred = self.x_c_pred, self.y_c_pred
                if self.x_c_pred == np.inf or self.y_c_pred == np.inf:
                    self.x_c_pred, self.y_c_pred = ball_x_c_pred, ball_y_c_pred
                    self.x_pre_c_pred, self.y_pre_c_pred = self.x_c_pred, self.y_c_pred
                elif distance > math.sqrt((ball_x_c_pred - self.x_c_pred) ** 2 + (ball_y_c_pred - self.y_c_pred) ** 2):
                    distance = math.sqrt((ball_x_c_pred - self.x_c_pred) ** 2 + (ball_y_c_pred - self.y_c_pred) ** 2)
                    self.x_c_pred, self.y_c_pred = ball_x_c_pred, ball_y_c_pred

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Predict")
        parser.add_argument("--input", required=True, type=str, help="Input video")
        args = parser.parse_args()
        return args

    def calculate_required_frame_ball_pixel(self, pixel_distance, target_speed_kmph, frame_rate):
        # Convert target speed from km/h to m/s
        target_speed_mps = (target_speed_kmph * 1000) / 3600

        # Calculate time in seconds per frame
        time_in_seconds = 1 / frame_rate

        # Rearrange the formula to solve for meters_per_pixel
        meters_per_pixel = (target_speed_mps * time_in_seconds) / pixel_distance

        # Calculate frame_ball_pixel
        frame_ball_pixel = 0.395 / meters_per_pixel

        return frame_ball_pixel

    def Calculate_Speed(self, image):
        # 針對每一貞做運算
        image_CV = image.copy()

        ## 有偵測到球體
        if self.x_c_pred != np.inf and self.y_c_pred != np.inf:
            balls = 2
            q_array = np.array(self.q)
            non_negatives_idx = np.where(np.all(q_array != (-1, -1), axis=1))[0][:balls]
            q_array = q_array[non_negatives_idx]
            if q_array.size == 0:
                x_tmp = np.array([])
                y_tmp = np.array([])
            else:
                x_tmp = q_array[:, 0]
                y_tmp = q_array[:, 1]

            if self.calculate_speed_direction != "right" and self.calculate_speed_direction != "left":
                raise Exception("error direction")

            if self.calculate_speed_direction == "right":
                if self.x_c_pred < self.x_pre_c_pred:
                    self.speed[self.count] = 0.0
                    return image_CV

            if self.calculate_speed_direction == "left":
                if self.x_c_pred > self.x_pre_c_pred:
                    self.speed[self.count] = 0.0
                    return image_CV

            ball_point = (self.x_c_pred, self.y_c_pred)
            if self.swtich_polygon_paths:
                if any(polygon.contains_point(ball_point) for polygon in self.swtich_polygon_paths):
                    if self.pixel_area_switch:
                        if self.pixel_area_current == None:
                            self.pixel_area_current = 0
                        elif self.pixel_area_current == 0:
                            self.pixel_area_current = 1
                        else:
                            self.pixel_area_current = 0
                    self.pixel_area_switch = False
                else:
                    self.pixel_area_switch = True

            if self.area_timestamp_ranges:
                self.Check_Pixel_Area_Timestamp()

            deistance_pixel = self.Euclidean_Distance(
                self.x_pre_c_pred, self.y_pre_c_pred, self.x_c_pred, self.y_c_pred
            )

            self.speed[self.count] = self.calculate_ball_speed(deistance_pixel)

        else:
            self.speed[self.count] = 0.0

        return image_CV

    def Draw_On_Image(self, image_CV):
        if self.x_c_pred != np.inf and self.y_c_pred != np.inf:
            cv2.circle(image_CV, (self.x_c_pred, self.y_c_pred), 5, (0, 255, 255), 1)

        cv2.putText(
            image_CV,
            f"Frame : {self.count}",
            (10, 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image_CV,
            f"Speed : {self.speed[self.count]}",
            (10, 80),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # cv2.putText(
        #     image_CV,
        #     f"Pixel Area : {self.pixel_area_current}",
        #     (10, 120),
        #     cv2.FONT_HERSHEY_TRIPLEX,
        #     1,
        #     (0, 255, 255),
        #     1,
        #     cv2.LINE_AA,
        # )

        return image_CV

    def __init__(self, root_path, video_fullname, timestamp_path, id, mark=False):
        # temp#
        self.only_speed = False

        self.HEIGHT = 288  # model input size
        self.WIDTH = 512

        # 影片跟目錄
        self.video_name = os.path.splitext(video_fullname)[0]
        self.video_suffix = os.path.splitext(video_fullname)[1]
        self.input_path = os.path.join(root_path, video_fullname)

        # 建立目錄
        output_path = f"./inference/output"
        self.video_path, self.speed_path = self.Create_Output_Dir(output_path)

        # yolo labels path
        self.label_path = os.path.join(root_path, "labels")

        # 透視變形
        self.PT_dict = {}

        # In order to draw the trajectory of tennis, we need to save the coordinate of preious 12 frames
        self.q = queue.deque([(-1, -1) for _ in range(12)])

        # bounce detection init
        self.q_bv = queue.deque([(-1, -1) for _ in range(6)])

        # 參數
        self.bounce = []
        self.count = 1  # 記錄處理幾個 Frame
        self.x_c_pred, self.y_c_pred = np.inf, np.inf  # 球體中心位置
        self.x_pre_c_pred, self.y_pre_c_pred = np.inf, np.inf  # 球體中心位置

        self.mark = mark

        # 球速參數
        self.speed = {}
        self.no_ball = 0
        self.real_ball_size = 0.0395  # 單位是m

        self.Read_ball_pixel()
        self.meters_per_pixel = [
            self.real_ball_size / pixel for pixel in self.frame_ball_pixel
        ]  # 依據每個 pixel 值計算 meters_per_pixel
        self.pixel_area_switch = True
        self.pixel_area_current = 0
        self.area_timestamp_ranges = None

        if id == "id13":
            self.Read_Timestamp(timestamp_path)
        self.Read_Direction()

        self.polygon_paths = None
        self.swtich_polygon_paths = None

    def Read_ball_pixel(self, filename="exhaustion/ball_size.csv"):
        df = pd.read_csv(filename)
        frame_ball_pixel = {}
        for _, row in df.iterrows():
            pixels = [row[col] for col in df.columns if col.startswith("pixel") and not pd.isna(row[col])]
            frame_ball_pixel[row["video_id"]] = pixels
        self.frame_ball_pixel = frame_ball_pixel.get(self.video_name)

    def Read_focus_bbox(self, filename="exhaustion/focus_bbox.csv"):
        df = pd.read_csv(filename)
        # 只取當前影片的資料
        row = df[df["video_id"] == self.video_name]
        if row.empty:
            raise Exception("No Focus BBox")
        # 解析點座標
        points = []
        for i in range(1, (len(row.columns) - 1) // 2 + 1):  # 從 point_x1, point_y1 開始
            x, y = row[f"point_x{i}"].values[0], row[f"point_y{i}"].values[0]
            if pd.notna(x) and pd.notna(y):  # 避免 NaN 值
                points.append((x, y))
        return points

    def Write_focus_bbox(self, points_list, filename="exhaustion/focus_bbox.csv"):
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["video_id"])  # 先初始化一個空的 DataFrame
        # 移除舊的該影片資料
        df = df[df["video_id"] != self.video_name]
        # 構建新的一行資料
        new_row = {"video_id": self.video_name}
        for i, (x, y) in enumerate(points_list, start=1):
            new_row[f"point_x{i}"] = x
            new_row[f"point_y{i}"] = y
        # 將新資料轉為 DataFrame 並合併
        new_data = pd.DataFrame([new_row])
        df = pd.concat([df, new_data], ignore_index=True)
        # 儲存回 CSV
        df.to_csv(filename, index=False)

    def Read_Timestamp(self, filename):
        df = pd.read_csv(filename)
        self.area_timestamp_ranges = list(df.itertuples(index=True, name=None))

    def Check_Pixel_Area_Timestamp(self):
        self.pixel_area_current = 0
        for index, start, end in self.area_timestamp_ranges:
            if start <= self.count <= end:
                self.pixel_area_current = 1 if index % 2 == 1 else 0
                break

    def Read_Direction(self, filename="exhaustion/direction.csv"):
        df = pd.read_csv(filename)
        # 只取當前影片的資料
        row = df[df["video_id"] == self.video_name]
        if row.empty:
            raise Exception("No Direction")
        self.calculate_speed_direction = row["direction"].iloc[0]

    def Set_Frame_Info(self, frame_height, frame_width, framerate):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.framerate = framerate
        self.time_in_seconds = 1 / self.framerate

    def Next_Count(self):
        self.count += 1

    def save_frame_speed_to_csv(self):
        filename = os.path.join(self.speed_path, self.video_name + ".csv")
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Frame", "Speed"])
            for frame, speed in sorted(self.speed.items()):
                writer.writerow([frame, speed])

    def calculate_ball_speed(self, pixel_distance):
        if self.pixel_area_current != None:
            # 計算球的實際距離（米）
            distance_in_meters = pixel_distance * self.meters_per_pixel[self.pixel_area_current]

            # 計算速度（米/秒）
            speed_in_mps = distance_in_meters / self.time_in_seconds

            # 轉換為米/小時
            speed_in_kmph = (speed_in_mps * 3600) / 1000

            if speed_in_kmph > 100:
                print("over speed")

            return round(speed_in_kmph, 1)
        else:
            return 0.0

    ### 後處理從此開始 ###
    def main(self):
        # 讀影片
        success, image, cap, framerate, frame_height, frame_width, total_frames = self.Read_Video(self.input_path)
        self.Set_Frame_Info(frame_height, frame_width, framerate)

        # 等比例縮放
        ratio = self.frame_height / self.HEIGHT
        size = (int(self.WIDTH * ratio), int(self.HEIGHT * ratio))

        # 寫 預測結果
        video_path = f"{self.video_path}/{self.video_name}_exhaustion.mp4"
        output = self.Write_Video(video_path, size)

        # 透視變形
        self.get_grouped_perspective_points(image, self.frame_width, self.frame_height)

        # 針對每一貞做運算
        start = time.time()
        with tqdm(total=total_frames, desc="Processing Frames") as pbar:
            while success:
                label_file = os.path.join(self.label_path, f"{self.video_name}_{self.count}.txt")
                self.ball_update = False
                if os.path.exists(label_file):
                    self.Read_Yolo_Label_One_Frame(label_file=label_file)
                if not self.ball_update:
                    self.x_c_pred, self.y_c_pred = np.inf, np.inf  # 球體中心位置
                    self.x_pre_c_pred, self.y_pre_c_pred = np.inf, np.inf  # 球體中心位置

                image_CV = self.Calculate_Speed(image)
                image_CV = self.Draw_On_Image(image_CV)

                output.write(image_CV)
                pbar.update(1)
                self.Next_Count()

                success, image = cap.read()

        # For releasing cap and out.
        cap.release()
        output.release()

        self.save_frame_speed_to_csv()

        end = time.time()
        print(f"Write video time: {end-start} seconds.")
        total_time = end - start

        print()
        print(f"Detect Result is saved in {self.video_path}")
        print(f"Total time: {total_time} seconds")
        print(f"Done......")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark", action="store_true")
    opt = parser.parse_args()

    videos = ["301", "302", "303", "304", "305", "307", "308", "309"]
    ids = ["id11", "id12", "id13"]

    for video in videos:
        for id in ids:
            root_path = f"./runs/detect/exhaustion_{video}_{id}"
            video_fullname = f"{video}_{id}.mp4"
            timestamp_path = f"./exhaustion/{video}_{id}_timestamp.csv"
            trajectory = Trajectory(root_path, video_fullname, timestamp_path, id, mark=opt.mark)
            trajectory.main()
