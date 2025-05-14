import argparse
import datetime
import json
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pandas as pd
from scipy.optimize import leastsq
from tqdm import tqdm


class Trajectory:
    def PJcurvature(self, x, y):
        # 計算離散取率
        """
        input  : the coordinate of the three point
        output : the curvature and norm direction
        """
        t_a = LA.norm([x[1] - x[0], y[1] - y[0]])
        t_b = LA.norm([x[2] - x[1], y[2] - y[1]])

        M = np.array([[1, -t_a, t_a**2], [1, 0, 0], [1, t_b, t_b**2]])

        try:
            inv = LA.inv(M)
        except:
            inv = LA.pinv(M)

        a = np.matmul(inv, x)
        b = np.matmul(inv, y)

        if (a[1] ** 2 + b[1] ** 2) ** (1.5) == 0:
            kappa = 0
        else:
            kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2 + b[1] ** 2) ** (1.5)

        return kappa, 0

    def Custom_Time(self, time):
        # time: in milliseconds
        seconds, milliseconds = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)

        # 格式化時間
        cts = "{:02d}:{:02d}:{:02d}.{:03d}".format(hours, minutes, seconds, milliseconds)
        return cts

    def Monotonic(self, L, strictly=False, half=False, return_unknown=True):
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
            total_increasing = np.sum(subsequence_lengths)

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
            total_decreasing = np.sum(subsequence_lengths)

            # 回傳結果
            if half_strictly_increasing:
                return "right"
            elif half_strictly_decreasing:
                return "left"
            else:
                if return_unknown:
                    return "unknown"
                else:
                    if total_increasing > total_decreasing:
                        return "right"
                    else:
                        return "left"
        else:
            if strictly:
                strictly_increasing = np.all(L[1:] > L[:-1])
                strictly_decreasing = np.all(L[1:] < L[:-1])
                inc_count = np.sum(L[1:] > L[:-1])
                dec_count = np.sum(L[1:] < L[:-1])
                # 回傳結果
                if strictly_increasing:
                    return "right"
                elif strictly_decreasing:
                    return "left"
                else:
                    if return_unknown:
                        return "unknown"
                    else:
                        if inc_count > dec_count:
                            return "right"
                        else:
                            return "left"
            else:
                non_strictly_increasing = np.all(L[1:] >= L[:-1])
                non_strictly_decreasing = np.all(L[1:] <= L[:-1])
                inc_count = np.sum(L[1:] >= L[:-1])
                dec_count = np.sum(L[1:] <= L[:-1])
                # 回傳結果
                if non_strictly_increasing:
                    return "right"
                elif non_strictly_decreasing:
                    return "left"
                else:
                    if return_unknown:
                        return "unknown"
                    else:
                        if inc_count > dec_count:
                            return "right"
                        else:
                            return "left"

    def Euclidean_Distance(self, x, y, x1, y1):
        # 計算歐式距離
        ed = math.sqrt(pow(x - x1, 2) + pow(y - y1, 2))
        return ed

    def Parabola_Function(self, params, x):
        # 拋物線函數 Parabola Function
        a, b, c = params
        return a * x**2 + b * x + c

    def Parabola_Error(self, params, x, y):
        # 拋物線偏移誤差
        return self.Parabola_Function(params, x) - y

    def Solve_Parabola(self, X, Y):
        # 解拋物線方程式
        p_arg = [10, 10, 10]
        parabola = leastsq(self.Parabola_Error, p_arg, args=(X, Y))
        return parabola

    def Perspective_Transform(self, matrix, coord):
        # 透視變形轉換
        x = (matrix[0][0] * coord[0] + matrix[0][1] * coord[1] + matrix[0][2]) / (
            (matrix[2][0] * coord[0] + matrix[2][1] * coord[1] + matrix[2][2])
        )
        y = (matrix[1][0] * coord[0] + matrix[1][1] * coord[1] + matrix[1][2]) / (
            (matrix[2][0] * coord[0] + matrix[2][1] * coord[1] + matrix[2][2])
        )
        PT = (int(x), int(y))
        return PT

    def Generate_HeatMap(self, w, h, x_c, y_c, r, mag):
        # 生成熱力圖(觀察球的mask)
        if x_c < 0 or y_c < 0:
            return np.zeros((h, w))
        x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
        heatmap = ((y - (y_c + 1)) ** 2) + ((x - (x_c + 1)) ** 2)
        heatmap[heatmap <= r**2] = 1
        heatmap[heatmap > r**2] = 0
        return heatmap * mag

    def Count_BounceLocation(self, frame):
        # 落點分析 bounce analyize function
        row = int(frame[0] / int(self.miniboard_width / 4))
        column = int(frame[1] / int(self.miniboard_height / 3))
        if 0 <= row < 4 and 0 <= column < 3:
            self.bounce_location_list[row][column] += 1

    def Detect_Color_Level(self, score, side, side_min, side_max):
        # 判斷落點方
        # gray_level_min  = 0
        gray_level_max = 255
        color = []
        if side_max == 0:
            normalize_score = 0
        else:
            # score / side_min+(side_max- side_min)
            normalize_score = int(np.round((score) * (255 / (side_max)), 0))
        if side == "left":
            color = (
                gray_level_max - normalize_score,
                gray_level_max - normalize_score,
                255,
            )  # (0,0,255)
            return color
        elif side == "right":
            color = (
                gray_level_max - normalize_score,
                255,
                gray_level_max - normalize_score,
            )
            return color

    def Draw_Bounce_Analysis(self):
        if self.is_show_bounce_analysis:
            # 落點分析圖
            self.bounce_analyze_img = self.Draw_MiniBoard("bounce")
            score_table = np.zeros((4, 3), dtype=int)

            # calculate side sum
            left_bounce_sum = np.sum(self.bounce_location_list[:2])
            right_bounce_sum = np.sum(self.bounce_location_list[2:])

            # calculate side score
            ### Calculate score for left side
            if left_bounce_sum != 0:
                left_scores = np.round((self.bounce_location_list[:2] / left_bounce_sum) * 100).astype(int)
            else:
                left_scores = np.zeros((2, 3), dtype=int)
            ### Calculate score for right side
            if right_bounce_sum != 0:
                right_scores = np.round((self.bounce_location_list[2:] / right_bounce_sum) * 100).astype(int)
            else:
                right_scores = np.zeros((2, 3), dtype=int)
            ### Assign scores to score_table
            score_table[:2] = left_scores
            score_table[2:] = right_scores

            # find max and min
            ### Find min and max for left side
            left_score_min = np.min(score_table[:2])
            left_score_max = np.max(score_table[:2])

            ### Find min and max for right side
            right_score_min = np.min(score_table[2:])
            right_score_max = np.max(score_table[2:])

            for i in range(4):
                for j in range(3):
                    if i < 2:
                        # left
                        color_detect = self.Detect_Color_Level(
                            score_table[i][j], "left", left_score_min, left_score_max
                        )
                    else:
                        # right
                        color_detect = self.Detect_Color_Level(
                            score_table[i][j], "right", right_score_min, right_score_max
                        )
                    text = str(score_table[i][j]) + "%"
                    cv2.rectangle(
                        self.bounce_analyze_img,
                        (
                            self.miniboard_edge + (i * int(self.miniboard_width / 4)) + 10,
                            self.miniboard_edge + (j * int(self.miniboard_height / 3)) + 10,
                        ),
                        (
                            ((i + 1) * int(self.miniboard_width / 4)) + self.miniboard_edge - 10,
                            ((j + 1) * int(self.miniboard_height / 3)) + self.miniboard_edge - 10,
                        ),
                        color=color_detect,
                        thickness=-1,
                    )
                    cv2.putText(
                        self.bounce_analyze_img,
                        text,
                        (
                            self.miniboard_edge + (i * int(self.miniboard_width / 4)) + self.miniboard_edge * 2,
                            self.miniboard_edge + (j * int(self.miniboard_height / 3)) + self.miniboard_text_bias,
                        ),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        (1, 1, 1),
                        1,
                        cv2.LINE_AA,
                    )

    def Show_Bounce(self):
        if self.is_show_bounce_window:
            cv2.imshow(self.bounce_title, self.img_opt)

    def Show_Bounce_Analysis(self):
        if self.is_show_bounce_analysis:
            cv2.imshow(self.bounce_analysis_title, self.bounce_analyze_img)

    def Save_Bounce_Analysis(self):
        if self.is_save_bounce_analysis:
            cv2.imwrite(
                f"{self.analysis_img_path}/{self.video_name}_analysis.jpg",
                self.bounce_analyze_img,
            )

    def Show_Bounce_Location(self):
        if self.is_show_bounce_location:
            cv2.imshow(self.bounce_location_title, self.img_opt_bounce_location)

    def Save_Bounce_Location(self):
        if self.is_save_bounce_location:
            cv2.imwrite(
                f"{self.bounce_img_path}/{self.video_name}_bounce.jpg",
                self.img_opt_bounce_location,
            )

    def Draw_SpeedHist(self, save=True, show=False):
        # 繪製速度直方圖
        stroke_length = 0
        # 平衡左右 list ??
        if len(self.left_speed_list) > len(self.right_speed_list):
            stroke_length = len(self.left_speed_list)
            for i in range(len(self.left_speed_list) - len(self.right_speed_list)):
                self.right_speed_list.append(0)
        else:
            stroke_length = len(self.right_speed_list)
            for i in range(len(self.right_speed_list) - len(self.left_speed_list)):
                self.left_speed_list.append(0)

        shots_list = np.arange(1, stroke_length + 1)

        label_left = f"left_player mean:{str(round(np.mean(self.left_speed_list),2))}"
        label_right = f"right_player mean:{str(round(np.mean(self.right_speed_list),2))}"
        plt.figure(figsize=(15, 10), dpi=100, linewidth=2)
        plt.plot(shots_list, self.left_speed_list, "s-", color="royalblue", label=label_left)
        plt.plot(shots_list, self.right_speed_list, "o-", color="darkorange", label=label_right)
        plt.title(f"Comparing the shot speed between the two players", x=0.5, y=1.03, fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(f"shots", fontsize=30, labelpad=15)
        plt.ylabel(f"m/s", fontsize=30, labelpad=20)
        plt.legend(loc="best", fontsize=20)
        if save:
            plt.savefig(os.path.join(self.speedhis_path, f"{self.video_name}_shot_speedhis.png"))
        if show:
            fig = plt.gcf()
            fig.canvas.draw()
            fig_img = np.array(fig.canvas.renderer.buffer_rgba())
            cv2.imshow(self.speedhis_title, cv2.cvtColor(fig_img, cv2.COLOR_RGBA2BGR))
        plt.clf()

        plt.figure(figsize=(15, 10), dpi=100, linewidth=2)
        plt.hist([self.left_speed_list, self.right_speed_list], bins="auto", alpha=1, label=["left", "right"])

        plt.xlabel(f"m/s", fontsize=30, labelpad=15)
        plt.ylabel(f"shots", fontsize=30, labelpad=20)
        plt.legend(loc="upper right")
        if save:
            plt.savefig(os.path.join(self.speed_distribution_path, f"{self.video_name}_shot_speed_distribution.png"))
        if show:
            fig = plt.gcf()
            fig.canvas.draw()
            fig_img = np.array(fig.canvas.renderer.buffer_rgba())
            cv2.imshow(self.speed_distribution_title, cv2.cvtColor(fig_img, cv2.COLOR_RGBA2BGR))
        plt.clf()

    def Draw_MiniBoard(self, option=None):
        img_opt = np.zeros(
            [self.miniboard_height + self.miniboard_edge * 2, self.miniboard_width + self.miniboard_edge * 2, 3],
            dtype=np.uint8,
        )
        cv2.rectangle(
            img_opt,
            (self.miniboard_edge, self.miniboard_edge),
            (self.miniboard_width + self.miniboard_edge, self.miniboard_height + self.miniboard_edge),
            color=(255, 255, 255),
            thickness=7,
        )
        cv2.rectangle(
            img_opt,
            (self.miniboard_edge, self.miniboard_edge),
            (self.miniboard_width + self.miniboard_edge, self.miniboard_height + self.miniboard_edge),
            color=(255, 150, 50),
            thickness=-1,
        )
        cv2.line(
            img_opt,
            (int(self.miniboard_width / 2) + self.miniboard_edge, self.miniboard_edge),
            (int(self.miniboard_width / 2) + self.miniboard_edge, self.miniboard_height + self.miniboard_edge),
            (255, 255, 255),
            5,
        )
        if option == "bounce":
            cv2.line(
                img_opt,
                (int(self.miniboard_width / 4) + self.miniboard_edge, self.miniboard_edge),
                (int(self.miniboard_width / 4) + self.miniboard_edge, self.miniboard_height + self.miniboard_edge),
                (128, 0, 128),
                3,
            )
            cv2.line(
                img_opt,
                (int((self.miniboard_width / 4) * 3) + self.miniboard_edge, self.miniboard_edge),
                (
                    int((self.miniboard_width / 4) * 3) + self.miniboard_edge,
                    self.miniboard_height + self.miniboard_edge,
                ),
                (128, 0, 128),
                3,
            )
            cv2.line(
                img_opt,
                (self.miniboard_edge, int(self.miniboard_height / 3) + self.miniboard_edge),
                (self.miniboard_width + self.miniboard_edge, int(self.miniboard_height / 3) + self.miniboard_edge),
                (128, 0, 128),
                3,
            )
            cv2.line(
                img_opt,
                (self.miniboard_edge, int((self.miniboard_height / 3) * 2) + self.miniboard_edge),
                (
                    self.miniboard_width + self.miniboard_edge,
                    int((self.miniboard_height / 3) * 2) + self.miniboard_edge,
                ),
                (128, 0, 128),
                3,
            )
        return img_opt

    def Draw_Circle(self, event, x, y, flags, param):
        # 用於透視變形取點
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(param["img"], (x, y), 3, (0, 255, 255), -1)
            param["point_x"].append(x)
            param["point_y"].append(y)

    def Draw_and_Collect_Data(
        self,
        color,
        loc_PT,
    ):
        cv2.circle(self.img_opt, self.PT_dict[self.count], 5, color, 4)
        # add to bounce map
        cv2.circle(
            self.img_opt_bounce_location,
            (
                self.PT_dict[self.count][0] + self.bouncing_offset_x,
                self.PT_dict[self.count][1] + self.bouncing_offset_y,
            ),
            5,
            color,
            4,
        )
        # analyze location
        self.Count_BounceLocation(self.PT_dict[self.count])
        self.Draw_Bounce_Analysis()
        self.Show_Bounce_Analysis()
        self.Show_Bounce_Location()
        p_inv = self.Perspective_Transform(self.inv, loc_PT)
        self.bounce.append([self.count, p_inv[0], p_inv[1]])
        self.q_bv.appendleft(p_inv)
        self.q_bv.pop()

    def Create_Output_Dir(self, output_path, video_name):
        # 建立輸出檔案夾
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        # 建立影片預測結果資料夾
        video_path = Path(output_path.joinpath("video"))
        video_path.mkdir(parents=True, exist_ok=True)
        video_path = video_path.as_posix()
        # 建立球點預測表格資料夾
        record_ball_path = Path(output_path.joinpath("record_balls"))
        record_ball_path.mkdir(parents=True, exist_ok=True)
        record_ball_path = record_ball_path.as_posix()
        # 建立球點預測表格資料夾
        record_pose_path = Path(output_path.joinpath("record_keypoints"))
        record_pose_path.mkdir(parents=True, exist_ok=True)
        record_pose_path = record_pose_path.as_posix()
        # 建立落點分析圖資料夾
        analysis_img_path = Path(output_path.joinpath("analysis"))
        analysis_img_path.mkdir(parents=True, exist_ok=True)
        analysis_img_path = analysis_img_path.as_posix()
        # 建立落點統計圖資料夾
        bounce_loc_path = Path(output_path.joinpath("bounce_location"))
        bounce_loc_path.mkdir(parents=True, exist_ok=True)
        bounce_loc_path = bounce_loc_path.as_posix()
        # 建立落點表格資料夾
        bounce_img_path = Path(output_path.joinpath("bounce"))
        bounce_img_path.mkdir(parents=True, exist_ok=True)
        bounce_img_path = bounce_img_path.as_posix()
        # 建立Keypoints資料夾
        keypoints_path = Path(output_path.joinpath("keypoints").joinpath(video_name))
        keypoints_path.mkdir(parents=True, exist_ok=True)
        keypoints_path = Path(output_path.joinpath("keypoints"))
        keypoints_path = keypoints_path.as_posix()
        # 建立球速直方圖
        speedhis_path = Path(output_path.joinpath("speedhis"))
        speedhis_path.mkdir(parents=True, exist_ok=True)
        speedhis_path = Path(output_path.joinpath("speedhis"))
        speedhis_path = speedhis_path.as_posix()
        # 建立球速直方圖
        speed_distribution_path = Path(output_path.joinpath("speed_distribution"))
        speed_distribution_path.mkdir(parents=True, exist_ok=True)
        speed_distribution_path = Path(output_path.joinpath("speed_distribution"))
        speed_distribution_path = speed_distribution_path.as_posix()

        return (
            video_path,
            record_ball_path,
            record_pose_path,
            analysis_img_path,
            bounce_loc_path,
            bounce_img_path,
            keypoints_path,
            speedhis_path,
            speed_distribution_path,
        )

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

        framerate = int(cap.get(cv2.CAP_PROP_FPS))
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

    def Load_Mark_Point(self, csv_path, PT_data, size):
        df = pd.read_csv(csv_path)
        if not df.empty:
            row = df.iloc[0]  # 取第一行數據
            PT_data["point_x"] = [row[f"point_x{i}"] for i in range(1, size + 1)]
            PT_data["point_y"] = [row[f"point_y{i}"] for i in range(1, size + 1)]
            return True
        else:
            return False

    def Save_Mark_Point(self, csv_path, PT_data, size):
        df = pd.read_csv(csv_path)
        new_row = {}
        for i in range(size):
            new_row[f"point_x{i+1}"] = PT_data["point_x"][i]
            new_row[f"point_y{i+1}"] = PT_data["point_y"][i]
        df = pd.DataFrame([new_row])
        df.to_csv(csv_path, index=False)

    def Mark_Perspective_Distortion_Point(self, image, frame_width, frame_height):
        # 點選透視變形位置, 順序為:左上,左下,右下,右上
        PT_data = {"img": image.copy(), "point_x": [], "point_y": []}
        table_csv_path = r"niag/focus_bbox.csv"
        if not self.Load_Mark_Point(table_csv_path, PT_data, 4):
            cv2.namedWindow("PIC2 (press Q to quit)", 0)
            cv2.resizeWindow("PIC2 (press Q to quit)", frame_width, frame_height)
            cv2.setMouseCallback("PIC2 (press Q to quit)", self.Draw_Circle, PT_data)
            while True:
                cv2.imshow("PIC2 (press Q to quit)", PT_data["img"])
                if cv2.waitKey(2) == ord("q"):
                    print(PT_data)
                    cv2.destroyWindow("PIC2 (press Q to quit)")
                    break
            self.Save_Mark_Point(table_csv_path, PT_data, 4)

        # PerspectiveTransform
        upper_left = [PT_data["point_x"][0], PT_data["point_y"][0]]
        lower_left = [PT_data["point_x"][1], PT_data["point_y"][1]]
        lower_right = [PT_data["point_x"][2], PT_data["point_y"][2]]
        upper_right = [PT_data["point_x"][3], PT_data["point_y"][3]]
        pts1 = np.float32([upper_left, lower_left, lower_right, upper_right])
        pts2 = np.float32(
            [
                [self.miniboard_edge, self.miniboard_edge],
                [self.miniboard_edge, self.miniboard_height + self.miniboard_edge],
                [self.miniboard_width + self.miniboard_edge, self.miniboard_height + self.miniboard_edge],
                [self.miniboard_width + self.miniboard_edge, self.miniboard_edge],
            ]
        )
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.inv = cv2.getPerspectiveTransform(pts2, pts1)

        # 繪製迷你落點板
        self.img_opt = self.Draw_MiniBoard()
        self.img_opt_bounce_location = self.Draw_MiniBoard("bounce")
        self.bounce_analyze_img = self.Draw_MiniBoard("bounce")

        # ----------------------------------------------------------------------------------------

        if self.is_real_time_speed:
            # 點選透視變形位置, 順序為:左上,左下,右下,右上
            PT_data = {"img": image.copy(), "point_x": [], "point_y": []}
            table_csv_path = r"niag/table_width.csv"
            if not self.Load_Mark_Point(table_csv_path, PT_data, 2):
                cv2.namedWindow("table_width (press Q to quit)", 0)
                cv2.resizeWindow("table_width (press Q to quit)", frame_width, frame_height)
                cv2.setMouseCallback("table_width (press Q to quit)", self.Draw_Circle, PT_data)
                while True:
                    cv2.imshow("table_width (press Q to quit)", PT_data["img"])
                    if cv2.waitKey(2) == ord("q"):
                        print(PT_data)
                        cv2.destroyWindow("table_width (press Q to quit)")
                        break
                self.Save_Mark_Point(table_csv_path, PT_data, 2)

            # PerspectiveTransform
            table_left = [PT_data["point_x"][0], PT_data["point_y"][0]]
            table_right = [PT_data["point_x"][1], PT_data["point_y"][1]]
            dx = table_right[0] - table_left[0]
            dy = table_right[1] - table_left[1]
            self.table_pixel_length = math.hypot(dx, dy)
            # 計算每 pixel 對應的實際距離（m/pixel）
            self.m_per_pixel = (self.real_table_width_cm / 100) / self.table_pixel_length

        # 顯示
        self.Draw_Bounce_Analysis()
        self.Show_Bounce_Analysis()
        self.Show_Bounce_Location()

    def Estimate_Ball_Speed_kmh(self):
        if self.x_c_pred == None or self.past_x_c_pred == None or self.y_c_pred == None or self.past_y_c_pred == None:
            return None
        delta_x_c_pred = self.x_c_pred - self.past_x_c_pred
        delta_y_c_pred = self.y_c_pred - self.past_y_c_pred
        # 計算球的 pixel 移動總長度
        delta_pixel = math.hypot(delta_x_c_pred, delta_y_c_pred)
        # 轉換為真實距離（單位：m）
        delta_m = delta_pixel * self.m_per_pixel

        frame_diff = self.count - self.past_c_frame_number + 1
        # 計算速度：m/frame → m/s
        return delta_m * self.framerate / frame_diff

    def Generate_Real_Time_Speed_index(self):
        if self.real_time_ball_direction != self.real_time_past_ball_direction:
            self.last_frame_switch_ball_direction = self.count
            self.real_time_speed_index_last += 1
        return self.real_time_speed_index_last

    def Add_Frame_In_Delay_Queue(self, real_time_speed_index_last, image_CV_real_time_speed):
        if len(self.delay_frame_queue) <= self.real_time_ball_direction_reference_frame_size // 2:
            self.delay_frame_queue.appendleft(image_CV_real_time_speed)
        else:
            self.delay_frame_queue.appendleft(image_CV_real_time_speed)
            image_CV_real_time_speed = self.delay_frame_queue.pop()
            self.Control_Queue("frame", real_time_speed_index_last, image_CV_real_time_speed)

    def Control_Queue(self, tag, index, image_CV):
        if tag != "frame" and tag != "save" and tag != "exit":
            raise Exception("unknow tag")
        self.control_queue.put((tag, index, image_CV))

    def Raise_Save_Queue(self):
        for index in range(self.real_time_speed_index_head, self.real_time_speed_index_last):
            self.Control_Queue("save", index, None)
        self.real_time_speed_index_head = self.real_time_speed_index_last

    def Is_Save_Queue(self):
        if self.count - self.last_frame_switch_ball_direction > self.save_queue_threadhold_by_switch_ball_direction:
            return True
        else:
            return False

    def Exit_Save_Queue(self):
        while self.delay_frame_queue:
            image_CV_real_time_speed = self.delay_frame_queue.pop()
            self.Control_Queue("frame", self.real_time_speed_index_last, image_CV_real_time_speed)
        for index in range(self.real_time_speed_index_head, self.real_time_speed_index_last + 1):
            self.Control_Queue("save", index, None)
        for index in range(self.real_time_speed_index_head, self.real_time_speed_index_last + 1):
            self.Control_Queue("exit", index, None)
        self.real_time_speed_process.join()

    def Real_Time_Speed_Process(self, root_path, control_queue, save_queue_minimum_frame_size):
        real_time_speed_pbar = tqdm(total=0)  # 擷取影像的分段標籤進度條
        real_time_speed_index_last = 0  # 擷取影像的分段標籤(尾)
        buffers = defaultdict(list)
        while True:
            if not control_queue.empty():
                tag, index, image_CV = control_queue.get()
                if tag == "frame":
                    buffers[index].append(image_CV)
                    if real_time_speed_index_last != index:
                        real_time_speed_index_last = index
                        real_time_speed_pbar.total = real_time_speed_index_last
                        real_time_speed_pbar.refresh()
                elif tag == "save":
                    if index in buffers:
                        if buffers[index]:
                            image_CVs = buffers[index]
                            image_CVs_size = len(image_CVs)
                            if save_queue_minimum_frame_size < image_CVs_size:
                                height, width, _ = image_CVs[0].shape
                                video_path = os.path.join(root_path, f"{index:03d}_{image_CVs_size}.mp4")
                                out = cv2.VideoWriter(
                                    video_path,
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    5,
                                    (width, height),
                                )
                                for f in image_CVs:
                                    out.write(f)
                                out.release()
                        buffers.pop(index, None)
                    real_time_speed_pbar.update()
                elif tag == "exit":
                    real_time_speed_pbar.close()
                    break
            time.sleep(0.01)

    def Read_Yolo_Label_One_Frame(self, label_file=None, balls=None, x_c_pred=None, y_c_pred=None):
        if x_c_pred != None and y_c_pred != None:
            self.past_c_frame_number = self.count
            self.past_x_c_pred, self.past_y_c_pred = self.x_c_pred, self.y_c_pred
            self.x_c_pred, self.y_c_pred = x_c_pred, y_c_pred
            return

        if self.x_c_pred != None and self.y_c_pred != None:
            self.past_c_frame_number = self.count
            self.past_x_c_pred, self.past_y_c_pred = self.x_c_pred, self.y_c_pred

        if balls == None:
            balls = []
        # 取得Yolo預測球的位置
        if label_file:
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    for line in f:
                        l = line.split()
                        if len(l) > 0:
                            if int(l[0]) == 0:
                                balls.append(l)

        if len(balls) == 0:
            self.x_c_pred, self.y_c_pred = None, None
            self.x_ltop_pred, self.y_ltop_pred = None, None
            return

        # 找尋最接近上次的球
        distance = sys.float_info.max
        for ball in balls:
            ball_x = int(float(ball[1]) * self.frame_width)
            ball_y = int(float(ball[2]) * self.frame_height)
            ball_w = float(ball[3]) * self.frame_width
            ball_h = float(ball[4]) * self.frame_height

            if self.x_c_pred == None or self.y_c_pred == None:
                self.x_c_pred, self.y_c_pred = ball_x, ball_y
                self.x_ltop_pred, self.y_ltop_pred = int(ball_x - ball_w / 2), int(ball_y - ball_h / 2)
            elif distance > math.sqrt((ball_x - self.x_c_pred) ** 2 + (ball_y - self.y_c_pred) ** 2):
                distance = math.sqrt((ball_x - self.x_c_pred) ** 2 + (ball_y - self.y_c_pred) ** 2)
                self.x_c_pred, self.y_c_pred = ball_x, ball_y
                self.x_ltop_pred, self.y_ltop_pred = int(ball_x - ball_w / 2), int(ball_y - ball_h / 2)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Predict")
        parser.add_argument("--input", required=True, type=str, help="Input video")
        args = parser.parse_args()
        return args

    def Detect_Trajectory(self, image):
        # 針對每一貞做運算
        image_CV = image.copy()

        # 計算順時速度
        if self.is_real_time_speed:
            self.real_time_speed = self.Estimate_Ball_Speed_kmh()

        # 計算方向
        q_array = np.array(self.q)
        non_negatives_idx = np.all(q_array != (-1, -1), axis=1)
        q_array = q_array[non_negatives_idx]
        if q_array.size == 0:
            x_tmp = np.array([])
            y_tmp = np.array([])
        else:
            x_tmp = q_array[:, 0]
            y_tmp = q_array[:, 1]

        # 累積有三顆球的軌跡且同一方向, 可計算拋物線
        if self.is_real_time_speed:
            self.real_time_past_ball_direction = self.real_time_ball_direction
        self.ball_direction, self.real_time_ball_direction = "unknown", "unknown"
        if len(x_tmp) >= 3:
            # 檢查是否嚴格遞增或嚴格遞減,(軌跡方向是否相同) x_tmp是左邊新右邊舊，所以要相反
            if self.is_first_ball:
                self.ball_direction = self.Monotonic(x_tmp[:4][::-1], strictly=False, half=False)
            else:
                self.ball_direction = self.Monotonic(x_tmp[:8][::-1], strictly=False, half=False)
            self.show_ball_direction = self.Monotonic(x_tmp[:2][::-1], strictly=False, half=False)
            if self.is_real_time_speed:
                self.real_time_ball_direction = self.Monotonic(
                    x_tmp[: self.real_time_ball_direction_reference_frame_size][::-1],
                    strictly=False,
                    half=False,
                    return_unknown=False,
                )

        ## 有偵測到球體
        if self.x_c_pred != None and self.y_c_pred != None:
            ## 落點預測 ######################################################################################################
            if self.ball_direction == "right" or self.ball_direction == "left":
                parabola = self.Solve_Parabola(x_tmp, y_tmp)
                a, b, c = parabola[0]
                fit = a * self.x_c_pred**2 + b * self.x_c_pred + c
                # cv2.circle(image_CV, (self.x_c_pred, int(fit)), 5, (255, 0, 0), 4)
                # 差距 10 個 pixel 以上視為脫離預測的拋物線
                if abs(self.y_c_pred - fit) >= 10:
                    x_last = x_tmp[0]
                    # 預測球在球桌上的落點, x_drop : 本次與前次的中點, y_drop : x_drop 於拋物線上的位置
                    x_drop = int(round((self.x_c_pred + x_last) / 2, 0))
                    y_drop = int(round(a * x_drop**2 + b * x_drop + c, 0))
                    # 繪製本次球體位置, Golden
                    cv2.circle(image_CV, (self.x_c_pred, self.y_c_pred), 5, (0, 215, 255), 4)
                    # 透視變形計算本次球體在迷你板上的位置
                    loc_PT = self.Perspective_Transform(self.matrix, (x_drop, y_drop))
                    # 如果變換後落在迷你板內
                    if (
                        loc_PT[0] >= self.miniboard_edge - 1
                        and loc_PT[0] < self.miniboard_width + self.miniboard_edge + 5
                        and loc_PT[1] >= self.miniboard_edge - 5
                        and loc_PT[1] < self.miniboard_height + self.miniboard_edge + 5
                    ):
                        self.PT_dict[self.count] = loc_PT
                        restart_list = list(self.PT_dict.keys())
                        """
                        一局結束判斷
                        1. 倒數兩球距離過大 (飛出界)
                        2. 停留在桌上 (被網子攔住)
                        """
                        if len(restart_list) >= 2 and (int(restart_list[-1]) - int(restart_list[-2])) > 200:
                            self.is_serve_wait = False
                            self.bounce_frame_L, self.bounce_frame_R = -1, -1
                            self.hit_count = 0
                            print(f"<---Frame : {self.count}, round end.--->")
                            self.img_opt = self.Draw_MiniBoard()
                        # 落點在左側
                        if self.PT_dict[self.count][0] <= int(self.miniboard_width / 2) + self.miniboard_edge:
                            # 首次發球 或 二次發球
                            if not self.is_serve_wait:
                                self.is_first_ball = True
                                self.is_serve_wait = True
                                self.hit_count = 1
                                self.now_player = 1  # switch player
                                self.bounce_frame_L = self.count
                                self.img_opt = self.Draw_MiniBoard()
                                self.Draw_and_Collect_Data(
                                    (0, 0, 255),
                                    loc_PT,
                                )

                            # 回擊
                            elif self.now_player == 0 and self.is_serve_wait:
                                if self.hit_count > 0:
                                    # cv2.line(
                                    #     self.img_opt,
                                    #     self.PT_dict[self.bounce_frame_R],
                                    #     self.PT_dict[self.count],
                                    #     (0, 255, 0),
                                    #     3,
                                    # )
                                    # 在miniboard上面兩顆球的距離 D2，單位是pixel
                                    bounce_len = self.Euclidean_Distance(
                                        self.PT_dict[self.bounce_frame_R][0],
                                        self.PT_dict[self.bounce_frame_R][1],
                                        self.PT_dict[self.count][0],
                                        self.PT_dict[self.count][1],
                                    )
                                    # D1的距離，單位是CM
                                    speed_bounce_distance_right = abs(
                                        self.shotspeed_previous
                                        * (100 / 1)
                                        * (self.right_shot_count - self.bounce_frame_R)
                                        / self.framerate
                                    )
                                    # miniboard轉成真實CM距離，加上上一球推測的距離，除以時間
                                    self.speed_right = np.round(
                                        (
                                            (bounce_len * (self.miniboard_to_real_ratio) + speed_bounce_distance_right)
                                            / (self.count - self.right_shot_count)
                                        )
                                        * self.framerate
                                        * (1 / 100),
                                        1,
                                    )
                                    if self.speed_right > 100:
                                        self.speed_right = 99

                                    self.shotspeed = self.speed_right
                                    self.shotspeed_previous = self.speed_right
                                    print(f"Frame : {self.count} self.speed_right : {self.speed_right} ")
                                    self.right_speed_list.append(self.speed_right)
                                    if self.is_show_speed_analysis:
                                        self.Draw_SpeedHist(save=False, show=self.is_show_speed_analysis)
                                self.is_first_ball = False
                                self.hit_count += 1
                                self.now_player = 1
                                self.bounce_frame_L = self.count
                                self.Draw_and_Collect_Data(
                                    (0, 0, 255),
                                    loc_PT,
                                )
                            # 其他
                            elif (self.count - self.bounce_frame_L) > 60:
                                print("[------------------------------------------------------------]")
                                print(
                                    f"sth wrong at frame : {self.count}, bounce_R : {self.bounce_frame_R}, self.hit_count : {self.hit_count}"
                                )
                                print("[------------------------------------------------------------]")
                                self.is_first_ball = False
                                self.is_serve_wait = True
                                self.now_player = 1
                                self.bounce_frame_L = self.count
                                self.hit_count = 1
                                self.img_opt = self.Draw_MiniBoard()
                                self.Draw_and_Collect_Data(
                                    (0, 0, 255),
                                    loc_PT,
                                )

                        # 落點在右側
                        elif self.PT_dict[self.count][0] >= int(self.miniboard_width / 2) + self.miniboard_edge:
                            # 首次發球 或 二次發球
                            if not self.is_serve_wait:
                                self.is_first_ball = True
                                self.is_serve_wait = True
                                self.hit_count = 1
                                self.now_player = 0  # switch player
                                self.bounce_frame_R = self.count
                                self.img_opt = self.Draw_MiniBoard()
                                self.Draw_and_Collect_Data(
                                    (80, 127, 255),
                                    loc_PT,
                                )

                            # 回擊
                            elif self.now_player == 1 and self.is_serve_wait:
                                if self.hit_count > 0:
                                    # like yellow
                                    # cv2.line(
                                    #     self.img_opt,
                                    #     self.PT_dict[self.bounce_frame_L],
                                    #     self.PT_dict[self.count],
                                    #     (115, 220, 255),
                                    #     3,
                                    # )
                                    bounce_len = self.Euclidean_Distance(
                                        self.PT_dict[self.bounce_frame_L][0],
                                        self.PT_dict[self.bounce_frame_L][1],
                                        self.PT_dict[self.count][0],
                                        self.PT_dict[self.count][1],
                                    )
                                    speed_bounce_distance_left = abs(
                                        self.shotspeed_previous
                                        * (100 / 1)
                                        * (self.left_shot_count - self.bounce_frame_L)
                                        / self.framerate
                                    )
                                    self.speed_left = np.round(
                                        (
                                            (bounce_len * (self.miniboard_to_real_ratio) + speed_bounce_distance_left)
                                            / (self.count - self.left_shot_count)
                                        )
                                        * self.framerate
                                        * (1 / 100),
                                        1,
                                    )
                                    if self.speed_left > 100:
                                        self.speed_left = 60

                                    self.shotspeed = self.speed_left
                                    self.shotspeed_previous = self.speed_left
                                    print(f"Frame : {self.count} self.speed_left : {self.speed_left} ")
                                    self.left_speed_list.append(self.speed_left)
                                    if self.is_show_speed_analysis:
                                        self.Draw_SpeedHist(save=False, show=self.is_show_speed_analysis)
                                self.is_first_ball = False
                                self.hit_count += 1
                                self.now_player = 0
                                self.bounce_frame_R = self.count
                                self.Draw_and_Collect_Data(
                                    (80, 127, 255),
                                    loc_PT,
                                )

                            # 其他
                            elif (self.count - self.bounce_frame_R) > 60:
                                print("[------------------------------------------------------------]")
                                print(
                                    f"sth wrong at frame : {self.count}, bounce_L : {self.bounce_frame_L}, self.hit_count : {self.hit_count}"
                                )
                                print("[------------------------------------------------------------]")
                                self.is_first_ball = False
                                self.is_serve_wait = True
                                self.now_player = 0
                                self.bounce_frame_R = self.count
                                self.hit_count = 1
                                self.img_opt = self.Draw_MiniBoard()
                                self.Draw_and_Collect_Data(
                                    (80, 127, 255),
                                    loc_PT,
                                )

            ## 超過一秒都沒有球落在球桌上
            if (self.count - self.bounce_frame_L) >= 60 and (self.count - self.bounce_frame_R) >= 60:  # 超過1秒
                self.is_first_ball = True
                self.is_serve_wait = True
                self.bounce_frame_L, self.bounce_frame_R = -1, -1
                self.hit_count = 0

        return image_CV

    def Add_Ball_In_Queue(self):
        self.q.appendleft(
            (self.x_c_pred, self.y_c_pred) if self.x_c_pred != None and self.y_c_pred != None else (-1, -1)
        )
        self.q.pop()

        self.q_bv.appendleft((-1, -1))
        self.q_bv.pop()

    def Detect_Ball_Direction(self):
        ball_direction, ball_direction_last = None, None
        if self.q[0] != (-1, -1) and self.q[1] != (-1, -1) and self.q[2] != (-1, -1):
            ball_direction = self.q[0][0] - self.q[1][0]
            ball_direction_last = self.q[1][0] - self.q[2][0]
            if self.MAX_velo == 0:
                self.MAX_velo = self.shotspeed
            if ball_direction > 0:  # Direction right
                if ball_direction_last >= 0:
                    self.right_shot_count = self.count
                    if self.shotspeed > self.MAX_velo:
                        self.MAX_velo = self.shotspeed
                else:
                    self.MAX_velo = 0

            elif ball_direction < 0:  # Direction left
                if ball_direction_last <= 0:
                    self.left_shot_count = self.count
                    if self.shotspeed > self.MAX_velo:
                        self.MAX_velo = self.shotspeed
                else:
                    self.MAX_velo = 0

        return ball_direction, ball_direction_last

    def Draw_On_Image(self, image_CV):
        # draw current frame prediction and previous 11 frames as yellow circle, total: 12 frames
        for i in range(12):
            if self.q[i] != (-1, -1):
                cv2.circle(image_CV, (self.q[i][0], self.q[i][1]), 5, (0, 255, 255), 1)

        # draw bounce point as red circle
        for i in range(6):
            if self.q_bv[i] != (-1, -1):
                cv2.circle(image_CV, (self.q_bv[i][0], self.q_bv[i][1]), 5, (0, 0, 255), 4)

        # Place miniboard on upper right corner
        if self.is_show_bounce:
            self.Show_Bounce()
            if not self.is_show_bounce_window:
                image_CV[
                    : self.miniboard_height + self.miniboard_edge * 2,
                    self.frame_width - (self.miniboard_width + self.miniboard_edge * 2) :,
                ] = self.img_opt

        # 順時速度
        image_CV_real_time_speed = image_CV.copy()
        if self.is_real_time_speed:
            if self.real_time_speed:
                cv2.putText(
                    image_CV_real_time_speed,
                    f"{self.real_time_speed:0.1f}(m/s)",
                    (self.x_ltop_pred, self.y_ltop_pred),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1.0,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 將球的方向判斷出來
        # if self.show_ball_direction == "right":  # Direction right
        #     cv2.putText(
        #         image_CV,
        #         "right",
        #         (240, 100),
        #         cv2.FONT_HERSHEY_TRIPLEX,
        #         1,
        #         (0, 255, 255),
        #         1,
        #         cv2.LINE_AA,
        #     )
        # elif self.show_ball_direction == "left":  # Direction left
        #     cv2.putText(
        #         image_CV,
        #         "left",
        #         (240, 100),
        #         cv2.FONT_HERSHEY_TRIPLEX,
        #         1,
        #         (0, 255, 255),
        #         1,
        #         cv2.LINE_AA,
        #     )

        # # 標示出球速
        if self.MAX_velo > 113:
            cv2.putText(
                image_CV,
                "          " + "Loss",
                (10, 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        elif self.show_ball_direction != "unknown":
            cv2.putText(
                image_CV,
                "          " + str(self.shotspeed),
                (10, 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        # 無法辨別球路方向時
        else:
            cv2.putText(
                image_CV,
                "          " + "0",
                (10, 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # # 其他左上角的文字
        cv2.putText(
            image_CV,
            "Speed:",
            (10, 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_CV,
            "(m/s)",
            (260, 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # cv2.putText(
        #     image_CV,
        #     "Direction :",
        #     (10, 100),
        #     cv2.FONT_HERSHEY_TRIPLEX,
        #     1,
        #     (0, 255, 255),
        #     1,
        #     cv2.LINE_AA,
        # )
        # cv2.putText(
        #     image_CV,
        #     f"Frame : {self.count}",
        #     (10, 160),
        #     cv2.FONT_HERSHEY_TRIPLEX,
        #     1,
        #     (0, 255, 255),
        #     1,
        #     cv2.LINE_AA,
        # )

        # 右下角顯示
        image_CV_height, image_CV_width, _ = image_CV.shape
        bounce_analyze_img = self.bounce_analyze_img
        set_height = 325
        set_radio = 584 / 325
        cv2.resize(bounce_analyze_img, (int(set_height), int(set_height * set_radio)))
        bounce_analyze_img_height, bounce_analyze_img_width, _ = bounce_analyze_img.shape
        image_CV[
            image_CV_height - bounce_analyze_img_height : image_CV_height,
            image_CV_width - bounce_analyze_img_width : image_CV_width,
            :,
        ] = bounce_analyze_img

        img_opt_bounce_location = self.img_opt_bounce_location
        set_height = 325
        set_radio = 584 / 325
        cv2.resize(img_opt_bounce_location, (int(set_height), int(set_height * set_radio)))
        img_opt_bounce_location_height, img_opt_bounce_location_width, _ = img_opt_bounce_location.shape
        image_CV[
            image_CV_height - img_opt_bounce_location_height : image_CV_height,
            image_CV_width
            - bounce_analyze_img_width
            - img_opt_bounce_location_width : image_CV_width
            - bounce_analyze_img_width,
            :,
        ] = img_opt_bounce_location

        return image_CV, image_CV_real_time_speed

    def Write_Bounce_Location(self):
        if self.is_write_bounce_location:
            bounce_loc_pd = pd.DataFrame(self.bounce_location_list)
            bounce_loc_pd.to_csv(f"{self.bounce_loc_path}/{self.video_name}_bounce_list.csv", index=False)

    def Draw_Speed_Under_Ball(self, image):
        if self.count in self.record_ball:
            # word position
            x_c_pred, y_c_pred, speed = (
                self.record_ball[self.count]["x_c_pred"],
                self.record_ball[self.count]["y_c_pred"],
                self.record_ball[self.count]["speed"],
            )
            interval = 50
            word_x, word_y = x_c_pred, y_c_pred + interval

            # text style
            text = str(speed)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2

            # center
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            word_x, word_y = word_x - text_size[0] // 2, word_y + text_size[1] // 2

            # add text
            cv2.putText(image, text, (word_x, word_y), font, font_scale, (255, 255, 255), font_thickness)
        else:
            print(self.count)
        return image

    def __init__(self, real_time=False):
        # temp#
        self.only_speed = False

        self.HEIGHT = 288  # model input size
        self.WIDTH = 512

        # 影片跟目錄
        root_path = f"./runs/detect/105_match_01_20250514"
        video_fullname = "105_match_01.mp4"
        self.video_name = os.path.splitext(video_fullname)[0]
        self.video_suffix = os.path.splitext(video_fullname)[1]
        self.input_path = os.path.join(root_path, video_fullname)

        # 建立目錄
        output_path = f"./inference/output"
        (
            self.video_path,
            record_ball_path,
            record_pose_path,
            self.analysis_img_path,
            self.bounce_loc_path,
            self.bounce_img_path,
            keypoints_path,
            self.speedhis_path,
            self.speed_distribution_path,
        ) = self.Create_Output_Dir(output_path, self.video_name)

        # yolo labels path
        self.label_path = os.path.join(root_path, "labels")

        # miniboard 的大小
        self.miniboard_width = 544  # 原先為548
        self.miniboard_height = 285  # 原先為305
        self.miniboard_edge = 20
        self.miniboard_text_bias = 60

        # 真實桌球桌子大小
        self.real_table_width_cm = 274.0
        self.real_table_height_cm = 152.5

        # 虛擬球桌(pixel) 轉 真實球桌大小(cm)
        self.miniboard_to_real_ratio = self.real_table_width_cm / (self.miniboard_width + 2 * self.miniboard_edge)

        # 透視變形
        self.PT_dict = {}

        # In order to draw the trajectory of tennis, we need to save the coordinate of preious 12 frames
        self.q = queue.deque([(-1, -1) for _ in range(12)])

        # bounce detection init
        self.q_bv = queue.deque([(-1, -1) for _ in range(6)])

        # 參數
        self.bounce = []
        self.left_speed_list, self.right_speed_list = [], []
        self.bounce_location_list = np.zeros((4, 3), dtype=int)
        self.bouncing_offset_x, self.bouncing_offset_y = 10, 15  # bouncing location offset
        self.speed_left, self.speed_right = 0, 0  # 左右選手球速
        self.bounce_frame_L, self.bounce_frame_R = -1, -1  # 出現落點的Frame
        self.left_shot_count, self.right_shot_count = 0, 0  # 左右選手擊球時的frame number
        self.now_player = 0  # 0:左邊選手, 1: 右邊選手
        self.hit_count = 0  # 擊球次數
        self.count = 1  # 記錄處理幾個 Frame
        self.MAX_velo = 0  # 最大球速
        self.past_x_c_pred, self.past_y_c_pred = None, None  # 球體上一次中心位置
        self.x_c_pred, self.y_c_pred = None, None  # 球體中心位置
        self.is_first_ball = True  # 每局第一球的時候frame只要5個，其他時間要9個
        self.is_serve_wait = False
        self.shotspeed = 0
        self.shotspeed_previous = 0

        # 顯示參數
        self.is_write_bounce_location = False  # True
        self.is_save_bounce_analysis = False  # True
        self.is_save_bounce_location = False  # True
        self.is_show_bounce = False
        self.is_show_bounce_window = False
        self.is_show_bounce_analysis = False
        self.is_show_bounce_location = False
        self.is_show_speed_analysis = False
        if real_time:
            self.is_show_bounce_window = False
            if self.is_show_bounce_window:
                self.bounce_title = "Bounce"
                cv2.namedWindow(self.bounce_title, cv2.WINDOW_NORMAL)
            self.is_show_bounce_analysis = False
            if self.is_show_bounce_analysis:
                self.bounce_analysis_title = "Bounce Analysis"
                cv2.namedWindow(self.bounce_analysis_title, cv2.WINDOW_NORMAL)
            self.is_show_bounce_location = False
            if self.is_show_bounce_location:
                self.bounce_location_title = "Bounce Location"
                cv2.namedWindow(self.bounce_location_title, cv2.WINDOW_NORMAL)
            self.is_show_speed_analysis = False
            if self.is_show_speed_analysis:
                self.speedhis_title = "Speed Histogram"
                cv2.namedWindow(self.speedhis_title, cv2.WINDOW_NORMAL)
                self.speed_distribution_title = "Speed Distribution"
                cv2.namedWindow(self.speed_distribution_title, cv2.WINDOW_NORMAL)
                self.Draw_SpeedHist(save=False, show=self.is_show_speed_analysis)
            self.video_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.ball_direction = "unknown"  # 球當下的方向(給拋物線用)
        self.show_ball_direction = "unknown"  # 球當下的方向(給顯示用的)
        self.is_real_time_speed = True  # 使否即時顯示球速
        if self.is_real_time_speed:
            self.past_c_frame_number = 1  # 上一次取得球的frame
            self.real_time_ball_direction = "unknown"  # 球當下的方向(給realtime speed)
            self.real_time_past_ball_direction = "unknown"  # 上一次球的方向(給realtime speed)
            self.real_time_ball_direction_reference_frame_size = 8  # 參考多少個frame決定方向
            self.real_time_speed = None  # 及時球速
            self.real_time_speed_index_head = 0  # 擷取影像的分段標籤(頭)
            self.real_time_speed_index_last = 0  # 擷取影像的分段標籤(尾)
            self.real_time_speed_save_tag = False  # 是否可以儲存影片
            self.last_frame_switch_ball_direction = 1  # 最後變換方向的frame
            self.save_queue_threadhold_by_switch_ball_direction = 120  # 多少個frame沒有變換方向則儲存
            self.save_queue_minimum_frame_size = 10  # 至少有多少個frame才儲存影片
            self.delay_frame_queue = deque()  # 因為方向在後real_time_ball_direction_reference_frame_size個frame才能決定
            self.control_queue = mp.Queue()  # 儲存影片
            self.real_time_speed_root_path = f"{self.video_path}/{self.video_name}"
            os.makedirs(self.real_time_speed_root_path, exist_ok=True)
            self.real_time_speed_process = mp.Process(
                target=self.Real_Time_Speed_Process,
                args=(self.real_time_speed_root_path, self.control_queue, self.save_queue_minimum_frame_size),
            )
            self.real_time_speed_process.start()

    def Set_Frame_Info(self, frame_height, frame_width, framerate):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.framerate = framerate

    def Next_Count(self):
        self.count += 1

    ### 後處理從此開始 ###
    def main(self):
        # 讀影片
        success, image, cap, framerate, frame_height, frame_width, total_frames = self.Read_Video(self.input_path)
        self.Set_Frame_Info(frame_height, frame_width, framerate)

        # 等比例縮放
        ratio = self.frame_height / self.HEIGHT
        size = (int(self.WIDTH * ratio), int(self.HEIGHT * ratio))

        # 寫 預測結果
        video_path = f"{self.video_path}/{self.video_name}_predict_12.mp4"
        output = self.Write_Video(video_path, size)

        # 透視變形
        self.Mark_Perspective_Distortion_Point(image, self.frame_width, self.frame_height)

        # 針對每一貞做運算
        start = time.time()
        batch = 12
        n = 4
        k = batch // 2
        while success:
            label_file = os.path.join(self.label_path, f"{self.video_name}_{self.count}.txt")
            self.Read_Yolo_Label_One_Frame(label_file=label_file)
            image_CV = self.Detect_Trajectory(image)
            self.Add_Ball_In_Queue()
            _ = self.Detect_Ball_Direction()
            image_CV, image_CV_real_time_speed = self.Draw_On_Image(image_CV)
            if self.is_real_time_speed:
                real_time_speed_index_last = self.Generate_Real_Time_Speed_index()
                self.Add_Frame_In_Delay_Queue(real_time_speed_index_last, image_CV_real_time_speed)
                is_save_real_time_speed = self.Is_Save_Queue()
                if is_save_real_time_speed:
                    self.Raise_Save_Queue()

            self.Next_Count()
            if self.count >= total_frames - 12:
                break
            output.write(image_CV_real_time_speed)
            success, image = cap.read()

        # For releasing cap and out.
        cap.release()
        output.release()

        # write bouncing list to csv file
        self.Write_Bounce_Location()

        # output bouncing analyze img
        self.Draw_Bounce_Analysis()
        self.Save_Bounce_Analysis()

        # For saving bounce map.
        self.Save_Bounce_Location()

        # For saving speedHist
        self.Draw_SpeedHist()

        # 離開
        self.Exit_Save_Queue()

        end = time.time()
        print(f"Write video time: {end-start} seconds.")
        total_time = end - start

        print()
        print(f"Detect Result is saved in {self.video_path}")
        print(f"Total time: {total_time} seconds")
        print(f"Done......")


if __name__ == "__main__":
    trajectory = Trajectory()
    trajectory.main()
