import argparse
import datetime
import json
import math
import os
import queue
import re
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pandas as pd
from scipy.optimize import leastsq


class Trajectory():
    def PJcurvature(self,x, y):
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


    def Custom_Time(self,time):
        # time: in milliseconds
        seconds, milliseconds = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)

        # 格式化時間
        cts = "{:02d}:{:02d}:{:02d}.{:03d}".format(hours, minutes, seconds, milliseconds)
        return cts

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
            return half_strictly_increasing or half_strictly_decreasing
        else:
            if strictly:
                strictly_increasing = np.all(L[1:] > L[:-1])
                strictly_decreasing = np.all(L[1:] < L[:-1])
                return strictly_increasing or strictly_decreasing
            else:
                non_increasing = np.all(L[1:] >= L[:-1])
                non_decreasing = np.all(L[1:] <= L[:-1])
                return non_increasing or non_decreasing


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
                    color_detect = self.Detect_Color_Level(score_table[i][j], "left", left_score_min, left_score_max)
                else:
                    # right
                    color_detect = self.Detect_Color_Level(score_table[i][j], "right", right_score_min, right_score_max)
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
    def Show_Bounce_Analysis(self):
        cv2.imshow(self.bounce_analysis_title, self.bounce_analyze_img)

    def Save_Bounce_Analysis(self):
        cv2.imwrite(
            f"{self.analysis_img_path}/{self.video_name}_analysis.jpg",
            self.bounce_analyze_img,
        )

    def Show_Bounce_Location(self):
        cv2.imshow(self.bounce_location_title, self.img_opt_bounce_location)

    def Save_Bounce_Location(self):
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
        plt.ylabel(f"Km/hr", fontsize=30, labelpad=20)
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

        plt.xlabel(f"Km/hr", fontsize=30, labelpad=15)
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
        img_opt = np.zeros([self.miniboard_height + self.miniboard_edge * 2, self.miniboard_width + self.miniboard_edge * 2, 3], dtype=np.uint8)
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
                (int((self.miniboard_width / 4) * 3) + self.miniboard_edge, self.miniboard_height + self.miniboard_edge),
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
                (self.miniboard_width + self.miniboard_edge, int((self.miniboard_height / 3) * 2) + self.miniboard_edge),
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
        self.Count_BounceLocation(
            self.PT_dict[self.count]
        )
        if self.is_show_bounce_analysis:
            self.Draw_Bounce_Analysis()
            self.Show_Bounce_Analysis()
        if self.is_show_bounce_location:
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
            speed_distribution_path
        )

    def Read_Video(self):
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
        cap = cv2.VideoCapture(self.input_path)
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
            f"{video_path}/{self.video_name}_predict_12.mp4",
            fourcc,
            self.framerate,
            size,
        )

        return output

    def Mark_Perspective_Distortion_Point(self, image, frame_width, frame_height):
        # 點選透視變形位置, 順序為:左上,左下,右下,右上
        PT_data = {"img": image.copy(), "point_x": [], "point_y": []}
        # TODO: 測試用
        # PT_data["point_x"] = [488,432,1383,1319]
        # PT_data["point_y"] = [675,789,796,679]
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

        # 顯示
        if self.is_show_bounce_analysis:
            self.Draw_Bounce_Analysis()
            self.Show_Bounce_Analysis()
        if self.is_show_bounce_location:
            self.Show_Bounce_Location()


    def Read_Yolo_Label_One_Frame(self, label_file=None, balls=None, x_c_pred = None, y_c_pred = None):
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
            if self.x_c_pred == np.inf or self.y_c_pred == np.inf:
                self.x_c_pred, self.y_c_pred = ball_x_c_pred, ball_y_c_pred
            elif distance > math.sqrt((ball_x_c_pred - self.x_c_pred) ** 2 + (ball_y_c_pred - self.y_c_pred) ** 2):
                distance = math.sqrt((ball_x_c_pred - self.x_c_pred) ** 2 + (ball_y_c_pred - self.y_c_pred) ** 2)
                self.x_c_pred, self.y_c_pred = ball_x_c_pred, ball_y_c_pred

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Predict")
        parser.add_argument("--input", required=True, type=str, help="Input video")
        args = parser.parse_args()
        return args

    def Detect_Trajectory(self, image):
        # 針對每一貞做運算
        image_CV = image.copy()

        ## 有偵測到球體
        if self.x_c_pred!=np.inf and self.y_c_pred!=np.inf:
            balls = 5 if self.is_first_ball else 9
            x_tmp = [self.q[j][0] for j in range(balls) if self.q[j] is not None]
            y_tmp = [self.q[j][1] for j in range(balls) if self.q[j] is not None]
            ## 落點預測 ######################################################################################################
            if len(x_tmp) >= 3:
                # 檢查是否嚴格遞增或嚴格遞減,(軌跡方向是否相同)
                isSameWay= self.Monotonic(np.array(x_tmp), strictly=False, half=True)
                # 累積有三顆球的軌跡且同一方向, 可計算拋物線
                if isSameWay:
                    parabola = self.Solve_Parabola(np.array(x_tmp), np.array(y_tmp))
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
                                        cv2.line(
                                            self.img_opt,
                                            self.PT_dict[self.bounce_frame_R],
                                            self.PT_dict[self.count],
                                            (0, 255, 0),
                                            3,
                                        )
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
                                            * (100000 / 3600)
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
                                            * (3600 / 100000),
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
                                        cv2.line(
                                            self.img_opt,
                                            self.PT_dict[self.bounce_frame_L],
                                            self.PT_dict[self.count],
                                            (115, 220, 255),
                                            3,
                                        )
                                        bounce_len = self.Euclidean_Distance(
                                            self.PT_dict[self.bounce_frame_L][0],
                                            self.PT_dict[self.bounce_frame_L][1],
                                            self.PT_dict[self.count][0],
                                            self.PT_dict[self.count][1],
                                        )
                                        speed_bounce_distance_left = abs(
                                            self.shotspeed_previous
                                            * (100000 / 3600)
                                            * (self.left_shot_count - self.bounce_frame_L)
                                            / self.framerate
                                        )
                                        self.speed_left = np.round(
                                            (
                                                (bounce_len * (self.miniboard_to_real_ratio) + speed_bounce_distance_left)
                                                / (self.count - self.left_shot_count)
                                            )
                                            * self.framerate
                                            * (3600 / 100000),
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
        self.q.appendleft((self.x_c_pred, self.y_c_pred) if self.x_c_pred!=np.inf and self.y_c_pred!=np.inf else None)
        self.q.pop()

        self.q_bv.appendleft(None)
        self.q_bv.pop()

    def Detect_Ball_Direction(self):
        ball_direction, ball_direction_last = None, None
        if self.q[0] is not None and self.q[1] is not None and self.q[2] is not None:
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


    def Draw_On_Image(self, image_CV, ball_direction):
        # draw current frame prediction and previous 11 frames as yellow circle, total: 12 frames
        for i in range(12):
            if self.q[i] is not None:
                cv2.circle(image_CV, (self.q[i][0], self.q[i][1]), 5, (0, 255, 255), 1)

        # draw bounce point as red circle
        for i in range(6):
            if self.q_bv[i] is not None:
                cv2.circle(image_CV, (self.q_bv[i][0], self.q_bv[i][1]), 5, (0, 0, 255), 4)

        # Place miniboard on upper right corner
        image_CV[
            : self.miniboard_height + self.miniboard_edge * 2,
            self.frame_width - (self.miniboard_width + self.miniboard_edge * 2) :,
        ] = self.img_opt

        # 將球的方向判斷出來
        if ball_direction != None and ball_direction > 0:  # Direction right
            cv2.putText(
                image_CV,
                "right",
                (240, 180),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        elif ball_direction != None and ball_direction < 0:  # Direction left
            cv2.putText(
                image_CV,
                "left",
                (240, 180),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 標示出球速
        if self.MAX_velo > 113:
            cv2.putText(
                image_CV,
                "              " + "Loss",
                (10, 100),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        elif ball_direction is not None:
            cv2.putText(
                image_CV,
                "              " + str(self.shotspeed),
                (10, 100),
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
                "              " + "0",
                (10, 100),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 其他左上角的文字
        cv2.putText(
            image_CV,
            "velocity:",
            (10, 100),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_CV,
            "(km/hr)",
            (10, 140),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_CV,
            "direction :",
            (10, 180),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_CV,
            f"Frame : {self.count}",
            (10, 260),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return image_CV

    def Write_Bounce_Location(self):
        bounce_loc_pd = pd.DataFrame(self.bounce_location_list)
        bounce_loc_pd.to_csv(f"{self.bounce_loc_path}/{self.video_name}_bounce_list.csv", index=False)

    def __init__(self,real_time=False):
        self.HEIGHT = 288  # model input size
        self.WIDTH = 512

        # 影片跟目錄
        root_path = f"./runs/detect/C00050015"
        video_fullname = "C00050015.mp4"
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
            self.speed_distribution_path
        ) = self.Create_Output_Dir(output_path, self.video_name)

        # yolo labels path
        self.label_path = os.path.join(root_path, "labels")

        # miniboard 的大小
        self.miniboard_width = 544 # 原先為548
        self.miniboard_height = 285 # 原先為305
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
        self.q = queue.deque([None for _ in range(12)])

        # bounce detection init
        self.q_bv = queue.deque([None for _ in range(6)])

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
        self.MAX_velo = 0 # 最大球速
        self.x_c_pred, self.y_c_pred = np.inf, np.inf  # 球體中心位置
        self.is_first_ball = True #每局第一球的時候frame只要5個，其他時間要9個
        self.is_serve_wait = False
        self.shotspeed = 0
        self.shotspeed_previous = 0

        # 顯示參數
        self.is_show_bounce_analysis = False
        self.is_show_bounce_location = False
        self.is_show_speed_analysis = False
        if real_time:
            self.is_show_bounce_analysis = True
            self.bounce_analysis_title = "Bounce Analysis"
            cv2.namedWindow(self.bounce_analysis_title, cv2.WINDOW_NORMAL)
            self.is_show_bounce_location = True
            self.bounce_location_title = "Bounce Location"
            cv2.namedWindow(self.bounce_location_title, cv2.WINDOW_NORMAL)
            self.is_show_speed_analysis = True
            self.speedhis_title = "Speed Histogram"
            cv2.namedWindow(self.speedhis_title, cv2.WINDOW_NORMAL)
            self.speed_distribution_title = "Speed Distribution"
            cv2.namedWindow(self.speed_distribution_title, cv2.WINDOW_NORMAL)
            self.Draw_SpeedHist(save=False, show=True)
            self.video_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def Set_Frame_Info(self,frame_height, frame_width, framerate):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.framerate = framerate

    def Next_Count(self):
        self.count += 1

    ### 後處理從此開始 ###
    def main(self):
        # 讀影片
        success, image, cap, framerate, frame_height, frame_width, total_frames = self.Read_Video()
        self.Set_Frame_Info(frame_height, frame_width, framerate)

        # 等比例縮放
        ratio = self.frame_height / self.HEIGHT
        size = (int(self.WIDTH * ratio), int(self.HEIGHT * ratio))

        # 寫 預測結果
        output = self.Write_Video(self.video_path, size)

        # 透視變形
        self.Mark_Perspective_Distortion_Point(image, self.frame_width, self.frame_height)

        # 針對每一貞做運算
        start = time.time()
        batch = 12
        n = 4
        k = batch // 2
        while success:
            label_file = os.path.join(self.label_path, f"{self.video_name}_{self.count}.txt")
            if os.path.exists(label_file):
                self.Read_Yolo_Label_One_Frame(label_file=label_file)
            if self.count == 568:
                print("test")
            image_CV = self.Detect_Trajectory(image)
            self.Add_Ball_In_Queue()
            ball_direction, ball_direction_last = self.Detect_Ball_Direction()
            image_CV = self.Draw_On_Image(image_CV, ball_direction)

            self.Next_Count()
            if self.count >= total_frames - 12:
                break
            output.write(image_CV)
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
