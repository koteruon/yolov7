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
from tqdm import tqdm


class Ball:
    def __init__(self, center, bbox_size, frame_number):
        self.ball_limit = 5  # 限制最多儲存多少歷史紀錄
        self.trajectory_ball_limit = 9 # 限制最多儲存多少歷史紀錄用於推算軌跡落點
        self.show_ball_limit = 12 # 限制最多儲存多少歷史紀錄用於畫畫用
        self.frame_limit = 30 # 限制相隔多少frame以後自動刪除
        self.center_history = [np.array(center)]  # 初始化中心點歷史紀錄
        self.trajectory_center_history = [np.array(center)] # 初始化中心點歷史紀錄用於推算軌跡落點
        self.show_center_history = [np.array(center)] # 初始化中心點歷史紀錄用於畫畫用
        self.direction_history = [np.array([0, 0])]  # 初始化運動方向歷史紀錄
        self.bbox_size_history = [bbox_size]  # 初始化BBOX大小歷史紀錄
        self.iou_history = [0]  # 初始化IOU歷史紀錄
        self.has_bounced = False
        self.new_center = np.array(center)
        self.new_bbox_size = np.array(bbox_size)
        self.new_frame_number = frame_number

        # 初始化平均值
        self.average_center = np.array(center)
        self.average_direction = np.array([0, 0])
        self.average_bbox_size = bbox_size
        self.average_iou = 0
        self.new_bbox = self.calculate_new_bbox(self.new_center, self.new_bbox_size)

    def no_detect_update_position(self):
        center = (-1, -1)
        self._update_history(self.trajectory_ball_limit, self.trajectory_center_history, center)
        self._update_history(self.show_ball_limit, self.show_center_history, center)

    def update_position(self, new_center, new_bbox_size, new_frame_number, iou):
        self.new_center = np.array(new_center)
        self.new_bbox_size = np.array(new_bbox_size)
        self.new_frame_number = new_frame_number
        new_direction = self.new_center - self.average_center  # 使用平均位置來計算運動方向

        # 更新歷史紀錄
        self._update_history(self.ball_limit, self.center_history, self.new_center)
        self._update_history(self.trajectory_ball_limit, self.trajectory_center_history, self.new_center)
        self._update_history(self.show_ball_limit, self.show_center_history, self.new_center)
        self._update_history(self.ball_limit, self.direction_history, new_direction)
        self._update_history(self.ball_limit, self.bbox_size_history, self.new_bbox_size)
        self._update_history(self.ball_limit, self.iou_history, iou)  # 更新 IOU 歷史紀錄

        # 計算並儲存最新的平均值
        self.average_center = self.get_average_position()
        self.average_direction = self.get_average_direction()
        self.average_bbox_size = self.get_average_bbox_size()
        self.average_iou = self.get_average_iou()
        self.new_bbox = self.calculate_new_bbox(self.new_center, self.new_bbox_size)

    def _update_history(self, limit, history, new_value):
        # 更新歷史紀錄，保留最新的 ball_limit 個記錄
        history.append(new_value)
        if len(history) > limit:
            history.pop(0)  # 刪除最早的記錄

    def get_average_position(self):
        return np.mean(self.center_history, axis=0)

    def get_average_direction(self):
        return np.mean(self.direction_history, axis=0)

    def get_average_bbox_size(self):
        return np.mean(self.bbox_size_history, axis=0)

    def get_average_iou(self):
        return np.mean(self.iou_history)

    def bounce(self):
        self.has_bounced = True

    # Sigmoid 函数映射
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-5 * (x - 0.5)))  # 调整参数以控制形状

    def calculate_weights(self, frame_width):
        # 計算權重
        historical_iou = self.get_average_iou()
        iou_weight = self.sigmoid(historical_iou)  # IOU 的權重
        x_position_ratio = self.average_center[0] / frame_width  # 球在畫面中的x位置比率 [0, 1]
        distance_weight = 0.2 + x_position_ratio * 0.5  # 離右邊越近，距離權重越高
        direction_weight = 0.05 + x_position_ratio * 0.65  # 離右邊越近，方向權重越高
        remaining_weight = 1.0 - (iou_weight  + distance_weight + direction_weight)
        aspect_ratio_weight = max(0.3, remaining_weight)
        total_weight = iou_weight + direction_weight + distance_weight + aspect_ratio_weight

        # 標準化權重
        if total_weight > 0:
            iou_weight /= total_weight
            distance_weight /= total_weight
            direction_weight /= total_weight
            aspect_ratio_weight /= total_weight

        return iou_weight, direction_weight, distance_weight, aspect_ratio_weight

    def calculate_new_bbox(self, center, bbox_size):
        ball_bbox = [
                    center[0] - bbox_size[0] / 2,
                    center[1] - bbox_size[1] / 2,
                    center[0] + bbox_size[0] / 2,
                    center[1] + bbox_size[1] / 2
                ]
        return ball_bbox

class BallTracker:
    def __init__(self):
        self.balls = []  # 儲存所有球的資訊

    def set_frame_info(self,frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def add_ball(self, center, bbox_size, frame_number):
        new_ball = Ball(center, bbox_size, frame_number)
        self.balls.append(new_ball)

    def remove_non_tracking_ball(self, frame_number):
        for ball in self.balls:
            if frame_number - ball.new_frame_number > ball.frame_limit:
                self.balls.remove(ball)

    def yolo2ball(self, bbox):
        # 提取 YOLO 格式的數據
        _, x_center, y_center, width, height, _ = bbox
        ball_x_center, ball_y_center = int(float(x_center) * self.frame_width), int(
            float(y_center) * self.frame_height
        )
        ball_width, ball_height = int(float(width) * self.frame_width), int(
            float(height) * self.frame_height
        )
        center = (ball_x_center, ball_y_center)
        bbox_size = (ball_width, ball_height)

        ball_bbox = [
                center[0] - bbox_size[0] / 2,
                center[1] - bbox_size[1] / 2,
                center[0] + bbox_size[0] / 2,
                center[1] + bbox_size[1] / 2
            ]

        return center, bbox_size, ball_bbox

    def no_detect_update_balls(self, frame_number):
        self.remove_non_tracking_ball(frame_number)
        for ball in self.balls:
            ball.no_detect_update_position()

    def update_balls(self, detected_bboxes, frame_number):
        self.remove_non_tracking_ball(frame_number)

        all_scores = []  # 儲存所有檢測框和所有球的配對及其分數

        for bbox in detected_bboxes:
            center, bbox_size, ball_bbox = self.yolo2ball(bbox)
            iou = 0

            # 遍歷所有已知的球，計算該檢測框和每個球的配對得分
            has_match = False
            for ball in self.balls:
                # 計算 IOU
                iou = self.calculate_iou(ball.new_bbox, ball_bbox)

                # 計算方向性
                direction_similarity = np.dot(ball.average_direction, (center - ball.average_center)) / (
                    np.linalg.norm(ball.average_direction) * np.linalg.norm(center - ball.average_center)) if np.linalg.norm(ball.average_direction) > 0 else 0
                direction_similarity = max(0, min(1, direction_similarity))  # 保證方向相似度在 [0, 1] 範圍內

                # 計算長寬比差異
                aspect_ratio_ball = ball.average_bbox_size[0] / ball.average_bbox_size[1]
                aspect_ratio_new = bbox_size[0] / bbox_size[1]
                aspect_ratio_diff = abs(aspect_ratio_ball - aspect_ratio_new)

                # 計算距離
                distance = np.linalg.norm(ball.average_center - center)

                # 獲取權重
                iou_weight, direction_weight, distance_weight, aspect_ratio_weight = ball.calculate_weights(self.frame_width)

                # 計算各個因素的分數
                iou_score = iou  # IOU 越大分數越高
                direction_score = direction_similarity  # 方向性越相似分數越高
                # 規範化距離分數
                max_distance = self.frame_width  # 假設最大可能的距離等於畫面寬度
                distance_normalized = min(distance / max_distance, 1)  # 將距離規範化到 [0, 1]
                distance_score = 1 - distance_normalized  # 距離越小分數越高，範圍 [0, 1]

                # 規範化長寬比差異分數
                max_aspect_ratio_diff = 1  # 假設長寬比的最大差異值
                aspect_ratio_normalized = min(aspect_ratio_diff / max_aspect_ratio_diff, 1)  # 將差異規範化到 [0, 1]
                aspect_ratio_score = 1 - aspect_ratio_normalized  # 差異越小分數越高，範圍 [0, 1]

                # 綜合得分公式：根據各個因素的分數和權重進行加權平均
                score = (iou_score * iou_weight +
                         direction_score * direction_weight +
                         distance_score * distance_weight +
                         aspect_ratio_score * aspect_ratio_weight)

                # 儲存所有配對的分數
                all_scores.append((score, ball, center, bbox_size, iou))

        # 按照得分排序，得分越高越好
        all_scores.sort(reverse=True, key=lambda x: x[0])

        used_balls = set()  # 紀錄已經使用的球以避免重複更新
        used_detections = set()  # 紀錄已經配對的檢測框

        # 逐步選擇得分最高的配對
        for score, ball, center, bbox_size, iou in all_scores:
            if ball not in used_balls and (center, bbox_size) not in used_detections:
                # 更新最佳球的狀態
                ball.update_position(center, bbox_size, frame_number, iou)
                used_balls.add(ball)  # 標記該球為已使用
                used_detections.add((center, bbox_size))  # 標記該檢測框為已配對

        # 對於未使用的檢測框，新增一顆球
        for bbox in detected_bboxes:
            center, bbox_size, ball_bbox = self.yolo2ball(bbox)
            if (center, bbox_size) not in used_detections:
                self.add_ball(center, bbox_size, frame_number)

    def calculate_center(self, bbox):
        x_center = (bbox[1])  # 使用 YOLO 的 x_center
        y_center = (bbox[2])  # 使用 YOLO 的 y_center
        return (x_center, y_center)

    def calculate_iou(self, bbox1, bbox2):
        x1_max = max(bbox1[0], bbox2[0])
        y1_max = max(bbox1[1], bbox2[1])
        x2_min = min(bbox1[2], bbox2[2])
        y2_min = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0


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

    def Show_Bounce(self):
        cv2.imshow(self.bounce_title, self.img_opt)

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
        if self.is_show_bounce_analysis:
            self.Draw_Bounce_Analysis()
            self.Show_Bounce_Analysis()
        if self.is_show_bounce_location:
            self.Show_Bounce_Location()
        p_inv = self.Perspective_Transform(self.inv, loc_PT)
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

    def Mark_Perspective_Distortion_Point(self, image, frame_width, frame_height):
        # 點選透視變形位置, 順序為:左上,左下,右下,右上
        PT_data = {"img": image.copy(), "point_x": [], "point_y": []}
        # TODO: 測試用
        PT_data["point_x"] = [498, 162, 1907, 1565]
        PT_data["point_y"] = [688, 828, 854, 692]
        # TODO 測試用
        # cv2.namedWindow("PIC2 (press Q to quit)", 0)
        # cv2.resizeWindow("PIC2 (press Q to quit)", frame_width, frame_height)
        # cv2.setMouseCallback("PIC2 (press Q to quit)", self.Draw_Circle, PT_data)
        # while True:
        #     cv2.imshow("PIC2 (press Q to quit)", PT_data["img"])
        #     if cv2.waitKey(2) == ord("q"):
        #         print(PT_data)
        #         cv2.destroyWindow("PIC2 (press Q to quit)")
        #         break

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

    def Read_Yolo_Label_One_Frame(self, label_file):
        balls = []
        # 取得Yolo預測球的位置
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    l = line.split()
                    if len(l) > 0:
                        if int(l[0]) == 0:
                            balls.append(l)

        if balls:
            self.ball_tracker.update_balls(balls, self.count)
        else:
            self.ball_tracker.no_detect_update_balls(self.count)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Predict")
        parser.add_argument("--input", required=True, type=str, help="Input video")
        args = parser.parse_args()
        return args

    def Detect_Trajectory(self, image):
        # 針對每一貞做運算
        image_CV = image.copy()

        for ball in self.ball_tracker.balls:
            if ball.has_bounced:
                continue
            q_array = np.array(ball.trajectory_center_history)
            non_negatives_idx = np.where(np.all(q_array != (-1, -1), axis=1))[0]
            q_array = q_array[non_negatives_idx]
            if q_array.size == 0:
                x_tmp = np.array([])
                y_tmp = np.array([])
            else:
                x_tmp = q_array[:, 0]
                y_tmp = q_array[:, 1]
            # x_tmp = [self.q[j][0] for j in range(balls) if self.q[j] is not None]
            # y_tmp = [self.q[j][1] for j in range(balls) if self.q[j] is not None]
            ## 落點預測 ######################################################################################################
            if len(x_tmp) >= 3:
                # 檢查是否嚴格遞增或嚴格遞減,(軌跡方向是否相同)
                direction = self.Monotonic(x_tmp, strictly=False, half=True)
                # 累積有三顆球的軌跡向右, 可計算拋物線
                if direction == "right":
                    x_c_pred, y_c_pred = ball.new_center
                    parabola = self.Solve_Parabola(x_tmp, y_tmp)
                    a, b, c = parabola[0]
                    fit = a * x_c_pred**2 + b * x_c_pred + c
                    # 差距 10 個 pixel 以上視為脫離預測的拋物線
                    if abs(y_c_pred - fit) >= 10:
                        x_last = x_tmp[0]
                        # 預測球在球桌上的落點, x_drop : 本次與前次的中點, y_drop : x_drop 於拋物線上的位置
                        x_drop = int(round((x_c_pred + x_last) / 2, 0))
                        y_drop = int(round(a * x_drop**2 + b * x_drop + c, 0))
                        # 繪製本次球體位置, Golden
                        cv2.circle(image_CV, (x_c_pred, y_c_pred), 5, (0, 215, 255), 4)
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
                            # 落點在左側
                            if self.PT_dict[self.count][0] <= int(self.miniboard_width / 2) + self.miniboard_edge:
                                ball.has_bounced = True
                                self.Draw_and_Collect_Data(
                                    (0, 0, 255),
                                    loc_PT,
                                )

                            # 落點在右側
                            elif self.PT_dict[self.count][0] >= int(self.miniboard_width / 2) + self.miniboard_edge:
                                ball.has_bounced = True
                                self.Draw_and_Collect_Data(
                                    (80, 127, 255),
                                    loc_PT,
                                )

        return image_CV

    def Add_Ball_In_Queue(self):
        self.q.appendleft(
            (self.x_c_pred, self.y_c_pred) if self.x_c_pred != np.inf and self.y_c_pred != np.inf else (-1, -1)
        )
        self.q.pop()

        self.q_bv.appendleft((-1, -1))
        self.q_bv.pop()

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
            if self.is_show_bounce_window:
                self.Show_Bounce()
            else:
                image_CV[
                    : self.miniboard_height + self.miniboard_edge * 2,
                    self.frame_width - (self.miniboard_width + self.miniboard_edge * 2) :,
                ] = self.img_opt

        return image_CV

    def Write_Bounce_Location(self):
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
        root_path = f"./runs/detect/pitching_machine_20241015"
        video_fullname = "C0086.MP4"
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
        self.left_speed_list, self.right_speed_list = [], []
        self.bounce_location_list = np.zeros((4, 3), dtype=int)
        self.bouncing_offset_x, self.bouncing_offset_y = 10, 15  # bouncing location offset
        self.speed_left, self.speed_right = 0, 0  # 左右選手球速
        self.bounce_frame_L, self.bounce_frame_R = -1, -1  # 出現落點的Frame
        self.left_shot_count, self.right_shot_count = 0, 0  # 左右選手擊球時的frame number
        self.hit_count = 0  # 擊球次數
        self.count = 1  # 記錄處理幾個 Frame
        self.MAX_velo = 0  # 最大球速
        self.x_c_pred, self.y_c_pred = np.inf, np.inf  # 球體中心位置
        self.is_first_ball = True  # 每局第一球的時候frame只要5個，其他時間要9個
        self.is_serve_wait = False
        self.shotspeed = 0
        self.shotspeed_previous = 0

        # 顯示參數
        self.is_show_bounce = True
        self.is_show_bounce_window = False
        self.is_show_bounce_analysis = False
        self.is_show_bounce_location = False
        self.is_show_speed_analysis = False
        if real_time:
            self.is_show_bounce_window = True
            self.bounce_title = "Bounce"
            cv2.namedWindow(self.bounce_title, cv2.WINDOW_NORMAL)
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

        self.ball_tracker = BallTracker()

    def Set_Frame_Info(self, frame_height, frame_width, framerate):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.framerate = framerate
        self.ball_tracker.set_frame_info(frame_width, frame_height)

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
            if self.count == 568:
                print("test")
            self.Read_Yolo_Label_One_Frame(label_file=label_file)
            image_CV = self.Detect_Trajectory(image)
            self.Add_Ball_In_Queue()
            image_CV = self.Draw_On_Image(image_CV)

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
