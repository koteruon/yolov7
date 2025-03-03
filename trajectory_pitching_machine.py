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
from matplotlib.colors import to_rgb
from matplotlib.path import Path as matplotlib_path
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import leastsq
from tqdm import tqdm


class Ball:
    def __init__(self, center, bbox_size, frame_number, color_bgr_255, frame_width):
        self.ball_limit = 5  # 限制最多儲存多少歷史紀錄
        self.trajectory_ball_limit = 9  # 限制最多儲存多少歷史紀錄用於推算軌跡落點
        self.show_ball_limit = 12  # 限制最多儲存多少歷史紀錄用於畫畫用
        self.bounce_ball_limit = 6  # 限制最多儲存多少歷史紀錄用於畫畫用
        self.frame_limit = 20  # 限制相隔多少frame以後自動刪除
        self.center_history = [np.array(center)]  # 初始化中心點歷史紀錄
        self.trajectory_center_history = [np.array(center)]  # 初始化中心點歷史紀錄用於推算軌跡落點
        self.show_center_history = [np.array(center)]  # 初始化中心點歷史紀錄用於畫畫用
        self.bounce_center_history = [np.array((-1, -1))]  # 初始化落下的位置用於畫畫用
        self.direction_history = [np.array([0, 0])]  # 初始化運動方向歷史紀錄
        self.bbox_size_history = [bbox_size]  # 初始化BBOX大小歷史紀錄
        self.iou_history = [0]  # 初始化IOU歷史紀錄
        self.score_history = [1.0]  # 初始化分數歷史紀錄
        self.has_bounced = False  # 已經有落點了
        self.bounced_frame_number = -1  # 落點的frame number
        self.bounced_side = None  # 落點在哪一邊
        self.is_tracking = True  # 使否還在追蹤
        self.has_switch_avg_direction = False  # 是否打擊後反彈
        self.new_center = np.array(center)
        self.new_bbox_size = np.array(bbox_size)
        self.new_frame_number = frame_number
        self.color_bgr_255 = color_bgr_255
        self.count_ball = 1  # 紀錄總共多少個紀錄
        self.frame_width = frame_width

        # 初始化平均值
        self.average_center = np.array(center)
        self.average_direction = np.array([0, 0])
        self.average_bbox_size = bbox_size
        self.average_iou = 0
        self.new_bbox = self.calculate_new_bbox(self.new_center, self.new_bbox_size)

    def get_color_bgr_255(self, draw_ball_at_frame_number):
        if isinstance(self.color_bgr_255, list):
            if self.has_bounced and draw_ball_at_frame_number > self.bounced_frame_number:
                return self.color_bgr_255[1]
            else:
                return self.color_bgr_255[0]
        else:
            return self.color_bgr_255

    def no_detect_update_position(self):
        center = (-1, -1)
        self._update_history(self.trajectory_ball_limit, self.trajectory_center_history, center)
        self._update_history(self.show_ball_limit, self.show_center_history, center)
        self._update_history(self.bounce_ball_limit, self.bounce_center_history, center)

    def update_position(self, new_center, new_bbox_size, new_frame_number, iou, score):
        self.new_center = np.array(new_center)
        self.new_bbox_size = np.array(new_bbox_size)
        self.new_frame_number = new_frame_number
        new_direction = self.new_center - self.average_center  # 使用平均位置來計算運動方向

        # 更新歷史紀錄
        self._update_history(self.ball_limit, self.center_history, self.new_center)
        self._update_history(self.trajectory_ball_limit, self.trajectory_center_history, self.new_center)
        self._update_history(self.show_ball_limit, self.show_center_history, self.new_center)
        self._update_history(self.bounce_ball_limit, self.bounce_center_history, (-1, -1))
        self._update_history(self.ball_limit, self.direction_history, new_direction)
        self._update_history(self.ball_limit, self.bbox_size_history, self.new_bbox_size)
        self._update_history(self.ball_limit, self.iou_history, iou)  # 更新 IOU 歷史紀錄
        self._update_history(sys.maxsize, self.score_history, score)  # 更新 IOU 歷史紀錄

        # 計算並儲存最新的平均值
        self.average_center = self.get_average_position()
        self.average_direction = self.get_average_direction()
        self.average_bbox_size = self.get_average_bbox_size()
        self.average_iou = self.get_average_iou()
        self.new_bbox = self.calculate_new_bbox(self.new_center, self.new_bbox_size)

        # 確認是否擊到球
        self.check_is_switch_avg_direction()

        # 增加記錄數量
        self.count_ball += 1

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

    def bounce(self, bounce_center, frame_number, side):
        self._update_history(self.bounce_ball_limit, self.bounce_center_history, bounce_center)
        self.has_bounced = True
        self.bounced_frame_number = frame_number
        self.bounced_side = side

    # Sigmoid 函数映射
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-10 * (x - 0.5)))  # 调整参数以控制形状

    def calculate_weights(self):
        # 計算權重
        historical_iou = self.get_average_iou()
        iou_weight = self.sigmoid(historical_iou)  # IOU 的權重
        x_position_ratio = self.average_center[0] / self.frame_width  # 球在畫面中的x位置比率 [0, 1]
        distance_weight = 0.45 + x_position_ratio * 0.25  # 離右邊越近，距離權重越高
        direction_weight = max(0.0, -0.2 + x_position_ratio * 0.9)  # 離右邊越近，方向權重越高
        aspect_ratio_weight = 0.25 + x_position_ratio * 0.25  # 離右邊越近，距離權重越高
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
            center[1] + bbox_size[1] / 2,
        ]
        return ball_bbox

    def check_is_switch_avg_direction(self):
        if self.new_center[0] < (self.frame_width / 2):  # 畫面左邊
            if self.get_average_direction()[0] > 0:  # 平均方向向右
                self.has_switch_avg_direction = True


class BallTracker:
    def __init__(self, show_debug_output=False):
        self.balls = []  # 儲存所有球的資訊
        self.balls_history = []  # 儲存歷史所有球的資訊
        self.ball_count = 0
        self.colormap = plt.get_cmap("Paired")  # 选择一个 colormap
        self.score_threshold = 0.5  # 分數筏值 0.5
        self.count_ball_threahold = 15  # 60fps = 15, 120fps = 40 # 有多少個歷史軌跡內才算發球
        self.count_ball_add_last_frame_number = 0  # 最後一個紀錄到發球軌跡的frame number
        self.count_balls = []  # 發球追蹤中
        self.count_ball_reset_threahold = 100  # 幾個frame之後都沒有增加球就reset
        self.count_ball_rounds = 1
        self.count_ball_score = {}  # key: round, value: (score)
        self.show_debug_output = show_debug_output

    def set_add_ball_polygon_path(self, add_ball_polygon_path):
        self.add_ball_polygon_path = add_ball_polygon_path

    def set_count_ball_polygon_path(self, count_ball_polygon_path):
        self.count_ball_polygon_path = count_ball_polygon_path

    def count_ball_reset(self, frame_number):
        if frame_number - self.count_ball_add_last_frame_number >= self.count_ball_reset_threahold:
            if len(self.count_balls) > 0:
                self.count_ball_rounds += 1
            self.count_balls = []
            return True
        return False

    def count_ball_calculate_score(self, frame_number):
        if frame_number - self.count_ball_add_last_frame_number >= self.count_ball_reset_threahold:
            if self.count_ball_rounds not in self.count_ball_score:
                if len(self.count_balls) > 0:
                    if self.count_ball_size() == self.count_ball_valid_hits():
                        self.count_ball_score[self.count_ball_rounds] = (1, 0)
                    else:
                        self.count_ball_score[self.count_ball_rounds] = (0, 1)

    def count_ball_get_score(self):
        if self.count_ball_score:
            return tuple(map(sum, zip(*self.count_ball_score.values())))
        else:
            return (0, 0)

    def count_ball_size(self):
        return len(self.count_balls)

    def count_ball_valid_hits(self):
        valid_hits = 0
        for ball in self.count_balls:
            if ball.bounced_side == "right":
                valid_hits += 1
        return valid_hits

    def count_ball_side_errors(self):
        side_errors = 0
        for ball in self.count_balls:
            if ball.bounced_side == "left":
                side_errors += 1
        return side_errors

    def count_ball_out_hits(self):
        out_hits = 0
        for ball in self.count_balls:
            if not ball.has_bounced and not ball.is_tracking:
                if ball.has_switch_avg_direction:
                    out_hits += 1
        return out_hits

    def count_ball_misses(self):
        misses = 0
        for ball in self.count_balls:
            if not ball.has_bounced and not ball.is_tracking:
                if not ball.has_switch_avg_direction:
                    misses += 1
        return misses

    def get_dynamic_color_bgr_255(self, index):
        if self.show_debug_output:
            num_colors = self.colormap.N
            color = self.colormap(index % num_colors)
            color_rgb = to_rgb(color)
            color_bgr = np.array(color_rgb)[::-1]
            color_bgr_255 = (color_bgr * 255).astype(np.uint8)
        else:
            color_bgr_255 = [(0, 255, 255), (200, 200, 200)]
        return color_bgr_255

    def set_frame_info(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def add_ball(self, center, bbox_size, frame_number, frame_width):
        self.ball_count += 1
        color_bgr_255 = self.get_dynamic_color_bgr_255(self.ball_count)
        new_ball = Ball(center, bbox_size, frame_number, color_bgr_255, frame_width)
        self.balls.append(new_ball)

    def remove_non_tracking_ball(self, frame_number):
        for ball in self.balls:
            if frame_number - ball.new_frame_number > ball.frame_limit:
                self.balls_history.append(ball)
                self.balls.remove(ball)
                ball.is_tracking = False

    def yolo2ball(self, bbox):
        # 提取 YOLO 格式的數據
        _, x_center, y_center, width, height, _ = bbox
        ball_x_center, ball_y_center = int(float(x_center) * self.frame_width), int(float(y_center) * self.frame_height)
        ball_width, ball_height = int(float(width) * self.frame_width), int(float(height) * self.frame_height)
        center = (ball_x_center, ball_y_center)
        bbox_size = (ball_width, ball_height)

        ball_bbox = [
            center[0] - bbox_size[0] / 2,
            center[1] - bbox_size[1] / 2,
            center[0] + bbox_size[0] / 2,
            center[1] + bbox_size[1] / 2,
        ]

        return center, bbox_size, ball_bbox

    def no_detect_update_balls(self, frame_number):
        self.remove_non_tracking_ball(frame_number)
        self.count_ball_calculate_score(frame_number)
        for ball in self.balls:
            ball.no_detect_update_position()

    def update_balls(self, detected_bboxes, frame_number):
        self.remove_non_tracking_ball(frame_number)
        self.count_ball_calculate_score(frame_number)

        all_scores = []  # 儲存所有檢測框和所有球的配對及其分數

        for bbox in detected_bboxes:
            center, bbox_size, ball_bbox = self.yolo2ball(bbox)
            iou = 0

            # 遍歷所有已知的球，計算該檢測框和每個球的配對得分
            for ball in self.balls:
                # 計算 IOU
                iou = self.calculate_logistic_iou(ball.new_bbox, ball_bbox)

                # 計算方向性
                direction_similarity = (
                    np.dot(ball.average_direction, (center - ball.average_center))
                    / (np.linalg.norm(ball.average_direction) * np.linalg.norm(center - ball.average_center))
                    if np.linalg.norm(ball.average_direction) != 0 and np.linalg.norm(center - ball.average_center) != 0
                    else 0
                )
                direction_similarity = max(-1, min(1, direction_similarity))  # 保證方向相似度在 [0, 1] 範圍內

                # 計算長寬比差異
                aspect_ratio_ball = ball.average_bbox_size[0] / ball.average_bbox_size[1]
                aspect_ratio_new = bbox_size[0] / bbox_size[1]
                aspect_ratio_diff = abs(aspect_ratio_ball - aspect_ratio_new)

                # 設定垂直距離和水平距離的權重
                vertical_weight = 0.7  # 更在意垂直距離
                horizontal_weight = 0.3  # 水平距離次要
                # 計算距離
                weighted_diff = np.array(
                    [
                        horizontal_weight * (ball.average_center[0] - center[0]),  # 水平方向加權
                        vertical_weight * (ball.average_center[1] - center[1]),  # 垂直方向加權
                    ]
                )
                distance = np.linalg.norm(weighted_diff)
                weighted_diff = np.array(
                    [
                        horizontal_weight * self.frame_width,  # 水平方向加權
                        vertical_weight * self.frame_height,  # 垂直方向加權
                    ]
                )
                max_distance = np.linalg.norm(weighted_diff) / 4

                # 獲取權重
                iou_weight, direction_weight, distance_weight, aspect_ratio_weight = ball.calculate_weights()

                # 計算各個因素的分數
                iou_score = iou  # IOU 越大分數越高
                direction_score = direction_similarity  # 方向性越相似分數越高
                # 規範化距離分數

                distance_normalized = min(distance / max_distance, 1)  # 將距離規範化到 [0, 1]
                distance_score = 1 - distance_normalized  # 距離越小分數越高，範圍 [0, 1]

                # 規範化長寬比差異分數
                max_aspect_ratio_diff = 1  # 假設長寬比的最大差異值
                aspect_ratio_normalized = min(aspect_ratio_diff / max_aspect_ratio_diff, 1)  # 將差異規範化到 [0, 1]
                aspect_ratio_score = 1 - aspect_ratio_normalized  # 差異越小分數越高，範圍 [0, 1]

                # 綜合得分公式：根據各個因素的分數和權重進行加權平均
                score = (
                    iou_score * iou_weight
                    + direction_score * direction_weight
                    + distance_score * distance_weight
                    + aspect_ratio_score * aspect_ratio_weight
                )

                if score > self.score_threshold:
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
                ball.update_position(center, bbox_size, frame_number, iou, score)
                used_balls.add(ball)  # 標記該球為已使用
                used_detections.add((center, bbox_size))  # 標記該檢測框為已配對

        # 對於未使用的檢測框，新增一顆球
        for bbox in detected_bboxes:
            center, bbox_size, ball_bbox = self.yolo2ball(bbox)
            if (center, bbox_size) not in used_detections:
                if self.add_ball_polygon_path.contains_point(center):
                    self.add_ball(center, bbox_size, frame_number, self.frame_width)

        # 更新是否為發球機剛發出的球
        has_reset = False
        for ball in self.balls:
            if ball not in self.count_balls:  # 沒有被記錄過
                if ball.count_ball < self.count_ball_threahold:  # 剛新增的球
                    if ball.get_average_direction()[0] < 0:  # 向左飛行
                        if self.count_ball_polygon_path.contains_point(ball.new_center):  # 在可以被記錄的區間內
                            has_reset = self.count_ball_reset(frame_number)
                            self.count_ball_add_last_frame_number = frame_number
                            self.count_balls.append(ball)  # 紀錄該球

        return has_reset

    def calculate_center(self, bbox):
        x_center = bbox[1]  # 使用 YOLO 的 x_center
        y_center = bbox[2]  # 使用 YOLO 的 y_center
        return (x_center, y_center)

    def calculate_logistic_iou(self, bbox1, bbox2, alpha=10, beta=0.3):
        x1_max = max(bbox1[0], bbox2[0])
        y1_max = max(bbox1[1], bbox2[1])
        x2_min = min(bbox1[2], bbox2[2])
        y2_min = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0
        logistic_iou = 1 / (1 + np.exp(-alpha * (iou - beta)))
        return logistic_iou


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
            cv2.circle(param["img"], (x, y), 3, param["color"], -1)
            param["point_x"].append(x)
            param["point_y"].append(y)

    def Draw_and_Collect_Data(
        self,
        color,
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

    def Create_Output_Dir(self, output_path):
        # 建立輸出檔案夾
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        # 建立影片預測結果資料夾
        video_path = Path(output_path.joinpath("video"))
        video_path.mkdir(parents=True, exist_ok=True)
        video_path = video_path.as_posix()

        return video_path

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
        self.output = cv2.VideoWriter(
            video_path,
            fourcc,
            self.framerate,
            size,
        )

    def Write_Frame_to_Video(self, image_CV):
        self.output.write(image_CV)

    def Release_Video(self):
        self.output.release()

    def Load_Mark_Point(self, csv_path, PT_data):
        df = pd.read_csv(csv_path)
        df = df[df["video_id"] == self.video_name]
        if not df.empty:
            row = df.iloc[0]  # 取第一行數據
            PT_data["point_x"] = [row[f"point_x{i}"] for i in range(1, 5)]
            PT_data["point_y"] = [row[f"point_y{i}"] for i in range(1, 5)]
            return True
        else:
            return False

    def Save_Mark_Point(self, csv_path, PT_data):
        df = pd.read_csv(csv_path)
        new_row = {
            "video_id": self.video_name,
            "point_x1": PT_data["point_x"][0],
            "point_x2": PT_data["point_x"][1],
            "point_x3": PT_data["point_x"][2],
            "point_x4": PT_data["point_x"][3],
            "point_y1": PT_data["point_y"][0],
            "point_y2": PT_data["point_y"][1],
            "point_y3": PT_data["point_y"][2],
            "point_y4": PT_data["point_y"][3],
        }
        if self.video_name in df["video_id"].values:
            df.loc[df["video_id"] == self.video_name, new_row.keys()] = new_row.values()
        else:
            new_data = pd.DataFrame([new_row])
            df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(csv_path, index=False)

    def Mark_Perspective_Distortion_Point(self, image, frame_width, frame_height):
        # 點選透視變形位置, 順序為:左上,左下,右下,右上
        PT_data = {"img": image.copy(), "point_x": [], "point_y": [], "color": (0, 255, 255)}
        table_csv_path = r"pitching_machine/table.csv"
        if not self.Load_Mark_Point(table_csv_path, PT_data):
            cv2.namedWindow("PIC2 (press Q to quit)", 0)
            cv2.resizeWindow("PIC2 (press Q to quit)", frame_width, frame_height)
            cv2.setMouseCallback("PIC2 (press Q to quit)", self.Draw_Circle, PT_data)
            while True:
                cv2.imshow("PIC2 (press Q to quit)", PT_data["img"])
                if cv2.waitKey(2) == ord("q"):
                    print(PT_data)
                    cv2.destroyWindow("PIC2 (press Q to quit)")
                    break
            self.Save_Mark_Point(table_csv_path, PT_data)

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

        # 框選發球機可增加球的位置，順序為:左上,左下,右下,右上
        PT_data = {"img": image.copy(), "point_x": [], "point_y": [], "color": (0, 255, 128)}
        pitching_csv_path = r"pitching_machine/pitching.csv"
        if not self.Load_Mark_Point(pitching_csv_path, PT_data):
            cv2.namedWindow("pitching maching (press Q to quit)", 0)
            cv2.resizeWindow("pitching maching (press Q to quit)", frame_width, frame_height)
            cv2.setMouseCallback("pitching maching (press Q to quit)", self.Draw_Circle, PT_data)
            while True:
                cv2.imshow("pitching maching (press Q to quit)", PT_data["img"])
                if cv2.waitKey(2) == ord("q"):
                    print(PT_data)
                    cv2.destroyWindow("pitching maching (press Q to quit)")
                    break
            self.Save_Mark_Point(pitching_csv_path, PT_data)

        upper_left = [PT_data["point_x"][0], PT_data["point_y"][0]]
        lower_left = [PT_data["point_x"][1], PT_data["point_y"][1]]
        lower_right = [PT_data["point_x"][2], PT_data["point_y"][2]]
        upper_right = [PT_data["point_x"][3], PT_data["point_y"][3]]
        add_ball_point = np.float32([upper_left, lower_left, lower_right, upper_right])
        self.ball_tracker.set_add_ball_polygon_path(matplotlib_path(add_ball_point))

        # 框選發球機計算球的位置，順序為:左上,左下,右下,右上
        PT_data = {"img": image.copy(), "point_x": [], "point_y": [], "color": (255, 0, 0)}
        checkpoint_csv_path = r"pitching_machine/checkpoint.csv"
        if not self.Load_Mark_Point(checkpoint_csv_path, PT_data):
            cv2.namedWindow("count ball (press Q to quit)", 0)
            cv2.resizeWindow("count ball (press Q to quit)", frame_width, frame_height)
            cv2.setMouseCallback("count ball (press Q to quit)", self.Draw_Circle, PT_data)
            while True:
                cv2.imshow("count ball (press Q to quit)", PT_data["img"])
                if cv2.waitKey(2) == ord("q"):
                    print(PT_data)
                    cv2.destroyWindow("count ball (press Q to quit)")
                    break
            self.Save_Mark_Point(checkpoint_csv_path, PT_data)

        upper_left = [PT_data["point_x"][0], PT_data["point_y"][0]]
        lower_left = [PT_data["point_x"][1], PT_data["point_y"][1]]
        lower_right = [PT_data["point_x"][2], PT_data["point_y"][2]]
        upper_right = [PT_data["point_x"][3], PT_data["point_y"][3]]
        count_ball_point = np.float32([upper_left, lower_left, lower_right, upper_right])
        self.ball_tracker.set_count_ball_polygon_path(matplotlib_path(count_ball_point))

    def Read_Yolo_Label_One_Frame(self, label_file=None, balls=None):
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

        if balls:
            has_reset = self.ball_tracker.update_balls(balls, self.count)
            if has_reset:
                self.img_opt = self.Draw_MiniBoard()
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
            ## 落點預測 ######################################################################################################
            if len(x_tmp) >= 3:
                # 檢查是否嚴格遞增或嚴格遞減,(軌跡方向是否相同)
                direction = self.Monotonic(x_tmp, strictly=False, half=False)
                # 累積有三顆球的軌跡向右, 可計算拋物線
                if direction == "right":
                    bounced = False
                    x_c_pred, y_c_pred = ball.new_center
                    parabola = self.Solve_Parabola(x_tmp, y_tmp)
                    a, b, c = parabola[0]
                    fit = a * x_c_pred**2 + b * x_c_pred + c
                    # 差距 10 個 pixel 以上視為脫離預測的拋物線
                    bounced = abs(y_c_pred - fit) >= 10
                    if not self.use_parabola:
                        vy = np.diff(y_tmp)  # 計算速度（差分）計算最近 8 幀的垂直速度
                        window_size = 1  # 平滑窗口大小，可根據需要調整
                        smoothed_vy = np.convolve(
                            vy, np.ones(window_size) / window_size, mode="valid"
                        )  # 平滑速度數據（移動平均）
                        ay = np.diff(vy)
                        if smoothed_vy[0] >= 0 and smoothed_vy[-1] < 0 and ay[-1] < 0.5:  # 閾值可調
                            bounced = True
                        else:
                            bounced = False
                    if bounced:
                        x_last = x_tmp[-2]
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
                                ball.bounce((x_drop, y_drop), self.count, "left")
                                self.Draw_and_Collect_Data(
                                    (80, 127, 255),
                                )

                            # 落點在右側
                            elif self.PT_dict[self.count][0] >= int(self.miniboard_width / 2) + self.miniboard_edge:
                                ball.bounce((x_drop, y_drop), self.count, "right")
                                self.Draw_and_Collect_Data(
                                    (0, 255, 0),
                                )

        return image_CV

    def Draw_On_Image(self, image_CV):
        for ball in self.ball_tracker.balls:
            # draw current frame prediction and previous 11 frames as yellow circle, total: 12 frames
            for privous_idx, ball_center in enumerate(reversed(ball.show_center_history)):
                if not np.array_equal(ball_center, np.array([-1, -1])):
                    ball_color = ball.get_color_bgr_255(self.count - privous_idx)
                    cv2.circle(image_CV, tuple(ball_center), 5, tuple(map(int, ball_color)), 1)

            # draw bounce point as red circle
            for bounce_center in ball.bounce_center_history:
                if not np.array_equal(bounce_center, np.array([-1, -1])):
                    cv2.circle(image_CV, tuple(bounce_center), 5, (0, 0, 255), 4)

        # Place miniboard on upper right corner
        if self.is_show_bounce:
            image_CV[
                : self.miniboard_height + self.miniboard_edge * 2,
                self.frame_width - (self.miniboard_width + self.miniboard_edge * 2) :,
            ] = self.img_opt

        if self.show_debug_output:
            for idx, ball in enumerate(self.ball_tracker.balls, 1):
                cv2.putText(
                    image_CV,
                    f"Score : {ball.score_history[-1]:.2f}",
                    (10, 40 + 40 * idx),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1,
                    tuple(map(int, ball.color_bgr_255)),
                    1,
                    cv2.LINE_AA,
                )
        else:
            self.freetype.putText(
                image_CV,
                f"幀數: {self.count}",
                (10, 20),
                36,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
                False,
            )
            self.freetype.putText(
                image_CV,
                f"回合數: {self.ball_tracker.count_ball_rounds}",
                (10, 70),
                36,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
                False,
            )
            self.freetype.putText(
                image_CV,
                f"發球數: {self.ball_tracker.count_ball_size()}",
                (10, 120),
                36,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
                False,
            )
            self.freetype.putText(
                image_CV,
                f"有效擊球: {self.ball_tracker.count_ball_valid_hits()}",
                (10, 170),
                36,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
                False,
            )

            if self.analysis_output:
                self.freetype.putText(
                    image_CV,
                    f"錯誤落點: {self.ball_tracker.count_ball_side_errors()}",
                    (10, 220),
                    36,
                    (80, 127, 255),
                    -1,
                    cv2.LINE_AA,
                    False,
                )
                self.freetype.putText(
                    image_CV,
                    f"擊球出界: {self.ball_tracker.count_ball_out_hits()}",
                    (10, 270),
                    36,
                    (0, 0, 255),
                    -1,
                    cv2.LINE_AA,
                    False,
                )
                self.freetype.putText(
                    image_CV,
                    f"未擊中: {self.ball_tracker.count_ball_misses()}",
                    (10, 320),
                    36,
                    (0, 0, 255),
                    -1,
                    cv2.LINE_AA,
                    False,
                )

        # 比分
        score = self.ball_tracker.count_ball_get_score()
        self.freetype.putText(image_CV, f"{score[0]}:{score[1]}", (850, 25), 250, (0, 255, 255), -1, cv2.LINE_AA, False)

        return image_CV

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

    def Create_Video_Output_Path(self, realtime=False):
        # 影片跟目錄3
        max_folder = ""
        if not realtime:
            max_folder = "C0002_20250303_01"  # realtime3
        max_num = -1
        root_path = f"./runs/detect"
        pattern = re.compile(r"^realtime(\d+)$")
        if max_folder == "":
            for folder_name in os.listdir(root_path):
                match = pattern.match(folder_name)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
                        max_folder = folder_name
        root_path = os.path.join(root_path, max_folder)
        sub_max_num = -1
        sub_max_folder = ""
        if not realtime:
            sub_max_folder = None
        sub_pattern = re.compile(r"^Realtime(\d+)$")
        if sub_max_folder != None and sub_max_folder == "":
            for folder_name in os.listdir(root_path):
                match = sub_pattern.match(folder_name)
                if match:
                    num = int(match.group(1))
                    if num > sub_max_num:
                        sub_max_num = num
                        sub_max_folder = folder_name
        if sub_max_folder != None:
            root_path = os.path.join(root_path, sub_max_folder)
        print(f"root_path: {root_path}")

        if not realtime:
            video_fullname = "C0002.MP4"
        else:
            video_fullname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        self.video_name = os.path.splitext(video_fullname)[0]
        self.video_suffix = os.path.splitext(video_fullname)[1]
        self.input_path = os.path.join(root_path, video_fullname)

        if sub_max_folder != None:
            self.video_path = os.path.join(self.video_path, max_folder, sub_max_folder)
        Path(self.video_path).mkdir(parents=True, exist_ok=True)

        # yolo labels path
        self.label_path = os.path.join(root_path, "labels")

    def __init__(self):
        # temp
        self.HEIGHT = 288  # model input size
        self.WIDTH = 512

        # 建立目錄
        output_path = f"./inference/output"
        self.video_path = self.Create_Output_Dir(output_path)

        self.Create_Video_Output_Path()

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

        # 參數
        self.bouncing_offset_x, self.bouncing_offset_y = 10, 15  # bouncing location offset
        self.count = 1  # 記錄處理幾個 Frame (從step開始)

        # 顯示參數
        self.is_show_bounce = True

        self.analysis_output = True
        self.show_debug_output = False
        self.use_parabola = True
        self.freetype = cv2.freetype.createFreeType2()
        self.freetype.loadFontData(fontFileName="ttf/MSJH.TTC", id=0)

        self.ball_tracker = BallTracker(self.show_debug_output)

    def Set_Frame_Info(self, frame_height, frame_width, framerate):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.framerate = framerate
        self.ball_tracker.set_frame_info(frame_width, frame_height)

    def Next_Count(self):
        self.count += 1

    ### 後處理從此開始 ###
    def main(self):
        start = time.time()

        # 讀影片
        success, image, cap, framerate, frame_height, frame_width, total_frames = self.Read_Video(self.input_path)
        self.Set_Frame_Info(frame_height, frame_width, framerate)

        # 等比例縮放
        ratio = self.frame_height / self.HEIGHT
        size = (int(self.WIDTH * ratio), int(self.HEIGHT * ratio))

        # 寫 預測結果
        video_path = f"{self.video_path}/{self.video_name}_predict_12.mp4"
        self.Write_Video(video_path, size)

        # 透視變形
        self.Mark_Perspective_Distortion_Point(image, self.frame_width, self.frame_height)

        # 針對每一貞做運算
        batch = 12
        n = 4
        k = batch // 2
        with tqdm(total=total_frames, desc="Processing Frames") as pbar:
            while success:
                label_file = os.path.join(self.label_path, f"{self.video_name}_{self.count}.txt")
                self.Read_Yolo_Label_One_Frame(label_file=label_file)
                image_CV = self.Detect_Trajectory(image)
                image_CV = self.Draw_On_Image(image_CV)
                self.Next_Count()
                if self.count >= total_frames - 12:
                    break
                self.Write_Frame_to_Video(image_CV)
                success, image = cap.read()
                pbar.update(1)

        # For releasing cap and out.
        cap.release()
        self.Release_Video()

        end = time.time()
        total_time = end - start

        print(f"Detect Result is saved in {self.video_path}")
        print(f"Total time: {total_time} seconds")
        print(f"Done......")


if __name__ == "__main__":
    trajectory = Trajectory()
    trajectory.main()
