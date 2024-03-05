import math
import multiprocessing as mp
import os
import queue

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
from tqdm import tqdm

HEIGHT = 1080
WIDTH = 1920

object_th = 0.5
person_th = 0.8

v_root = f"./runs/detect/yolov7_20231122/"
lbl_root = f"./runs/detect/yolov7_20231122/labels"
result_root = f"./runs/detect/yolov7_20231122/"


def Monotonic(L):
    # 檢查單調函數(嚴格遞增或遞減)
    strictly_increasing = all(x > y for x, y in zip(L, L[1:]))
    strictly_decreasing = all(x < y for x, y in zip(L, L[1:]))
    # non_increasing = all(x>=y for x, y in zip(L, L[1:]))
    # non_decreasing = all(x<=y for x, y in zip(L, L[1:]))
    return strictly_increasing or strictly_decreasing


def Parabola_Function(params, x):
    # 拋物線函數 Parabola Function
    a, b, c = params
    return a * x**2 + b * x + c


def Parabola_Error(params, x, y):
    # 拋物線偏移誤差
    return Parabola_Function(params, x) - y


def Solve_Parabola(X, Y):
    # 解拋物線方程式
    p_arg = [10, 10, 10]
    parabola = leastsq(Parabola_Error, p_arg, args=(X, Y))
    return parabola


def Perspective_Transform(matrix, coord):
    # 透視變形轉換
    x = (matrix[0][0] * coord[0] + matrix[0][1] * coord[1] + matrix[0][2]) / (
        (matrix[2][0] * coord[0] + matrix[2][1] * coord[1] + matrix[2][2])
    )
    y = (matrix[1][0] * coord[0] + matrix[1][1] * coord[1] + matrix[1][2]) / (
        (matrix[2][0] * coord[0] + matrix[2][1] * coord[1] + matrix[2][2])
    )
    PT = (int(x), int(y))
    return PT


def Draw_Circle(event, x, y, flags, param):
    # 用於透視變形取點
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(param["img"], (x, y), 3, (0, 255, 255), -1)
        param["point_x"].append(x)
        param["point_y"].append(y)


def perspective_distortion_correlation(image, frame_width, frame_height):
    # 點選透視變形位置, 順序為:左上,左下,右下,右上
    PT_dict = {}
    PT_data = {"img": image.copy(), "point_x": [], "point_y": []}
    # TODO: 測試用
    # PT_data["point_x"] = [574, 402, 1649, 1450]
    # PT_data["point_y"] = [700, 767, 773, 704]
    # TODO 測試用
    cv2.namedWindow("PIC2 (press Q to quit)", 0)
    cv2.resizeWindow("PIC2 (press Q to quit)", frame_width, frame_height)
    cv2.setMouseCallback("PIC2 (press Q to quit)", Draw_Circle, PT_data)
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
            [20, 20],  # 白線為2公分(20mm)
            [20, 1525 + 20],  # 桌寬為152.5公分(1525mm)
            [2740 + 20, 1525 + 20],  # 桌長為274公分(2740mm)
            [2740 + 20, 20],
        ]
    )
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    inv = cv2.getPerspectiveTransform(pts2, pts1)
    return matrix, inv


def subtask(v_name, v_path, lbl_dir):
    flag = False

    # 讀取影片
    cap = cv2.VideoCapture(v_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    new_v = cv2.VideoWriter(
        os.path.join(result_root, f"output.mp4"),
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (WIDTH, HEIGHT),
    )

    #
    lbl_list = sorted(
        os.listdir(lbl_dir),
        key=lambda x: int(x.replace(f"{v_name}_", "").split(".")[0]),
    )

    # track_length = int(cap.get(cv2.CAP_PROP_FPS)) *2
    track_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # ball_track_list = [(-1,-1,-1,-1) for _ in range(track_length)]     ## (x, y, w, h)
    ball_track_list = []

    # 變數
    past_direction = "left"
    stop_idx = 0
    lbl = lbl_list.pop(0)
    isball = []
    x_c_pred, y_c_pred, past_frame_number = 0, 0, 0
    x_speed = 0

    #
    ret, frame = cap.read()
    frame_height, frame_width = int(cap.get(4)), int(cap.get(3))
    if not ret:
        return

    # 初始化帧编号和当前帧
    frame_number = 0
    current_frame = 0

    # 透視變形矯正
    matrix, inv = perspective_distortion_correlation(frame, frame_width, frame_height)

    # 球速計算變數
    # In order to draw the trajectory of tennis, we need to save the coordinate of preious 12 frames
    q = queue.deque([None for _ in range(12)])

    while cap.isOpened():
        ret, frame = cap.read()
        frame_number += 1
        if not ret:
            break

        # ball_track_list.pop(0)
        lbl_cnt = int(lbl.replace(f"{v_name}_", "").split(".")[0])
        if frame_number != lbl_cnt:
            ball_track_list.append((-1, -1, -1, -1))
        else:
            ## 有偵測到球體
            ## 讀取label檔案
            with open(os.path.join(lbl_dir, lbl), "r") as txt_f:
                lines = txt_f.readlines()
                line_split = [x.split() for x in lines]
                paragraph = [
                    [float(word) if idx != 0 else int(word) for idx, word in enumerate(line)] for line in line_split
                ]

                objects_sorted = sorted(paragraph, key=lambda x: (x[0], x[1]))
                ## 取出球
                ball_list = [b for b in objects_sorted if b[5] >= object_th and b[0] == 0]

                if ball_list != []:
                    ball_list = sorted(ball_list, key=lambda x: x[5], reverse=True)
                    ball_track_list.append((ball_list[0][1], ball_list[0][2], ball_list[0][3], ball_list[0][4]))
                else:
                    ball_track_list.append((-1, -1, -1, -1))
            if lbl_list != []:
                lbl = lbl_list.pop(0)

            # 球速計算
            past_x_c_pred, past_y_c_pred = x_c_pred, y_c_pred
            x_c_pred, y_c_pred = float(ball_list[0][1]) * frame_width, float(ball_list[0][2]) * frame_height
            x_c_pred, y_c_pred = int(x_c_pred), int(y_c_pred)
            # loc_x_c_pred, loc_y_c_pred = Perspective_Transform(matrix, (x_c_pred, y_c_pred))
            x_speed = abs(
                (((x_c_pred - past_x_c_pred) * (2740.0 / 1011.0)) / ((frame_number - past_frame_number) * (1.0 / 50.0)))
                * 3600.0
                / 1000000.0
            )
            speed = math.hypot(x_c_pred - past_x_c_pred, y_c_pred - past_y_c_pred) / (
                (frame_number - past_frame_number) * (1.0 / 50.0)
            )
            past_frame_number = frame_number
            print(x_c_pred - past_x_c_pred, (x_c_pred - past_x_c_pred) * (2740.0 / 1011.0), x_speed)

            # x_tmp = [q[j][0] for j in range(5) if q[j] is not None]
            # y_tmp = [q[j][1] for j in range(5) if q[j] is not None]

            ## 落點預測 ######################################################################################################
            # if len(x_tmp) >= 3:
            #     # 檢查是否嚴格遞增或嚴格遞減,(軌跡方向是否相同)
            #     isSameWay = Monotonic(x_tmp)
            #     # 累積有三顆球的軌跡且同一方向, 可計算拋物線
            #     if isSameWay:
            #         parabola = Solve_Parabola(np.array(x_tmp), np.array(y_tmp))
            #         a, b, c = parabola[0]
            #         fit = a * x_c_pred**2 + b * x_c_pred + c
            #         print(fit)
            #     else:
            #         print(isSameWay)

            isball.append((x_c_pred, y_c_pred))
            q.appendleft(isball[-1] if len(isball) != 0 else None)
            q.pop()

        # 顯示當前桢編號
        # cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示帧并等待键盘输入
        # cv2.imshow("Video Player", frame)
        # key = cv2.waitKey(0) & 0xFF

        # 检查按键
        # if key == ord("q"):  # 按 'q' 键退出
        #     break
        # elif key == ord("a"):  # 按 'a' 键后退一帧
        #     if current_frame > 0:
        #         current_frame -= 1
        #         frame_number -= 1
        #         cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        # elif key == ord("d"):  # 按 'd' 键前进一帧
        #     if current_frame < cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
        #         current_frame += 1
        #         frame_number += 1
        #         cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        ## 畫軌跡圖
        # past_ = sorted([b_idx for b_idx,(b_x, b_y, b_w, b_h) in enumerate(ball_track_list) if (b_x, b_y, b_w, b_h) != (-1,-1,-1,-1)], reverse=True)
        past_ = sorted(
            [
                b_idx
                for b_idx, (b_x, b_y, b_w, b_h) in enumerate(ball_track_list)
                if (b_x, b_y, b_w, b_h) != (-1, -1, -1, -1)
            ],
            reverse=False,
        )
        if past_ != []:
            # past_idx = max(past_)
            past_idx = min(past_)
            # if stop_idx == 0:
            #     stop_idx = past_[0]

        # 變數
        # 010六角正手拉黑色
        shot_split = [(285, 503), (1056, 1250), (1842, 2070), (2634, 2840), (3456, 3660)]

        # 010傳統反手拉紅色
        # shot_split = [(2, 133), (745, 914), (1500, 1641), (2266, 2405), (2998, 3147)]
        for b_cnt, b_idx in enumerate(past_):
            # if b_idx == stop_idx:
            #     break

            (b_x, b_y, b_w, b_h) = ball_track_list[b_idx]
            b_cx, b_cy = int((b_x) * WIDTH), int((b_y) * HEIGHT)
            # cv2.circle(frame, (b_cx, b_cy), 4, (97, 220, 186-5*(len(past_)-b_idx)), 2)

            ## 畫線
            (past_x, past_y, _, _) = ball_track_list[past_idx]
            past_cx, past_cy = int((past_x) * WIDTH), int((past_y) * HEIGHT)
            # if np.abs(past_cx - b_cx) < 100 and np.abs(past_cy - b_cy) < 100:
            #     cv2.line(frame, (b_cx, b_cy), (past_cx, past_cy), (97, 220, max(200-5*(len(past_)-b_idx), 0)), 5)
            #     past_idx = b_idx
            past_idx = b_idx

            if math.sqrt(sum((b - past) ** 2 for b, past in zip((b_cx, b_cy), (past_cx, past_cy)))) > 300:
                # if math.dist((b_cx, b_cy), (past_cx, past_cy)) > 300:
                stop_idx = b_idx
                continue
                # break

            # if past_[0] == stop_idx:
            #     color_R = 0
            # else:
            #     color_R = np.abs(b_idx - stop_idx) * np.abs(255 / (past_[0] - stop_idx))
            # cv2.line(frame, (b_cx, b_cy), (past_cx, past_cy), (97, 220, color_R), 5)

            if b_cx - past_cx > 0:
                if shot_split[0][0] <= b_idx <= shot_split[0][1]:
                    cv2.line(frame, (b_cx, b_cy), (past_cx, past_cy), (0, 255, 255), 3)
                elif shot_split[1][0] <= b_idx <= shot_split[1][1]:
                    cv2.line(frame, (b_cx, b_cy), (past_cx, past_cy), (0, 97, 255), 3)
                elif shot_split[2][0] <= b_idx <= shot_split[2][1]:
                    cv2.line(frame, (b_cx, b_cy), (past_cx, past_cy), (0, 252, 124), 3)
                elif shot_split[3][0] <= b_idx <= shot_split[3][1]:
                    cv2.line(frame, (b_cx, b_cy), (past_cx, past_cy), (255, 255, 0), 3)
                elif shot_split[4][0] <= b_idx <= shot_split[4][1]:
                    cv2.line(frame, (b_cx, b_cy), (past_cx, past_cy), (240, 32, 160), 3)
        if shot_split[0][0] <= frame_number <= shot_split[0][1]:
            cv2.putText(
                frame,
                f"velocity: {x_speed:.2f} (km/hr)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
        elif shot_split[1][0] <= frame_number <= shot_split[1][1]:
            cv2.putText(
                frame,
                f"velocity: {x_speed:.2f} (km/hr)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 97, 255),
                2,
            )
        elif shot_split[2][0] <= frame_number <= shot_split[2][1]:
            cv2.putText(
                frame,
                f"velocity: {x_speed:.2f} (km/hr)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 252, 124),
                2,
            )
        elif shot_split[3][0] <= frame_number <= shot_split[3][1]:
            cv2.putText(
                frame,
                f"velocity: {x_speed:.2f} (km/hr)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )
        elif shot_split[4][0] <= frame_number <= shot_split[4][1]:
            cv2.putText(
                frame,
                f"velocity: {x_speed:.2f} (km/hr)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (240, 32, 160),
                2,
            )

            # direction = 'left' if b_cx - past_cx > 0 else 'right' if b_cx - past_cx < 0 else 'null'
            # if direction != 'null':
            #     if direction != past_direction:
            #         wrong_direct_list.append(1)
            #     else:
            #         wrong_direct_list.clear()

            #     if len(wrong_direct_list) == 3:
            #         stop_idx = b_idx
            #         past_direction = 'right' if past_direction == 'left' else 'left'
            #         wrong_direct_list.clear()
            #         break
        #     if set(ball_track_list[-30:]) == {(-1, -1, -1, -1)} :
        #         stop_idx = past_[1]

        # stop_idx -= 1 if stop_idx != 0 else stop_idx

        # cv2.putText(
        #             frame,
        #             f'frame : {frame_number}',
        #             (int(50), int(50)),
        #             0,
        #             1,
        #             (0,0,0),
        #             2,
        #             cv2.LINE_AA,
        #         )
        new_v.write(frame)

    cap.release()
    new_v.release()

    return None


def main(num_workers=4):
    # v_name_list = [v_path.split('.')[0] for v_path in os.listdir(v_root)]
    # v_path_list = [os.path.join(v_root, v_path) for v_path in os.listdir(v_root)]
    # lbl_dir_list = [os.path.join(lbl_root, v_name) for v_name in v_name_list]

    v_name = "010HexagonalForehandPullBlack"
    v_path = "./runs/detect/yolov7_20231122/010HexagonalForehandPullBlack.MP4"
    lbl_dir = lbl_root

    subtask(v_name, v_path, lbl_dir)

    return None


if __name__ == "__main__":
    main()
