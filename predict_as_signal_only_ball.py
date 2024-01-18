import argparse
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


def PJcurvature(x, y):
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


def Custom_Time(time):
    # time: in milliseconds
    remain = int(time / 1000)
    ms = (time / 1000) - remain
    cnt = remain % 60
    cnt += ms
    remain = int(remain / 60)
    m = remain % 60
    remain = int(remain / 60)
    h = remain
    # Generate custom time string
    cts = ""
    if len(str(h)) >= 2:
        cts += str(h)
    else:
        for i in range(2 - len(str(h))):
            cts += "0"
        cts += str(h)

    cts += ":"
    if len(str(m)) >= 2:
        cts += str(m)
    else:
        for i in range(2 - len(str(m))):
            cts += "0"
        cts += str(m)

    cts += ":"
    if len(str(int(cnt))) == 1:
        cts += "0"
    cts += str(cnt)

    return cts


def Custom_Loss(y_true, y_pred):
    # Loss function
    loss = (-1) * (
        K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1))
        + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1))
    )
    return K.mean(loss)


def Monotonic(L):
    # 檢查單調函數(嚴格遞增或遞減)
    # strictly_increasing = all(x > y for x, y in zip(L, L[1:]))
    # strictly_decreasing = all(x < y for x, y in zip(L, L[1:]))
    non_increasing = all(x >= y for x, y in zip(L, L[1:]))
    non_decreasing = all(x <= y for x, y in zip(L, L[1:]))
    return non_increasing or non_decreasing


def Euclidean_Distance(x, y, x1, y1):
    # 計算歐式距離
    ed = math.sqrt(pow(x - x1, 2) + pow(y - y1, 2))
    return ed


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


def Generate_HeatMap(w, h, x_c, y_c, r, mag):
    # 生成熱力圖(觀察球的mask)
    if x_c < 0 or y_c < 0:
        return np.zeros((h, w))
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (y_c + 1)) ** 2) + ((x - (x_c + 1)) ** 2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap * mag


def Count_BounceLocation(frame, location_list, height, width):
    # 落點分析 bounce analyize function
    row = int(frame[0] / int(width / 4))
    column = int(frame[1] / int(height / 3))
    for i in range(4):
        for j in range(3):
            if row == i and column == j:
                location_list[i][j] += 1
    return location_list


def Detect_Color_Level(score, side, side_min, side_max):
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


def Draw_Bounce_Analysis(
    bounce_analyze_img,
    bounce_location_list,
    miniboard_edge,
    miniboard_height,
    miniboard_width,
    miniboard_text_bias,
):
    # 落點分析圖
    left_bounce_sum = 0
    right_bounce_sum = 0
    score_table = np.zeros([4, 3], dtype=int)
    left_score = 0
    right_score = 0
    left_score_max = 0
    left_score_min = 0
    right_score_max = 0
    right_score_min = 0
    color = []
    text = ""
    # calculate side sum
    for i in range(4):
        for j in range(3):
            if i < 2:  # left
                left_bounce_sum += bounce_location_list[i][j]
            else:  # right
                right_bounce_sum += bounce_location_list[i][j]
    # print("left_sum: ",left_bounce_sum,"right_sum: ",right_bounce_sum)
    # calculate side score
    for i in range(4):
        for j in range(3):
            if i < 2:
                if left_bounce_sum == 0:
                    score_table[i][j] = 0
                else:
                    score_table[i][j] = int(np.round((bounce_location_list[i][j] / left_bounce_sum) * 100, 0))
            else:
                if right_bounce_sum == 0:
                    score_table[i][j] = 0
                else:
                    score_table[i][j] = int(np.round((bounce_location_list[i][j] / right_bounce_sum) * 100, 0))
    # print("score_table",score_table)

    # find max and min
    left_score_min = score_table[0][0]
    left_score_max = score_table[0][0]
    right_score_min = score_table[2][0]
    right_score_max = score_table[2][0]

    for i in range(4):
        for j in range(3):
            if i < 2:
                left_score = score_table[i][j]
                if left_score < left_score_min:
                    left_score_min = left_score
                if left_score > left_score_max:
                    left_score_max = left_score
            else:
                right_score = score_table[i][j]
                if right_score < right_score_min:
                    right_score_min = right_score
                if right_score > right_score_max:
                    right_score_max = right_score

    for i in range(2):
        for j in range(3):
            color_detect = Detect_Color_Level(score_table[i][j], "left", left_score_min, left_score_max)
            text = str(score_table[i][j]) + "%"
            #   print("color detect:",color_detect)
            cv2.rectangle(
                bounce_analyze_img,
                (
                    miniboard_edge + (i * int(miniboard_width / 4)) + 10,
                    miniboard_edge + (j * int(miniboard_height / 3)) + 10,
                ),
                (
                    ((i + 1) * int(miniboard_width / 4)) + miniboard_edge - 10,
                    ((j + 1) * int(miniboard_height / 3)) + miniboard_edge - 10,
                ),
                color=color_detect,
                thickness=-1,
            )
            cv2.putText(
                bounce_analyze_img,
                text,
                (
                    miniboard_edge + (i * int(miniboard_width / 4)) + miniboard_edge * 2,
                    miniboard_edge + (j * int(miniboard_height / 3)) + miniboard_text_bias,
                ),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (1, 1, 1),
                1,
                cv2.LINE_AA,
            )
    for i in range(2, 4):
        for j in range(3):
            color_detect = Detect_Color_Level(score_table[i][j], "right", right_score_min, right_score_max)
            text = str(score_table[i][j]) + "%"
            #   print("color detect:",color_detect)
            cv2.rectangle(
                bounce_analyze_img,
                (
                    miniboard_edge + (i * int(miniboard_width / 4)) + 10,
                    miniboard_edge + (j * int(miniboard_height / 3)) + 10,
                ),
                (
                    ((i + 1) * int(miniboard_width / 4)) + miniboard_edge - 10,
                    ((j + 1) * int(miniboard_height / 3)) + miniboard_edge - 10,
                ),
                color=color_detect,
                thickness=-1,
            )
            cv2.putText(
                bounce_analyze_img,
                text,
                (
                    miniboard_edge + (i * int(miniboard_width / 4)) + miniboard_edge * 2,
                    miniboard_edge + (j * int(miniboard_height / 3)) + miniboard_text_bias,
                ),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (1, 1, 1),
                1,
                cv2.LINE_AA,
            )
    return bounce_analyze_img


def Draw_SpeedHist(left_speed_list, right_speed_list):
    # 繪製速度直方圖
    stroke_length = 0
    shots_list = []
    # 平衡左右 list ??
    if len(left_speed_list) > len(right_speed_list):
        stroke_length = len(left_speed_list)
        for i in range(len(left_speed_list) - len(right_speed_list)):
            right_speed_list.append(0)
    else:
        stroke_length = len(right_speed_list)
        for i in range(len(right_speed_list) - len(left_speed_list)):
            left_speed_list.append(0)

    for i in range(stroke_length):
        shots_list.append(i + 1)

    label_left = f"left_player mean:{str(round(np.mean(left_speed_list),2))}"
    label_right = f"right_player mean:{str(round(np.mean(right_speed_list),2))}"
    # bins = np.linspace(0, 10, 10)
    plt.figure(figsize=(15, 10), dpi=100, linewidth=2)
    plt.plot(shots_list, left_speed_list, "s-", color="royalblue", label=label_left)
    plt.plot(shots_list, right_speed_list, "o-", color="darkorange", label=label_right)
    plt.title(f"Comparing the shot speed between the two players", x=0.5, y=1.03, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(f"shots", fontsize=30, labelpad=15)
    plt.ylabel(f"Km/hr", fontsize=30, labelpad=20)
    plt.legend(loc="best", fontsize=20)
    plt.savefig(f"shot_speed_compare.png")
    plt.clf()

    plt.figure(figsize=(15, 10), dpi=100, linewidth=2)
    plt.hist([left_speed_list, right_speed_list], bins="auto", alpha=1, label=["left", "right"])

    plt.xlabel(f"Km/hr", fontsize=30, labelpad=15)
    plt.ylabel(f"shots", fontsize=30, labelpad=20)
    plt.legend(loc="upper right")
    plt.savefig(f"shot_speed_compare_2.png")
    plt.clf()
    # plt.show()


def Draw_MiniBoard(height, width, edge, option=None):
    img_opt = np.zeros([height + edge * 2, width + edge * 2, 3], dtype=np.uint8)
    cv2.rectangle(
        img_opt,
        (edge, edge),
        (width + edge, height + edge),
        color=(255, 255, 255),
        thickness=7,
    )
    cv2.rectangle(
        img_opt,
        (edge, edge),
        (width + edge, height + edge),
        color=(255, 150, 50),
        thickness=-1,
    )
    cv2.line(
        img_opt,
        (int(width / 2) + edge, edge),
        (int(width / 2) + edge, height + edge),
        (255, 255, 255),
        5,
    )
    if option == "bounce":
        cv2.line(
            img_opt,
            (int(width / 4) + edge, edge),
            (int(width / 4) + edge, height + edge),
            (128, 0, 128),
            3,
        )
        cv2.line(
            img_opt,
            (int((width / 4) * 3) + edge, edge),
            (int((width / 4) * 3) + edge, height + edge),
            (128, 0, 128),
            3,
        )
        cv2.line(
            img_opt,
            (edge, int(height / 3) + edge),
            (width + edge, int(height / 3) + edge),
            (128, 0, 128),
            3,
        )
        cv2.line(
            img_opt,
            (edge, int((height / 3) * 2) + edge),
            (width + edge, int((height / 3) * 2) + edge),
            (128, 0, 128),
            3,
        )
    return img_opt


def Draw_Circle(event, x, y, flags, param):
    # 用於透視變形取點
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(param["img"], (x, y), 3, (0, 255, 255), -1)
        param["point_x"].append(x)
        param["point_y"].append(y)


def Draw_and_Collect_Data(
    count,
    color,
    img_opt,
    img_opt_bounce_location,
    bouncing_offset,
    miniboard_args,
    PT_dict,
    inv,
    loc_PT,
    bounce_location_list,
    bounce,
    q_bv,
):
    bouncing_offset_x, bouncing_offset_y = bouncing_offset
    miniboard_height, miniboard_width = miniboard_args
    cv2.circle(img_opt, PT_dict[count], 5, color, 4)  # red
    # add to bounce map
    cv2.circle(
        img_opt_bounce_location,
        (
            PT_dict[count][0] + bouncing_offset_x,
            PT_dict[count][1] + bouncing_offset_y,
        ),
        5,
        color,
        4,
    )
    # analyze location
    bll = Count_BounceLocation(
        PT_dict[count],
        bounce_location_list,
        miniboard_height,
        miniboard_width,
    )
    p_inv = Perspective_Transform(inv, loc_PT)
    bounce.append([count, p_inv[0], p_inv[1]])
    q_bv.appendleft(p_inv)
    q_bv.pop()
    return bll, bounce, q_bv


def predict_class(df_player, threshold=10):
    df = pd.DataFrame(df_player)
    df["seq"] = df.ne(df.shift()).cumsum()
    section = df.groupby(df["seq"]).size().ge(threshold)

    df["class"] = 0
    for idx, sec in section.items():
        if sec:
            df.loc[df["seq"] == idx, "class"] = df[df.columns[0]]

    return df["class"]


def Normalize_df(df, usecols):
    data_all = pd.DataFrame(columns=usecols)

    for i in pd.unique(df["Person_id"]):
        data_tmp = df[usecols].loc[df["Person_id"] == i].copy()
        col_X = [col for col in data_tmp.columns if "_X" in col]
        col_Y = [col for col in data_tmp.columns if "_Y" in col]
        data_tmp[col_X] = (data_tmp[col_X] - data_tmp[col_X].min()) / (data_tmp[col_X].max() - data_tmp[col_X].min())
        data_tmp[col_Y] = (data_tmp[col_Y] - data_tmp[col_Y].min()) / (data_tmp[col_Y].max() - data_tmp[col_Y].min())
        data_all = pd.concat([data_all, data_tmp])

    return data_all


def data_preprocessing(data):
    usecols = [
        "Frame",
        "Person_id",
        "RShoulder",
        "RElbow",
        "RWrist",
        "LShoulder",
        "LElbow",
        "LWrist",
    ]

    data_all = pd.DataFrame()
    for col in usecols:
        if col in ["Frame", "Person_id"]:
            data_all[col] = []
        else:
            data_all[col + "_X"] = []
            data_all[col + "_Y"] = []
    for i in pd.unique(data["Person_id"]):
        data_tmp = data[data_all.columns].loc[data["Person_id"] == i].copy()

        col_X = [col for col in data_tmp.columns if "_X" in col]
        col_Y = [col for col in data_tmp.columns if "_Y" in col]

        ## 補缺失值
        imp_mean = IterativeImputer(random_state=0, max_iter=10000, min_value=0, max_value=1.0)

        while data_tmp.isnull().sum().any():
            data_tmp[col_X] = imp_mean.fit_transform(data_tmp[col_X])
            data_tmp[col_Y] = imp_mean.fit_transform(data_tmp[col_Y])

        ## 刪除異常值(outlier), 以 Z-score 統計方法來計算
        data_z = data_tmp.iloc[:, 2:].copy()
        zcore = ((data_z - data_z.mean()) / data_z.std()).abs() > 1.5
        # data_z = data_z[zcore == False]
        data_z[zcore == True] = np.nan
        data_tmp.iloc[:, 2:] = data_z

        ## 補償異常值
        while data_tmp.isnull().sum().any():
            data_tmp[col_X] = imp_mean.fit_transform(data_tmp[col_X])
            data_tmp[col_Y] = imp_mean.fit_transform(data_tmp[col_Y])

        # ## 資料正規化
        # data_tmp[col_X] = ((data_tmp[col_X]-data_tmp[col_X].min())/(data_tmp[col_X].max()-data_tmp[col_X].min()))
        # data_tmp[col_Y] = ((data_tmp[col_Y]-data_tmp[col_Y].min())/(data_tmp[col_Y].max()-data_tmp[col_Y].min()))

        data_all = pd.concat([data_all, data_tmp])

    ## reindexing
    data_all.reset_index(inplace=True, drop=True)
    return data_all


def string_to_numeric(df):
    df = df.apply(lambda x: re.sub("[(,)]", "", x))
    df = df.apply(lambda x: [float(y) for _, y in enumerate(x.split(" "))])
    return df


def list_split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def keypoints_to_data(root, files, output_dir):
    ## BODY_25 keypoints
    allcols = [
        "Frame",
        "Person_id",
        "Nose",
        "Neck",
        "RShoulder",
        "RElbow",
        "RWrist",
        "LShoulder",
        "LElbow",
        "LWrist",
        "MidHip",
        "RHip",
        "RKnee",
        "RAnkle",
        "LHip",
        "LKnee",
        "LAnkle",
        "REye",
        "LEye",
        "REar",
        "LEar",
        "LBigToe",
        "LSmallToe",
        "LHeel",
        "RBigToe",
        "RSmallToe",
        "RHeel",
    ]
    usecols = [
        "Frame",
        "Person_id",
        "RShoulder",
        "RElbow",
        "RWrist",
        "LShoulder",
        "LElbow",
        "LWrist",
    ]

    current = os.path.split(root)[-1]
    output_path = f"{os.path.join(output_dir, current)}.csv"

    ## DataFrame initialization
    json_df = pd.DataFrame()
    for col in allcols:
        if col in ["Frame", "Person_id"]:
            json_df[col] = []
        else:
            json_df[col + "_X"] = []
            json_df[col + "_Y"] = []

    for frame, json_file in enumerate(sorted(files)):
        with open(os.path.join(root, json_file), "r", encoding="utf-8") as jf:
            json_dict = json.load(jf)

        ## 這次畫面偵測到幾人, 從左到右排序
        num_of_person = len(json_dict["people"])
        sort_position = sorted([json_dict["people"][idx]["pose_keypoints_2d"][0] for idx in range(num_of_person)])

        for idx in range(num_of_person):
            ## 挑選由左數來第p人
            p = sort_position.index(sort_position[idx])
            keypoints = list_split(json_dict["people"][p]["pose_keypoints_2d"], len(allcols[2:]))
            ## 分別取出 x, y 座標
            keypoints_x = [x for x, _, _ in keypoints]
            keypoints_y = [y for _, y, _ in keypoints]
            ## 檢查 keypoints 位置  (假設最左及最右為選手)
            k_test = np.mean(np.array([x for x in keypoints_x if x != 0]))
            ## 0 : 左邊的人, 1 : 右邊的人
            if 0 <= k_test < 0.4:
                person_id = 0
            elif 0.6 < k_test <= 1:
                person_id = 1
            else:
                continue

            if frame in json_df[json_df["Person_id"] == person_id]["Frame"].astype(int).values:
                print(frame, "Special case")
                continue
            keypoints_x = [frame] + keypoints_x
            keypoints_y = [person_id] + keypoints_y

            json_df.loc[len(json_df.index)] = pd.Series([None for _ in json_df.columns])  ## initial row with None
            json_df.loc[len(json_df.index) - 1][json_df.columns[0::2]] = keypoints_x  # Frame and col_X
            json_df.loc[len(json_df.index) - 1][json_df.columns[1::2]] = keypoints_y  # Person_id and col_Y

        a = json_df[json_df["Person_id"] == 0]
        b = json_df[json_df["Person_id"] == 1]
        if frame not in a["Frame"].astype(int).values:
            json_df.loc[len(json_df.index)] = pd.Series([None for _ in json_df.columns])
            json_df.iloc[-1:, :2] = [frame, 0]

        if frame not in b["Frame"].astype(int).values:
            json_df.loc[len(json_df.index)] = pd.Series([None for _ in json_df.columns])
            json_df.iloc[-1:, :2] = [frame, 1]
    json_df = json_df.sort_values(by=["Person_id", "Frame"])
    json_df.reset_index(inplace=True, drop=True)

    ## 資料合併
    json_df = data_preprocessing(json_df.copy())
    # for col in usecols:
    #     if col not in ["Frame", "Person_id"]:
    #         json_df[col] = list(zip(json_df[col + "_X"], json_df[col + "_Y"]))
    #         json_df.drop(columns=[col + "_X", col + "_Y"], inplace=True)
    #     else:
    #         json_df[col] = json_df[col].astype(int)

    json_df.to_csv(f"{output_path}", index=False)

    return json_df


def video_to_keypoints(video_path, keypoints_dir):
    origin_path = os.getcwd()
    openpose_path = "../openpose/"
    openpose_exe_path = "./build/examples/openpose/openpose.bin"
    os.chdir(f"{openpose_path}")
    video_name = os.path.basename(video_path).split(".")[0]
    json_path = f"{keypoints_dir}/{video_name}/"
    os.makedirs(json_path, exist_ok=True)
    os.system(
        f"{openpose_exe_path} --video {video_path} --write_json {json_path} --keypoint_scale 3  --display 0 --render_pose 0 --number_people_max 3"
    )
    os.chdir(f"{origin_path}")
    return "Done!"


def Create_Output_Dir(output_path, video_name):
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

    return (
        video_path,
        record_ball_path,
        record_pose_path,
        analysis_img_path,
        bounce_loc_path,
        bounce_img_path,
        keypoints_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("--input", required=True, type=str, help="Input video")
    args = parser.parse_args()
    return args


### 後處理從此開始 ###
def main():
    # 全域參數
    HEIGHT = 288  # model input size
    WIDTH = 512

    # 影片跟目錄
    root_path = f"./runs/detect/yolov7_202311222"
    input_path = os.path.join(root_path, "010_Hexagonal_Backhand_Pull_Red.MP4")

    # 發球(A是左邊，B是右邊)
    is_A_serve = False
    is_B_serve = True

    # yolo labels path
    label_path = os.path.join(root_path, "labels")

    # 影片位址
    base_name = os.path.splitext(os.path.basename(input_path))
    video_name = base_name[0]
    video_suffix = base_name[1]

    # 建立目錄
    output_path = f"./inference/output"
    (
        video_path,
        record_ball_path,
        record_pose_path,
        analysis_img_path,
        bounce_loc_path,
        bounce_img_path,
        keypoints_path,
    ) = Create_Output_Dir(output_path, video_name)

    # 選擇影片編碼
    if video_suffix in [".avi"]:
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    elif video_suffix in [".mp4", ".MP4", ".mov", ".MOV"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        print("usage: video type can only be .avi or .mp4 or .MOV")
        exit(1)

    # 讀取影片
    cap = cv2.VideoCapture(input_path)
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height, frame_width = int(cap.get(4)), int(cap.get(3))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    success, image = cap.read()
    if not success:
        return

    # 等比例縮放
    ratio = frame_height / HEIGHT
    size = (int(WIDTH * ratio), int(HEIGHT * ratio))

    # 寫 預測結果
    output = cv2.VideoWriter(
        f"{video_path}/{video_name}_predict_12.mp4",
        fourcc,
        framerate,
        size,
    )

    # 點選透視變形位置, 順序為:左上,左下,右下,右上
    PT_dict = {}
    PT_data = {"img": image.copy(), "point_x": [], "point_y": []}
    # TODO: 測試用
    # PT_data["point_x"] = [690, 625, 1362, 1274]
    # PT_data["point_y"] = [709, 812, 821, 717]
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

    # 小黑板的設定值
    if frame_height == 720:
        miniboard_width = 366
        miniboard_height = 204
        miniboard_edge = 15
        miniboard_text_bias = 40
    else:
        miniboard_width = 548
        miniboard_height = 305
        miniboard_edge = 20
        miniboard_text_bias = 60

    # PerspectiveTransform
    upper_left = [PT_data["point_x"][0], PT_data["point_y"][0]]
    lower_left = [PT_data["point_x"][1], PT_data["point_y"][1]]
    lower_right = [PT_data["point_x"][2], PT_data["point_y"][2]]
    upper_right = [PT_data["point_x"][3], PT_data["point_y"][3]]
    pts1 = np.float32([upper_left, lower_left, lower_right, upper_right])
    pts2 = np.float32(
        [
            [miniboard_edge, miniboard_edge],
            [miniboard_edge, miniboard_height + miniboard_edge],
            [miniboard_width + miniboard_edge, miniboard_height + miniboard_edge],
            [miniboard_width + miniboard_edge, miniboard_edge],
        ]
    )
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    inv = cv2.getPerspectiveTransform(pts2, pts1)

    # 繪製迷你落點板
    img_opt = Draw_MiniBoard(miniboard_height, miniboard_width, miniboard_edge)
    img_opt_bounce_location = Draw_MiniBoard(miniboard_height, miniboard_width, miniboard_edge, "bounce")
    img_opt_Bounce_analysis = Draw_MiniBoard(miniboard_height, miniboard_width, miniboard_edge, "bounce")

    # 參數
    bounce = []
    FPS_list = []
    left_speed_list, right_speed_list = [], []
    bounce_location_list = [[0 for _ in range(3)] for _ in range(4)]
    bouncing_offset_x, bouncing_offset_y = 10, 15  # bouncing location offset
    speed_left, speed_right = 0, 0  # 左右選手球速
    bounce_frame_A, bounce_frame_B = -1, -1  # 出現落點的Frame
    left_shot_count, right_shot_count = 0, 0  #
    serve_detection_count = 0
    miss_detection_count = 0
    shotspeed = 0
    shotspeed_tmp = 0
    if is_A_serve:
        now_player = 0  # 0:左邊選手, 1: 右邊選手
    if is_B_serve:
        now_player = 1  # 0:左邊選手, 1: 右邊選手
    hit_count = 0  # 擊球次數
    count = 1  # 記錄處理幾個 Frame
    MAX_velo = 0
    serve_signal_count = np.NINF  # 發球時間
    x_c_pred, y_c_pred = np.inf, np.inf  # 球體中心位置
    is_first_ball = True

    # In order to draw the trajectory of tennis, we need to save the coordinate of preious 12 frames
    q = queue.deque([None for _ in range(12)])
    # bounce detection init
    q_bv = queue.deque([None for _ in range(6)])

    # 針對每一貞做運算
    start = time.time()
    batch = 12
    n = 4
    k = batch // 2
    while success:
        prev_time = time.time()

        """ TODO: 不需要判斷發球
        if count >= k:
            cnt = count - k

            if ans_df.loc[cnt, "class_1"] in [1, 2]:
                is_A_serve = True
                print(f"<-------Frame : {count} A serve!!------->")
            elif ans_df.loc[cnt, "class_2"] in [1, 2]:
                is_B_serve = True
                print(f"<-------Frame : {count} B serve!!------->")


            ## 偷吃步 過濾一些非發球
            point_x = (upper_right[0] + upper_left[0]) / 2
            range_x = (upper_right[0] - upper_left[0]) / 3
            point_y = max(lower_right[1], lower_left[1])
            range_y = 75
            point_lbd = (1 * lower_left[0]) / 10
            point_rbd = (9 * 1920 - lower_right[0]) / 10

            temp_ball = record_ball_df.loc[cnt : cnt + batch - 1, "Visibility"].values
            if np.count_nonzero(temp_ball) > k / 2:
                temp_ball_x = record_ball_df[record_ball_df["Visibility"] != 0].loc[cnt : cnt + batch - 1, "X"].values
                temp_ball_y = record_ball_df[record_ball_df["Visibility"] != 0].loc[cnt : cnt + batch - 1, "Y"].values

                thresh_ball_x = np.mean(temp_ball_x)
                thresh_ball_y = np.mean(temp_ball_y)

                if (is_A_serve or is_B_serve) and (
                    (point_x - range_x <= thresh_ball_x <= point_x + range_x)
                    or (thresh_ball_y >= point_y + range_y)
                    or not (point_lbd <= thresh_ball_x <= point_rbd)
                ):
                    miss_detection_count += 1
                    is_A_serve = False
                    is_B_serve = False
        """
        if is_A_serve or is_B_serve:
            is_serve_wait = False
            serve_signal_count = count
            hit_count = 0
            bounce_frame_A, bounce_frame_B = -1, -1
            img_opt = Draw_MiniBoard(miniboard_height, miniboard_width, miniboard_edge)
            print(f"<---Serve detected by OpenPose with ball trajectory--->")
            serve_detection_count += 1
            if ((count / 59) % 60) > 0:
                print(f"Frame : {count}, serve ball at {int((count/59)/60)} m {int(count/59) % (60)} s")
            else:
                print(f"Frame : {count}, serve ball at {int(count/59)} s")
            # 初始化參數後取消
            is_A_serve, is_B_serve = False, False
            print(f"<----------------------------------------------------->")
        ######################################################
        isball = []
        image_CV = image.copy()

        label_file = os.path.join(label_path, f"{video_name}_{count}.txt")
        ## 檔案存在
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                balls = []
                has_ball = False
                for line in f:
                    l = line.split()
                    if len(l) > 0:
                        if int(l[0]) == 0:
                            has_ball = True
                            balls.append(l)
                ## 有偵測到球體
                if has_ball:
                    distance = sys.float_info.max
                    for ball in balls:
                        ball_x_c_pred, ball_y_c_pred = int(float(ball[1]) * frame_width), int(
                            float(ball[2]) * frame_height
                        )
                        # 找尋最接近上次的球
                        if x_c_pred == float("inf") or y_c_pred == float("inf"):
                            x_c_pred, y_c_pred = ball_x_c_pred, ball_y_c_pred
                        elif distance > math.sqrt((ball_x_c_pred - x_c_pred) ** 2 + (ball_y_c_pred - y_c_pred) ** 2):
                            distance = math.sqrt((ball_x_c_pred - x_c_pred) ** 2 + (ball_y_c_pred - y_c_pred) ** 2)
                            x_c_pred, y_c_pred = ball_x_c_pred, ball_y_c_pred

                    balls = 5 if is_first_ball else 9
                    x_tmp = [q[j][0] for j in range(balls) if q[j] is not None]
                    y_tmp = [q[j][1] for j in range(balls) if q[j] is not None]
                    ## 落點預測 ######################################################################################################
                    if len(x_tmp) >= 3:
                        # 檢查是否嚴格遞增或嚴格遞減,(軌跡方向是否相同)
                        isSameWay = Monotonic(x_tmp)
                        # 累積有三顆球的軌跡且同一方向, 可計算拋物線
                        if isSameWay:
                            parabola = Solve_Parabola(np.array(x_tmp), np.array(y_tmp))
                            a, b, c = parabola[0]
                            fit = a * x_c_pred**2 + b * x_c_pred + c
                            # cv2.circle(image_CV, (x_c_pred, int(fit)), 5, (255, 0, 0), 4)
                            # 差距 5 個 pixel 以上視為脫離預測的拋物線
                            if abs(y_c_pred - fit) >= 10:
                                x_last = x_tmp[0]
                                # 預測球在球桌上的落點, x_drop : 本次與前次的中點, y_drop : x_drop 於拋物線上的位置
                                x_drop = int(round((x_c_pred + x_last) / 2, 0))
                                y_drop = int(round(a * x_drop**2 + b * x_drop + c, 0))
                                # 繪製本次球體位置, Golden
                                cv2.circle(image_CV, (x_c_pred, y_c_pred), 5, (0, 215, 255), 4)
                                # 透視變形計算本次球體在迷你板上的位置
                                loc_PT = Perspective_Transform(matrix, (x_drop, y_drop))
                                # 如果變換後落在迷你板內
                                if (
                                    loc_PT[0] >= miniboard_edge - 1
                                    and loc_PT[0] < miniboard_width + miniboard_edge + 5
                                    and loc_PT[1] >= miniboard_edge - 5
                                    and loc_PT[1] < miniboard_height + miniboard_edge + 5
                                ):
                                    PT_dict[count] = loc_PT
                                    restart_list = list(PT_dict.keys())
                                    """
                                    一局結束判斷
                                    1. 倒數兩球距離過大 (飛出界)
                                    2. 停留在桌上 (被網子攔住)
                                    """
                                    if len(restart_list) >= 2 and (int(restart_list[-1]) - int(restart_list[-2])) > 200:
                                        is_serve_wait = False
                                        if is_A_serve:
                                            now_player = 0
                                        elif is_B_serve:
                                            now_player = 1
                                        bounce_frame_A, bounce_frame_B = -1, -1
                                        hit_count = 0
                                        print(f"<---Frame : {count}, round end.--->")
                                        img_opt = Draw_MiniBoard(
                                            miniboard_height,
                                            miniboard_width,
                                            miniboard_edge,
                                        )
                                    # 落點在左側
                                    if PT_dict[count][0] <= int(miniboard_width / 2) + miniboard_edge:
                                        # 首次發球 或 二次發球
                                        if not is_serve_wait:
                                            is_first_ball = True
                                            is_serve_wait = True
                                            hit_count = 1
                                            now_player = 1  # switch player
                                            bounce_frame_A = count
                                            img_opt = Draw_MiniBoard(
                                                miniboard_height,
                                                miniboard_width,
                                                miniboard_edge,
                                            )
                                            (
                                                bounce_location_list,
                                                bounce,
                                                q_bv,
                                            ) = Draw_and_Collect_Data(
                                                count,
                                                (0, 0, 255),
                                                img_opt,
                                                img_opt_bounce_location,
                                                (bouncing_offset_x, bouncing_offset_y),
                                                (miniboard_height, miniboard_width),
                                                PT_dict,
                                                inv,
                                                loc_PT,
                                                bounce_location_list,
                                                bounce,
                                                q_bv,
                                            )

                                        # 回擊
                                        elif now_player == 0 and is_serve_wait:
                                            if hit_count > 0:
                                                cv2.line(
                                                    img_opt,
                                                    PT_dict[bounce_frame_B],
                                                    PT_dict[count],
                                                    (0, 255, 0),
                                                    3,
                                                )
                                                bounce_len = Euclidean_Distance(
                                                    PT_dict[bounce_frame_B][0],
                                                    PT_dict[bounce_frame_B][1],
                                                    PT_dict[count][0],
                                                    PT_dict[count][1],
                                                )
                                                speed_bounce_distance_right = abs(
                                                    shotspeed_tmp
                                                    * (100000 / 3600)
                                                    * (right_shot_count - bounce_frame_B)
                                                    / framerate
                                                )
                                                speed_right = np.round(
                                                    (
                                                        (bounce_len * (274 / 366) + speed_bounce_distance_right)
                                                        / (count - right_shot_count)
                                                    )
                                                    * framerate
                                                    * (3600 / 100000),
                                                    1,
                                                )
                                                if speed_right > 100:
                                                    speed_right = 99

                                                shotspeed = speed_right
                                                shotspeed_tmp = speed_right
                                                print(f"Frame : {count} speed_right : {speed_right} ")
                                                right_speed_list.append(speed_right)
                                            is_first_ball = False
                                            hit_count += 1
                                            now_player = 1
                                            bounce_frame_A = count
                                            (
                                                bounce_location_list,
                                                bounce,
                                                q_bv,
                                            ) = Draw_and_Collect_Data(
                                                count,
                                                (0, 0, 255),
                                                img_opt,
                                                img_opt_bounce_location,
                                                (bouncing_offset_x, bouncing_offset_y),
                                                (miniboard_height, miniboard_width),
                                                PT_dict,
                                                inv,
                                                loc_PT,
                                                bounce_location_list,
                                                bounce,
                                                q_bv,
                                            )
                                        # 其他
                                        elif (count - bounce_frame_A) > 60:
                                            print("[------------------------------------------------------------]")
                                            print(
                                                f"sth wrong at frame : {count}, bounce_B : {bounce_frame_B}, hit_count : {hit_count}"
                                            )
                                            print("[------------------------------------------------------------]")
                                            is_first_ball = False
                                            is_serve_wait = True
                                            now_player = 1
                                            bounce_frame_A = count
                                            hit_count = 1
                                            img_opt = Draw_MiniBoard(
                                                miniboard_height,
                                                miniboard_width,
                                                miniboard_edge,
                                            )
                                            (
                                                bounce_location_list,
                                                bounce,
                                                q_bv,
                                            ) = Draw_and_Collect_Data(
                                                count,
                                                (0, 0, 255),
                                                img_opt,
                                                img_opt_bounce_location,
                                                (bouncing_offset_x, bouncing_offset_y),
                                                (miniboard_height, miniboard_width),
                                                PT_dict,
                                                inv,
                                                loc_PT,
                                                bounce_location_list,
                                                bounce,
                                                q_bv,
                                            )

                                    # 落點在右側
                                    elif PT_dict[count][0] >= int(miniboard_width / 2) + miniboard_edge:
                                        # 首次發球 或 二次發球
                                        if not is_serve_wait:
                                            is_first_ball = True
                                            is_serve_wait = True
                                            hit_count = 1
                                            now_player = 0  # switch player
                                            bounce_frame_B = count
                                            img_opt = Draw_MiniBoard(
                                                miniboard_height,
                                                miniboard_width,
                                                miniboard_edge,
                                            )
                                            (
                                                bounce_location_list,
                                                bounce,
                                                q_bv,
                                            ) = Draw_and_Collect_Data(
                                                count,
                                                (80, 127, 255),
                                                img_opt,
                                                img_opt_bounce_location,
                                                (bouncing_offset_x, bouncing_offset_y),
                                                (miniboard_height, miniboard_width),
                                                PT_dict,
                                                inv,
                                                loc_PT,
                                                bounce_location_list,
                                                bounce,
                                                q_bv,
                                            )

                                        # 回擊
                                        elif now_player == 1 and is_serve_wait:
                                            if hit_count > 0:
                                                # like yellow
                                                cv2.line(
                                                    img_opt,
                                                    PT_dict[bounce_frame_A],
                                                    PT_dict[count],
                                                    (115, 220, 255),
                                                    3,
                                                )
                                                bounce_len = Euclidean_Distance(
                                                    PT_dict[bounce_frame_A][0],
                                                    PT_dict[bounce_frame_A][1],
                                                    PT_dict[count][0],
                                                    PT_dict[count][1],
                                                )
                                                speed_bounce_distance_left = abs(
                                                    shotspeed_tmp
                                                    * (100000 / 3600)
                                                    * (left_shot_count - bounce_frame_A)
                                                    / framerate
                                                )
                                                speed_left = np.round(
                                                    (
                                                        (bounce_len * (274 / 366) + speed_bounce_distance_left)
                                                        / (count - left_shot_count)
                                                    )
                                                    * framerate
                                                    * (3600 / 100000),
                                                    1,
                                                )
                                                if speed_left > 100:
                                                    speed_left = 60

                                                shotspeed = speed_left
                                                shotspeed_tmp = speed_left
                                                print(f"Frame : {count} speed_left : {speed_left} ")
                                                left_speed_list.append(speed_left)
                                            is_first_ball = False
                                            hit_count += 1
                                            now_player = 0
                                            bounce_frame_B = count
                                            (
                                                bounce_location_list,
                                                bounce,
                                                q_bv,
                                            ) = Draw_and_Collect_Data(
                                                count,
                                                (80, 127, 255),
                                                img_opt,
                                                img_opt_bounce_location,
                                                (bouncing_offset_x, bouncing_offset_y),
                                                (miniboard_height, miniboard_width),
                                                PT_dict,
                                                inv,
                                                loc_PT,
                                                bounce_location_list,
                                                bounce,
                                                q_bv,
                                            )

                                        # 其他
                                        elif (count - bounce_frame_B) > 60:
                                            print("[------------------------------------------------------------]")
                                            print(
                                                f"sth wrong at frame : {count}, bounce_A : {bounce_frame_A}, hit_count : {hit_count}"
                                            )
                                            print("[------------------------------------------------------------]")
                                            is_first_ball = False
                                            is_serve_wait = True
                                            now_player = 0
                                            bounce_frame_B = count
                                            hit_count = 1
                                            img_opt = Draw_MiniBoard(
                                                miniboard_height,
                                                miniboard_width,
                                                miniboard_edge,
                                            )
                                            (
                                                bounce_location_list,
                                                bounce,
                                                q_bv,
                                            ) = Draw_and_Collect_Data(
                                                count,
                                                (80, 127, 255),
                                                img_opt,
                                                img_opt_bounce_location,
                                                (bouncing_offset_x, bouncing_offset_y),
                                                (miniboard_height, miniboard_width),
                                                PT_dict,
                                                inv,
                                                loc_PT,
                                                bounce_location_list,
                                                bounce,
                                                q_bv,
                                            )

                    ## 超過一秒都沒有球落在球桌上
                    if (count - bounce_frame_A) >= 60 and (count - bounce_frame_B) >= 60:  # 超過1秒
                        is_first_ball = True
                        is_serve_wait = True
                        bounce_frame_A, bounce_frame_B = -1, -1
                        hit_count = 0

                    isball.append((x_c_pred, y_c_pred))

        q.appendleft(isball[-1] if len(isball) != 0 else None)
        q.pop()

        q_bv.appendleft(None)
        q_bv.pop()

        # draw current frame prediction and previous 11 frames as yellow circle, total: 12 frames
        for i in range(12):
            if q[i] is not None:
                cv2.circle(image_CV, (q[i][0], q[i][1]), 5, (0, 255, 255), 1)

        # draw bounce point as red circle
        for i in range(6):
            if q_bv[i] is not None:
                cv2.circle(image_CV, (q_bv[i][0], q_bv[i][1]), 5, (0, 0, 255), 4)

        # Place miniboard on upper right corner
        image_CV[
            : miniboard_height + miniboard_edge * 2,
            frame_width - (miniboard_width + miniboard_edge * 2) :,
        ] = img_opt

        velo = 0
        if q[0] is not None and q[1] is not None and q[2] is not None:
            ball_direction = q[0][0] - q[1][0]
            ball_direction_last = q[1][0] - q[2][0]
            velo = shotspeed
            if MAX_velo == 0:
                MAX_velo = velo
            if ball_direction > 0:  # Direction right
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
                if ball_direction_last >= 0:
                    right_shot_count = count
                    if velo > MAX_velo:
                        MAX_velo = velo
                else:
                    MAX_velo = 0
            elif ball_direction < 0:  # Direction left
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
                if ball_direction_last <= 0:
                    left_shot_count = count
                    if velo > MAX_velo:
                        MAX_velo = velo
                else:
                    MAX_velo = 0
            if MAX_velo > 113:
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
            else:
                cv2.putText(
                    image_CV,
                    "              " + str(velo),
                    (10, 100),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
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
        # cv2.putText(
        #     image_CV,
        #     "serve ball:",
        #     (10, 50),
        #     cv2.FONT_HERSHEY_TRIPLEX,
        #     1,
        #     (0, 255, 255),
        #     1,
        #     cv2.LINE_AA,
        # )
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
            f"Frame : {count}",
            (10, 260),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 發球開綠燈 其餘開紅燈
        # if count < serve_signal_count + 45:
        #     cv2.circle(image_CV, (240, 40), 12, (0, 255, 0), -1)
        # else:
        #     cv2.circle(image_CV, (240, 40), 12, (0, 0, 255), -1)

        # if count > serve_signal_count + 60:
        #     is_serve_wait = True

        output.write(image_CV)

        fps = 1 / (time.time() - prev_time)
        FPS_list.append(fps)

        success, image = cap.read()
        count += 1
        if count >= total_frames - 12:
            break
        """
        elif count >= len(ans_df):
            break
        """

    # For releasing cap and out.
    cap.release()
    output.release()

    # write bouncing list to csv file
    bounce_loc_pd = pd.DataFrame(bounce_location_list)
    bounce_loc_pd.to_csv(f"{bounce_loc_path}/{video_name}_bounce_list.csv", index=False)

    # output bouncing analyze img
    img_opt_Bounce_analysis = Draw_Bounce_Analysis(
        img_opt_Bounce_analysis,
        bounce_location_list,
        miniboard_edge,
        miniboard_height,
        miniboard_width,
        miniboard_text_bias,
    )
    cv2.imwrite(
        f"{analysis_img_path}/{video_name}_analysis.jpg",
        img_opt_Bounce_analysis,
    )
    # For saving bounce map.
    cv2.imwrite(
        f"{bounce_img_path}/{video_name}_bounce.jpg",
        img_opt_bounce_location,
    )

    end = time.time()
    print(f"Write video time: {end-start} seconds.")
    total_time = end - start

    print()
    print(f"Total serve detected : {serve_detection_count}")
    print(f"Total miss catched  : {miss_detection_count}")
    print(f"Detect Result is saved in {video_path}")
    print(f"avg FPS: {np.array(FPS_list).mean()}")
    print(f"Total time: {total_time} seconds")
    print(f"Done......")


if __name__ == "__main__":
    main()
