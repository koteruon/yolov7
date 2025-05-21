import os
import re
from glob import glob

import cv2
from tqdm import tqdm

# 設定路徑
video_dir = "./draw"
output_dir = "./draw_output"
os.makedirs(output_dir, exist_ok=True)

# 匹配影片名稱的格式
pattern = re.compile(r"(\d+)_\d+_(left|right|unknown)\.mp4", re.IGNORECASE)

# 分類儲存
video_groups = {"left": [], "right": [], "unknown": [], "combine": []}

# 將影片分類
for filepath in glob(os.path.join(video_dir, "*.mp4")):
    filename = os.path.basename(filepath)
    match = pattern.match(filename)
    if match:
        prefix = int(match.group(1))
        label = match.group(2).lower()
        video_groups[label].append((prefix, filepath))
        video_groups["combine"].append((prefix, filepath))

# 按照 prefix 排序
for key in video_groups:
    video_groups[key] = sorted(video_groups[key], key=lambda x: x[0])


# 輔助函數：拼接影片
def concat_videos(video_list, output_path):
    if not video_list:
        print(f"[警告] {output_path} 無影片可處理")
        return

    # 讀取第一支影片資訊
    cap = cv2.VideoCapture(video_list[0][1])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 初始化寫入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _, video_path in tqdm(video_list, desc=f"Processing {os.path.basename(output_path)}"):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
    out.release()
    print(f"[完成] 輸出：{output_path}")


# 執行拼接
concat_videos(video_groups["left"], os.path.join(output_dir, "slow_motion_left.mp4"))
concat_videos(video_groups["right"], os.path.join(output_dir, "slow_motion_right.mp4"))
concat_videos(video_groups["unknown"], os.path.join(output_dir, "slow_motion_unknown.mp4"))
concat_videos(video_groups["combine"], os.path.join(output_dir, "slow_motion_combine.mp4"))
