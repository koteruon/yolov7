import cv2
import numpy as np


def compare_frames(frame1, frame2):
    """
    比較兩個frame是否完全相同
    """
    if frame1.shape != frame2.shape:
        return False
    return np.array_equal(frame1, frame2)


def count_matched_frames(video_a_path, video_b_path):
    """
    計算B影片在A影片中成功配對的frame數量
    """
    # 開啟兩個影片
    cap_a = cv2.VideoCapture(video_a_path)
    cap_b = cv2.VideoCapture(video_b_path)

    if not cap_a.isOpened() or not cap_b.isOpened():
        print("Error: 無法開啟影片檔案")
        return None

    matched_count = 0  # 成功配對的frame計數
    start_frame_a = None  # 記錄開始配對的A影片frame位置
    end_frame_a = None  # 記錄結束配對的A影片frame位置

    # 讀取B的第一個frame
    ret_b, frame_b = cap_b.read()
    if not ret_b:
        print("Error: B影片是空的")
        return None

    # 開始在A中尋找匹配
    frame_count_a = 0
    while True:
        ret_a, frame_a = cap_a.read()
        if not ret_a:
            break

        if compare_frames(frame_a, frame_b):
            # 找到第一個匹配，記錄A的位置
            if start_frame_a is None:
                start_frame_a = frame_count_a
                matched_count += 1

            # 讀取B的下一個frame
            while True:
                ret_b, next_frame_b = cap_b.read()
                if not ret_b:  # B已經讀完
                    end_frame_a = frame_count_a
                    break

                # 檢查B是否有重複frame
                if not compare_frames(next_frame_b, frame_b):
                    break  # B重複，保持A不動，讀取B的下一個

            frame_b = next_frame_b  # 更新B的current frame

        frame_count_a += 1

    # 計算區間內的總frame數
    total_frames_in_range = (end_frame_a - start_frame_a) if start_frame_a is not None else 0

    # 釋放資源
    cap_a.release()
    cap_b.release()

    # 回傳結果
    return {
        "matched_frames": matched_count,
        "start_frame_in_a": start_frame_a,
        "end_frame_in_a": end_frame_a,
        "total_frames_in_range": total_frames_in_range,  # 新增：區間內的總frame數
    }


def main():
    # 使用範例
    video_a_path = r"inference/videos/C0002_60fps.MP4"  # 標準答案影片路徑
    video_b_path = r"runs/detect/realtime34/Realtime2/Realtime.mp4"  # 擷取的影片路徑

    result = count_matched_frames(video_a_path, video_b_path)

    if result:
        print(f"成功配對的frame數量: {result['matched_frames']}")
        print(f"在A影片中的起始frame: {result['start_frame_in_a']}")
        print(f"在A影片中的結束frame: {result['end_frame_in_a']}")
        print(f"A影片在此區間的總frame數: {result['total_frames_in_range']}")  # 新增的輸出


if __name__ == "__main__":
    main()
