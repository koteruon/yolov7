import mmap
import struct
import subprocess
import time

import cv2

# 設定環形緩衝區的大小
FRAME_SIZE = 21
TOTAL_BUFFER_SIZE = 1920 * 1080 * (FRAME_SIZE + 1) * 1024  # 40GB (根據需要調整大小)
BUFFER_SIZE = 1920 * 1080 * FRAME_SIZE * 1024
WRITE_POSITION_START = BUFFER_SIZE
WRITE_POSITION_SIZE = 8
FRAME_SIZE = 1920 * 1080 * 3  # 假設視頻流的每幀大小為 1920x1080 像素，bgr24 格式

# 設定 ffmpeg 命令行，將視頻流作為原始視頻數據輸出到標準輸出
ffmpeg_command = [
    "ffmpeg",
    "-stream_loop",
    "-1",
    "-i",
    "/home/chaoen/yoloNhit_calvin/yolov7/inference/videos/C0086.MP4",
    "-map",
    "0:v",
    "-f",
    "rawvideo",
    "-r",
    "60",
    "-c:v",
    "h264_vaapi",  # 啟用硬體加速 H.264 編碼
    "-",
]

# 啟動 ffmpeg 並捕獲其輸出
ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 創建一個環形緩衝區並使用 mmap 映射到內存
buffer = mmap.mmap(-1, TOTAL_BUFFER_SIZE)

# 環形緩衝區的寫入位置
write_position = 0

# 用來計算 FPS 的變數
frame_count = 0
start_time = time.time()

# 開啟 V4L2 設備，這裡使用 cv2.VideoCapture
# video_device = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

# 確認設備是否成功開啟
# if not video_device.isOpened():
#     print("Error: Cannot open video device")
#     exit()

# 設定視頻流的幀率，這裡強制設為 60 FPS
# video_device.set(cv2.CAP_PROP_FPS, 60)


# 寫入位置的 helper 函數
def update_write_position(position):
    buffer[WRITE_POSITION_START : WRITE_POSITION_START + WRITE_POSITION_SIZE] = struct.pack("<Q", position)


# 從緩衝區讀取 write_position
def get_write_position():
    return struct.unpack("<Q", buffer[WRITE_POSITION_START : WRITE_POSITION_START + WRITE_POSITION_SIZE])[0]


try:
    # 記錄初始時間
    start_time = time.time()

    while True:
        # 從 cv2 讀取視頻幀
        # ret, frame = video_device.read()

        # if not ret:
        #     print("Error: Failed to read frame")
        #     break

        # 將幀轉換為原始視頻數據（BGR24 格式）
        # frame_data = frame.tobytes()

        # 從 ffmpeg 的 stdout 中讀取視頻流
        frame_data = ffmpeg_process.stdout.read(FRAME_SIZE)
        if not frame_data:
            print("Error: Failed to read frame")
            break  # 當沒有數據時退出

        # 將數據寫入環形緩衝區
        buffer[write_position : write_position + len(frame_data)] = frame_data

        # 更新寫入位置，確保它不會超過緩衝區的大小
        write_position = (write_position + len(frame_data)) % BUFFER_SIZE

        # 更新 write_position 存儲區
        update_write_position(write_position)
        get_write_position()

        # 計算 FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:  # 每過 1 秒，更新 FPS
            print(f"FPS: {frame_count / elapsed_time:.2f}")
            frame_count = 0  # 重置幀數
            start_time = time.time()  # 重置計時

except KeyboardInterrupt:
    print("Process interrupted")

finally:
    ffmpeg_process.stdout.close()
    ffmpeg_process.stderr.close()
    ffmpeg_process.wait()
    buffer.close()
