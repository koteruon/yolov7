import mmap
import time

import numpy as np

# 設定環形緩衝區的大小
BUFFER_SIZE = 1920 * 1080 * 21 * 1024  # 40GB (根據需要調整大小)
FRAME_SIZE = 1920 * 1080 * 3  # 假設視頻流的每幀大小為 1920x1080 像素，bgr24 格式

# 創建一個映射到相同區域的 mmap
buffer = mmap.mmap(-1, BUFFER_SIZE, access=mmap.ACCESS_READ)

# 用來計算 FPS 的變數
frame_count = 0
start_time = time.time()

# 設定圖像的寬高
frame_width = 1920
frame_height = 1080

# 設定幀率 (60 FPS)
frame_delay = 1 / 60

try:
    while True:
        # 根據環形緩衝區的位置讀取數據
        # 計算目前幀的位置
        current_position = (frame_count * FRAME_SIZE) % BUFFER_SIZE

        # 從 mmap 讀取幀數據
        frame_data = buffer[current_position : current_position + FRAME_SIZE]

        if not frame_data:
            # 如果沒有數據，休眠 1/60 秒
            time.sleep(frame_delay)
            continue  # 繼續等待下一幀

        # 將原始數據轉換為 NumPy 陣列，並重塑為 OpenCV 可以處理的形狀
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((frame_height, frame_width, 3))

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
    buffer.close()
