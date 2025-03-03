import cv2

# 影片路徑
video_path = "/home/chaoen/yoloNhit_calvin/yolov7/inference/videos/C0002_60fps.MP4"

# 打開影片
cap = cv2.VideoCapture(video_path)

# 檢查是否成功打開影片
if not cap.isOpened():
    print("無法開啟影片")
    exit()

skip = 0

# 讀取並顯示影片
while True:
    ret, frame = cap.read()
    skip += 1
    if not ret:
        print("影片播放完畢或讀取錯誤")
        break

    if skip > 2000:
        cv2.imwrite("test_b.jpg", frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
