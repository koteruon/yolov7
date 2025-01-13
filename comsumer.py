import cv2

# 指定FIFO文件的路徑
fifo_path = "/home/chaoen/my_fifo"

# 打開FIFO作為視頻流源
cap = cv2.VideoCapture(f"pipe:{fifo_path}")

if not cap.isOpened():
    print("無法打開FIFO視頻流")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取視頻幀，退出")
        continue

    # 顯示每一幀
    # cv2.imshow("Video Stream", frame)

    # 按 'q' 鍵退出
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break
    print("read")

# 釋放資源
cap.release()
cv2.destroyAllWindows()
