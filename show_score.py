import cv2
import numpy as np

# 建立 16:9 畫布
width, height = 1280, 720
image_CV = np.zeros((height, width, 3), dtype=np.uint8)

# 初始化 freetype
freetype = cv2.freetype.createFreeType2()
freetype.loadFontData(fontFileName="ttf/MSJH.TTC", id=0)

# 分數與樣式
score = [4, 3]
text = f"{score[0]} : {score[1]}"
font_height = 300
color = (0, 255, 255)
thickness = -1

# 取得文字大小
(text_width, text_height), _ = freetype.getTextSize(text, font_height, thickness)

# 正中央座標修正
x = 350
y = 150  # baseline 位置

# 畫上文字
freetype.putText(image_CV, text, (x, y), font_height, color, thickness, cv2.LINE_AA, False)

# 顯示畫面
cv2.imshow("Scoreboard", image_CV)
cv2.waitKey(0)
cv2.destroyAllWindows()
