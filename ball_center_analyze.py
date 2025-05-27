import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === 基本參數 ===
FPS = 60
REAL_DISTANCE_CM = 274.0
x1, y1 = 513, 599
x2, y2 = 1432, 591

# === Pixel → cm 轉換 ===
pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
cm_per_pixel = REAL_DISTANCE_CM / pixel_distance

# === 讀取 CSV ===
csv_path = "runs/detect/105_01_20250526/ball_center.csv"
df = pd.read_csv(csv_path)
df = df.sort_values("frame").reset_index(drop=True)

# === 計算速度與加速度（像素/frame）===
df["vx_pix"] = df["ball_center_x"].diff() / df["frame"].diff()
df["vy_pix"] = df["ball_center_y"].diff() / df["frame"].diff()
df["ax_pix"] = df["vx_pix"].diff() / df["frame"].diff()
df["ay_pix"] = df["vy_pix"].diff() / df["frame"].diff()

# === 轉換為實際單位 ===
df["vx_cm"] = df["vx_pix"] * cm_per_pixel * FPS
df["vy_cm"] = df["vy_pix"] * cm_per_pixel * FPS
df["ax_cm"] = df["ax_pix"] * cm_per_pixel * (FPS**2)
df["ay_cm"] = df["ay_pix"] * cm_per_pixel * (FPS**2)


# === 排除異常值 (IQR法) ===
def remove_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series.where((series >= q1 - 1.5 * iqr) & (series <= q3 + 1.5 * iqr))


# df["vx_cm_clean"] = remove_outliers(df["vx_cm"])
# df["vy_cm_clean"] = remove_outliers(df["vy_cm"])
# df["ax_cm_clean"] = remove_outliers(df["ax_cm"])
# df["ay_cm_clean"] = remove_outliers(df["ay_cm"])

df["vx_cm_clean"] = df["vx_cm"]
df["vy_cm_clean"] = df["vy_cm"]
df["ax_cm_clean"] = df["ax_cm"]
df["ay_cm_clean"] = df["ay_cm"]

# === 建立輸出資料夾 ===
output_dir = "ball_center"
os.makedirs(output_dir, exist_ok=True)

# === 繪圖 ===

# 位置圖 - X
plt.figure(figsize=(8, 5))
plt.plot(df["frame"], df["ball_center_x"], label="X Position (px)")
plt.title("Ball X Position over Time")
plt.xlabel("Frame")
plt.ylabel("X Position (pixels)")
plt.grid()
plt.savefig(os.path.join(output_dir, "position_x.png"))
plt.close()

# 位置圖 - Y
plt.figure(figsize=(8, 5))
plt.plot(df["frame"], df["ball_center_y"], label="Y Position (px)", color="orange")
plt.title("Ball Y Position over Time")
plt.xlabel("Frame")
plt.ylabel("Y Position (pixels)")
plt.grid()
plt.savefig(os.path.join(output_dir, "position_y.png"))
plt.close()

# 速度圖 - X
plt.figure(figsize=(8, 5))
plt.plot(df["frame"], df["vx_cm_clean"], label="Velocity X (cm/s)", color="blue")
plt.title("Cleaned Velocity in X")
plt.xlabel("Frame")
plt.ylabel("Velocity X (cm/s)")
plt.grid()
plt.savefig(os.path.join(output_dir, "velocity_x.png"))
plt.close()

# 速度圖 - Y
plt.figure(figsize=(8, 5))
plt.plot(df["frame"], df["vy_cm_clean"], label="Velocity Y (cm/s)", color="green")
plt.title("Cleaned Velocity in Y")
plt.xlabel("Frame")
plt.ylabel("Velocity Y (cm/s)")
plt.grid()
plt.savefig(os.path.join(output_dir, "velocity_y.png"))
plt.close()

# 加速度圖 - X
plt.figure(figsize=(8, 5))
plt.plot(df["frame"], df["ax_cm_clean"], label="Acceleration X (cm/s²)", color="red")
plt.title("Cleaned Acceleration in X")
plt.xlabel("Frame")
plt.ylabel("Acceleration X (cm/s²)")
plt.grid()
plt.savefig(os.path.join(output_dir, "acceleration_x.png"))
plt.close()

# 加速度圖 - Y
plt.figure(figsize=(8, 5))
plt.plot(df["frame"], df["ay_cm_clean"], label="Acceleration Y (cm/s²)", color="purple")
plt.title("Cleaned Acceleration in Y")
plt.xlabel("Frame")
plt.ylabel("Acceleration Y (cm/s²)")
plt.grid()
plt.savefig(os.path.join(output_dir, "acceleration_y.png"))
plt.close()
