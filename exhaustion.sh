#!/bin/sh

# 定義數字和 ID 的組合
numbers="301 302 303 304 305 307 308 309"
ids="id11 id12 id13"

# 遍歷所有組合
for num in $numbers; do
    for id in $ids; do
        python detect.py \
            --save-ori-img \
            --weights ./runs/train/exhaustion_20250220/weights/best.pt \
            --conf 0.5 \
            --img-size 960 \
            --source ./inference/videos/exhaustion/${num}_${id}.mp4 \
            --onlyball \
            --save-txt \
            --project runs/detect/ \
            --name exhaustion_${num}_${id}
    done
done