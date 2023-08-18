import argparse
import os
import shutil
import time
from glob import glob


def main(is_train=False):

    results_root = r'./runs/detect/'
    results_list = list(glob(os.path.join(results_root, 'hit*')))

    target_root = r'/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/'
    video_dir = os.path.join(target_root, 'videos/test/')
    if is_train:
        lbl_dir = os.path.join(target_root, 'train/')
    else:
        lbl_dir = os.path.join(target_root, 'test/')

    for root_idx, results_dir in enumerate(results_list):
        video_name = [v for v in os.listdir(results_dir) if v != 'labels']
        if video_name != []:
            video_name = video_name[0]
        else:
            print(results_dir)
            continue
        video_basename = video_name.split('.')[0]

        ## 搬移影片
        if os.path.exists(os.path.join(video_dir, video_name)):
            os.remove(os.path.join(video_dir, video_name))
        shutil.move(os.path.join(results_dir, video_name), os.path.join(video_dir, video_name))
        ## 搬移標籤
        if os.path.exists(os.path.join(lbl_dir, video_basename)):
            shutil.rmtree(os.path.join(lbl_dir, video_basename))
        shutil.move(os.path.join(results_dir, 'labels'), os.path.join(lbl_dir, video_basename))
        ## 刪除原資料夾
        os.rmdir(results_dir)

    return ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Moving yolov7 results to hit dataset")
    parser.add_argument(
        "--train",
        help="Build training dataset",
        action='store_true',
    )

    args = parser.parse_args()

    t1 = time.time()
    main(is_train=args.train)
    t2 = time.time()
    print("Total Time :", t2-t1)