import os
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

WIDTH = 1920
HEIGHT = 1080

def main():

    raw_root = r'/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/videos/test/'
    video_root = r'/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/videos/yolov7_videos/'
    result_root = r'/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/videos/yolov7_kp_videos/'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load('./weights/yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)

    video_names = os.listdir(video_root)
    video_list = [os.path.join(video_root, v) for v in video_names]
    raw_list = [os.path.join(raw_root, v) for v in video_names]

    with torch.no_grad():

        for v_idx, video_path in enumerate(video_list):
            print(f"Now handling : {video_names[v_idx]}")
            # if video_names[v_idx] != 'M-4.MOV':
            #     continue
            raw_cap = cv2.VideoCapture(raw_list[v_idx])
            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            result_v = cv2.VideoWriter(
                os.path.join(result_root, f'{video_names[v_idx].replace(".MOV",".mp4")}'),
                fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                (WIDTH, HEIGHT),
            )
            progress = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            while cap.isOpened():

                ret, im = cap.read()
                if not ret:
                    break
                raw_ret, raw_im = raw_cap.read()
                if not raw_ret:
                    break

                im, _, (im_dw, im_dh) = letterbox(im, 960, stride=64, auto=True)
                im = transforms.ToTensor()(im)
                im = torch.tensor(np.array([im.numpy()]))

                raw_im = letterbox(raw_im, 960, stride=64, auto=True)[0]
                raw_im = transforms.ToTensor()(raw_im)
                raw_im = torch.tensor(np.array([raw_im.numpy()]))

                if torch.cuda.is_available():
                    raw_im = raw_im.half().to(device)

                output, _ = model(raw_im)
                output = non_max_suppression_kpt(output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                output = np.array(sorted(output, key=lambda x: (x[2], x[3])))

                nimg = im[0].permute(1, 2, 0) * 255
                nimg = nimg.cpu().numpy().astype(np.uint8)
                nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

                for idx in range(output.shape[0]):
                    if 0<= output[idx, 2] <=  960*5/9 \
                        and 0 <= output[idx, 3] <=  960 * 0.25:
                        continue
                    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

                nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
                nimg = nimg[int(im_dh):-int(im_dh),:]
                nimg = cv2.resize(nimg, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

                result_v.write(nimg.astype(np.uint8))

                progress.update(1)

            progress.close()
            cap.release()
            raw_cap.release()
            result_v.release()

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total Time :", t2-t1)