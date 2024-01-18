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


class DrawSkeleton:
    def __init__(self):
        self.WIDTH = 1920
        self.HEIGHT = 1080
        self.raw_root = r"/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/videos/test/"
        self.video_root = r"/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/videos/yolov7_videos/"
        self.result_root = r"/home/chaoen/yoloNhit_calvin/HIT/data/table_tennis/videos/yolov7_kp_videos/"
        self.video_names = os.listdir(self.video_root)
        self.video_list = [os.path.join(self.video_root, v) for v in self.video_names]
        self.raw_list = [os.path.join(self.raw_root, v) for v in self.video_names]

    def load_model(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weigths = torch.load("./weights/yolov7-w6-pose.pt", map_location=self.device)
        self.model = weigths["model"]
        _ = self.model.float().eval()

        if torch.cuda.is_available():
            self.model.half().to(self.device)

    def cap_from_resized_video(self, v_idx, video_path):
        cap = cv2.VideoCapture(video_path)
        progress = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        return cap, progress

    def cap_from_raw_video(self, v_idx, video_path):
        raw_cap = cv2.VideoCapture(self.raw_list[v_idx])
        return raw_cap

    def output_video(self, v_idx, fps):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        result_v = cv2.VideoWriter(
            os.path.join(self.result_root, f'{self.video_names[v_idx].replace(".MOV",".mp4")}'),
            fourcc,
            fps,
            (self.WIDTH, self.HEIGHT),
        )
        return result_v

    def read_image_from_resized_video(self, cap):
        ret, im = cap.read()
        if not ret:
            raise Exception("ret is None")

        im, _, (im_dw, im_dh) = letterbox(im, 960, stride=64, auto=True)
        im = transforms.ToTensor()(im)
        im = torch.tensor(np.array([im.numpy()]))

        return im, im_dw, im_dh

    def read_image_from_raw_video(self, raw_cap):
        raw_ret, raw_im = raw_cap.read()
        if not raw_ret:
            raise Exception("raw_ret is None")

        raw_im = letterbox(raw_im, 960, stride=64, auto=True)[0]
        raw_im = transforms.ToTensor()(raw_im)
        raw_im = torch.tensor(np.array([raw_im.numpy()]))

        if torch.cuda.is_available():
            raw_im = raw_im.half().to(self.device)

        return raw_im

    def model_detect(self, raw_im):
        with torch.no_grad():
            output, _ = self.model(raw_im)
        output = non_max_suppression_kpt(
            output, 0.5, 0.65, nc=self.model.yaml["nc"], nkpt=self.model.yaml["nkpt"], kpt_label=True
        )
        output = output_to_keypoint(output)
        output = np.array(sorted(output, key=lambda x: (x[2], x[3])))
        return output

    def draw_skeletons(self):
        for v_idx, video_path in enumerate(self.video_list):
            print(f"Now handling : {self.video_names[v_idx]}")
            cap, progress = self.cap_from_resized_video(v_idx, video_path)
            raw_cap = self.cap_from_raw_video(v_idx, video_path)
            result_v = self.output_video(v_idx, cap.get(cv2.CAP_PROP_FPS))
            while cap.isOpened():
                try:
                    im, im_dw, im_dh = self.read_image_from_resized_video(cap)
                    raw_im = self.read_image_from_raw_video(raw_cap)
                except:
                    break
                output = self.model_detect(raw_im)
                nimg = self.draw_skeleton(im, im_dw, im_dh, output)
                # write image
                result_v.write(nimg.astype(np.uint8))
                progress.update(1)
            self.release_resources(cap, raw_cap, result_v, progress)

    def draw_skeleton(self, im, im_dw, im_dh, output):
        nimg = im[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

        for idx in range(output.shape[0]):
            if 0 <= output[idx, 2] <= 960 * 5 / 9 and 0 <= output[idx, 3] <= 960 * 0.25:
                continue
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        nimg = nimg[int(im_dh) : -int(im_dh), :]
        nimg = cv2.resize(nimg, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_AREA)

        return nimg

    def release_resources(self, cap, raw_cap, result_v, progress):
        cap.release()
        raw_cap.release()
        result_v.release()
        progress.close()


if __name__ == "__main__":
    t1 = time.time()
    draw_skeleton = DrawSkeleton()
    draw_skeleton.load_model()
    draw_skeleton.draw_skeletons()
    t2 = time.time()
    print("Total Time :", t2 - t1)
