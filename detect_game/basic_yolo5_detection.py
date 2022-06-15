import os, sys


import pygame
import math
import numpy as np

import cv2
import torch
from PIL import Image


# import parent directory 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# The returned object is a 2-D array. The output depends on the size of the input. For example, with 
# the default input size 640, we get a 2D-array of size 25200×85 (rows and columns). 
# The rows represent the number of detections. So each time the network runs, it predicts 25200 bounding boxes. 
# Every bounding box has a 1-D array of 85 entries that tells the quality of the detection. 
# This information is enough to filter out the desired detections.

color = (255,255,255)
yllw = (255,255,0)

# model = torch.load(f='/Users/brett/Desktop/sim/game/models/best.pt', map_location='cpu') 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/brett/Desktop/sim/game/models/best.pt', force_reload=True) 


def main():

    pygame.display.set_caption("3DSPORTS SIM")
    screen = pygame.display.set_mode((640, 640))
    clock = pygame.time.Clock()
    coords = 100, 100
    rect = pygame.Rect(*coords,20,20)

     
    running = True

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
         
        screen.fill((0,0,30))
        screen.fill((0,150,0), rect)
        pygame.image.save(screen,"tst.jpg")
        im1 = Image.open('tst.jpg')  # PIL image


        pygame.display.flip()
        clock.tick(30)

        # inference 
        x3 = pygame.surfarray.pixels3d(screen)
        # x3 = x3[:,:,::-1]
        # x3n = x3 / 255.

        results = model(x3)  # includes NMS

        print(results.pandas().xyxy[0]['xmin'] )

        # results.show()

        # pred = non_max_suppression(pred, 0.25, 0.25, 0, False, max_det=10)
        # The output is [xywh, conf, class0, class1, ...]

        # labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()




        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        # for i, det in enumerate(pred):  # per image
        #     seen += 1

        #     # p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        #     annotator = Annotator(im0, line_width=5.0, example=str(names))


        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        #         # Print results
        #         # for c in det[:, -1].unique():
        #         #     n = (det[:, -1] == c).sum()  # detections per class
        #         #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        #         # # Write results
        #         # for *xyxy, conf, cls in reversed(det):
        #         #     if save_txt:  # Write to file
        #         #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        #         #         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        #         #         with open(f'{txt_path}.txt', 'a') as f:
        #         #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

        #         #     if save_img or save_crop or view_img:  # Add bbox to image
        #         #         c = int(cls)  # integer class
        #         #         label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
        #         #         annotator.box_label(xyxy, label, color=colors(c, True))
        #         #     if save_crop:
        #         #         save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        #     # Stream results
        #     im0 = annotator.result()
 
        #     cv2.imshow(str(p), im0)
        #     cv2.waitKey(1)  # 1 millisecond

     
    pygame.quit()
 
if __name__ == '__main__':
    main()


