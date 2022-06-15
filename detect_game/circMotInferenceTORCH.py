# interpreter stuff 
# MacOS Intel CPU Results (CoreML-capable)
# iMac (Retina 5K, 27-inch, 2020) - 3.8 GHz 8-Core Intel Core i7

# benchmarks: weights=yolov5s.pt, imgsz=640, batch_size=1, data=/Users/glennjocher/PycharmProjects/yolov5/data/coco128.yaml, device=, half=False, test=False, pt_only=False
# Checking setup...
# YOLOv5 🚀 v6.1-171-gb4f7fc5 torch 1.10.1 CPU
# Setup complete ✅ (16 CPUs, 32.0 GB RAM, 213.4/465.6 GB disk)

# Benchmarks complete (276.40s)
#                    Format  mAP@0.5:0.95  Inference time (ms)
# 0                 PyTorch        0.4623               222.37
# 1             TorchScript        0.4623               231.01
# 2                    ONNX        0.4623                54.41
############# 3                OpenVINO        0.4623                40.74
# 4                TensorRT           NaN                  NaN
# 5                  CoreML        0.4620                39.09
# 6   TensorFlow SavedModel        0.4623               153.32
# 7     TensorFlow GraphDef        0.4623               148.00
# 8         TensorFlow Lite        0.4623               162.60
# 9     TensorFlow Edge TPU           NaN                  NaN
# 10          TensorFlow.js           NaN                  NaN


## Openvino seems like a good option https://github.com/openvinotoolkit/openvino#tutorials
## for right now, am just gonna focus on the actual models and stuff, and then just 
## run in on my lambda lab for good performance 




import torch 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/brett/Desktop/sim/game/models/best.pt', force_reload=True) 

import pygame
import math
import numpy as np
import labeleWriter

def move_coords(angle, radius, coords):
    theta = math.radians(angle)
    return coords[0] + radius * math.cos(theta), coords[1] + radius * math.sin(theta)



color = (255,255,255)
yllw = (255,255,0)

def main():

    pygame.display.set_caption("3DSPORTS SIM")
    screen = pygame.display.set_mode((640, 640))
    clock = pygame.time.Clock()
    
    track = []
    nstrack = []

    coords = 100, 100
    angle = 0
    rect = pygame.Rect(*coords,20,20)
    speed = 20
    next_tick = 500
     
    running = True
    cnt = 0 

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
         
        ticks = pygame.time.get_ticks() 
        track.append(coords)

        if len(track) > 40:
            track.pop(0)
            nstrack.pop(0)


        if ticks > next_tick:
            next_tick += speed
            angle += 1
            coords = move_coords(angle, 1, coords)
            rect.center = coords
             
        screen.fill((0,0,30))
        screen.fill((0,150,0), rect)

        if len(track) > 5:
 #            pygame.draw.lines(screen,color,False, track, 1)
 #           pygame.draw.lines(screen,yllw,False, nstrack, 1)
            pass ## commented out for data generation 

        boxL = coords[0]-15
        boxT  = coords[1]-15
        noise = np.random.normal(0,1,1) # mean = 0, std = 1
        boxLNoisy = boxL + (3*noise)
        noise = np.random.normal(0,1,1) # mean = 0, std = 1
        boxTNoisy = boxT + (3*noise)
        nstrack.append((boxLNoisy+7.5, boxTNoisy+7.5))

        #pygame.draw.rect(screen, color, pygame.Rect(boxLNoisy, boxTNoisy, 30, 30), 2) ## goal is to smooth this track with a kalman filter 

        # inference 
        x3 = pygame.surfarray.pixels3d(screen)
        x3 = x3[:,:,::-1]
        # image_data = x3 / 255.
        # image_data = x3[np.newaxis, ...].astype(np.float32)
        # cv2.imshow('image',x3)
        # cv2.waitKey(0)
        results = model(x3)

        # Results
        results.print() 






        pygame.display.flip()
        clock.tick(30)
     
    pygame.quit()
 
if __name__ == '__main__':
    main()