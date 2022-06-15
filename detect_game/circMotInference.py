# interpreter stuff 
import tensorflow as tf
import cv2

interpreter = tf.lite.Interpreter(model_path="/Users/brett/Desktop/sim/game/models/best-fp16.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)



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

        x3 = pygame.surfarray.pixels3d(screen)
        x3 = x3[:,:,::-1]
        image_data = x3 / 255.
        image_data = x3[np.newaxis, ...].astype(np.float32)
        # cv2.imshow('image',x3)
        # cv2.waitKey(0)
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        print(pred)
        boxes, pred_conf = pred[1], pred[0]







        pygame.display.flip()
        clock.tick(30)
     
    pygame.quit()
 
if __name__ == '__main__':
    main()