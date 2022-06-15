# Simple pygame program
# Simple pygame program
# Simple pygame program
# Simple pygame program
# Simple pygame program

# Import and initialize the pygame library
from asyncore import write
from tokenize import Double
import pygame
import io 
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt



def move_coords(angle, radius, coords):
    theta = math.radians(angle)
    return coords[0] + radius * math.cos(theta), coords[1] + radius * math.sin(theta)


def writeData(filename, x,y, h, w):
    f = open(filename, "a")
    f.write("0 " + x + " " + y + " "+ " " + h + " " + w + "\n")

pygame.init()
color = (255,0,0)
# Set up the drawing window
screen = pygame.display.set_mode([500, 500])

# Run until the user asks to quit
sz = 500.0
running = True
x,y = 0.0,0.0
h = 75
w = 75 
bsz = 75.0
boxSizeScaled = bsz/sz
boxSzStr = f'{boxSizeScaled:.6f}'
##### TODO need to adjust box sizes to the full circle size 

cnt = 0 

while running:
    flnm = str(cnt)
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill((255, 255, 255))

    # Draw a solid blue circle in the center

    pygame.draw.circle(screen, (0, 0, 255), (x, y), w)
    # ensure circle is in the frame fully 
    l = x-75.0
    r = x+75.0
    b = y+75.0
    t = y-75.0


    #### say our detection engine has some noise 
    ### its usually has 90% overlap with our target (10% error)
    ### total area = 150*150 
    ### so multiply offset noise times 15 on both top and left postion !!!
    noise = np.random.normal(0,1,1) # mean = 0, std = 1
    noisyL = l + (15*noise)
    noise = np.random.normal(0,1,1) # mean = 0, std = 1
    noisyT = t + (15*noise)
    pygame.draw.rect(screen, color, pygame.Rect(noisyL, noisyT, 150, 150), 2) ## goal is to smooth this track with a kalman filter 


    if l > 0 and r > 0 and l < 500 and r < 500 and t > 0 and b > 0 and t < 500 and b < 500: 
        xs = f'{x/sz:.6f}'
        ys = f'{y/sz:.6f}'

        x3 = pygame.surfarray.pixels3d(screen)
        x3 = x3[:,:,::-1]
        image_data = x3 / 255.
        image_data = x3[np.newaxis, ...].astype(np.float32)
        cv2.imshow('image',x3)
        cv2.waitKey(0)

        # pygame.image.save(screen,"images/"+flnm+".jpg")
        # writeData("annotations/"+flnm+".txt", xs, ys,boxSzStr,boxSzStr)    
        cnt+=1 
        
    if y == 500:
        x+=10 
        y = 0
        if x > 500:
            break 
        continue
    if y < 500:
        y +=1 
    
    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()



