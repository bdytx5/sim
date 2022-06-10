import pygame
# YOLO V5 format is label (int), x center, y center, width, height   ---- (all values normalized). class indexes are zero indexed 

def saveLabels(filename, x,y, windowSize, objectRadius):
    
    l = x-objectRadius
    r = x+objectRadius
    b = y+objectRadius
    t = y-objectRadius

    hs = ws = (objectRadius * 2) / windowSize


    if l > 0 and r > 0 and l < windowSize and r < windowSize and t > 0 and b > 0 and t < windowSize and b < windowSize: 
        
        xs = f'{x/windowSize:.6f}'
        ys = f'{y/windowSize:.6f}'

        f = open(filename, "a")
        f.write("0 " + str(xs) + " " + str(ys) + " "+ " " + str(hs) + " " + str(ws) + "\n")


def saveImg(path,screen):
    pygame.image.save(screen,path)





