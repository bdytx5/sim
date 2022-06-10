from asyncore import write
from tokenize import Double
import pygame
import io 
import numpy as np
import math


def saveLabels(filename, x,y, h, w):
    f = open(filename, "a")
    f.write("0 " + x + " " + y + " "+ " " + h + " " + w + "\n")


def saveImg(img,path,screen, ):
    pygame.image.save(screen,[path])
