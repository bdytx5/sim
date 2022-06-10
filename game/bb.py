import pygame
  
# Initializing Pygame
pygame.init()
  
# Initializing surface
surface = pygame.display.set_mode((400,300))
  
# Initialing Color
color = (255,0,0)
  
# Drawing Rectangle



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

def writeData(filename, x,y, h, w):
    f = open(filename, "a")
    f.write("0 " + x + " " + y + " "+ " " + h + " " + w + "\n")

pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode([500, 500])

# Run until the user asks to quit
sz = 500.0
running = True
x,y = 0.0,0.0
h = 75
w = 75 
bsz = 150.0
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
    # Flip the display
    pygame.draw.rect(surface, color, pygame.Rect(30, 30, 60, 60))

    pygame.display.flip()

# Done! Time to quit.
pygame.quit()



