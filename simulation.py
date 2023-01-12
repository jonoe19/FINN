import pygame
from drone import Drone
# Initialize pygame
pygame.init()

# Set the window size
size = (1000, 1000)
center = (size[0]/2, size[1]/2)
screen = pygame.display.set_mode(size)

# Wall points
wall_p1 = (center[0]+250, 100)
wall_p2 = (center[0]-250, 100)
wall_color = (0, 0, 0)
wall_w = 5

# Check points
point_A = (wall_p1[0], wall_p1[1]+200)
point_B = (wall_p2[0], wall_p2[1]+200)
checkpoint_color = (0, 0, 0)
checkpoint_w = 5

# Set the title of the window
pygame.display.set_caption("Drone simulation")

# Init drone
drone = Drone(point_A, "drone_img.png")

# Run the game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((255, 255, 255))

    # Create_wall & points
    pygame.draw.line(screen, wall_color, wall_p1, wall_p2, wall_w)
    pygame.draw.circle(screen, checkpoint_color, point_A, checkpoint_w, checkpoint_w)
    pygame.draw.circle(screen, checkpoint_color, point_B, checkpoint_w, checkpoint_w)
    drone.draw(screen)
    drone.move(-1, 0)
    # Update the screen
    pygame.display.update()

# Exit pygame
# pygame.quit()
