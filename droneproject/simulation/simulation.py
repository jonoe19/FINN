from pygame.math import Vector2
from pathplanner import PathPlanner
from drone import Drone
import pygame
import time
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

pygame.init()

size = (1000, 1000)
center = Vector2(size[0]/2, size[1]/2)
screen = pygame.display.set_mode(size)

walls = [
    Vector2(800, 100),
    Vector2(400, 200),
    Vector2(200, 300),
    Vector2(100, 700),
    Vector2(300, 800),
    Vector2(600, 600),
    Vector2(800, 800)
]

wall = 0
wall_color = (0, 0, 0)
wall_w = 5

checkpoint_color = (100, 100, 100)
checkpoint_w = 5
point_color = (200, 100, 100)
point_w = 2

pygame.display.set_caption("Drone simulation")
path_planner = PathPlanner(walls, 100, 50)
path_planner.get_checkpoints()
checkpoints = path_planner.checkpoints
points = path_planner.get_points()
drone = Drone(checkpoints[0], "drone.png")
angle = 0
w, h = drone.drone_img.get_size()

running = True
step = 0
wait1 = True
wait2 = True
wait3 = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    

    for idx, _wall in enumerate(walls):
        if _wall != walls[-1]:
            pygame.draw.line(screen, wall_color, _wall, walls[idx+1], wall_w)

    if wait1:
        pygame.display.update()
        time.sleep(1)
        wait1 = False
    for checkpoint in checkpoints:
        pygame.draw.circle(screen, checkpoint_color,
                           checkpoint, checkpoint_w, checkpoint_w)
    if wait2:
        pygame.display.update()
        time.sleep(1)
        wait2 = False
    for point in points:
        pygame.draw.circle(screen, point_color, point, point_w, point_w)
    if wait3:
        pygame.display.update()
        time.sleep(1)
        wait3 = False
    if step < len(points):
        drone.move(points[step])
    else:
        running = False
    drone.draw(screen)

    step += 1
    time.sleep(0.5)
    pygame.display.update()

pygame.quit()
