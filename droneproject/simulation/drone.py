import pygame
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class Drone():
    def __init__(self, center, drone_path) -> None:
        drone_img = pygame.image.load(drone_path)
        self.drone_img_size = drone_img.get_size()
        self.drone_img = pygame.transform.scale(
            drone_img, (self.drone_img_size[0] // 3, self.drone_img_size[1] // 3))
        self.drone_rect = self.drone_img.get_rect(center=center)

    def move(self, point) -> None:
        self.drone_rect.center = point

    def draw(self, screen) -> None:
        screen.blit(self.drone_img, self.drone_rect)
