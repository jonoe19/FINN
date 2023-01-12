import pygame

class Drone():
    def __init__(self, center, drone_path) -> None:
        self.direction = (0, 1)
        self.thickness = 5
        drone_img = pygame.image.load(drone_path)
        self.drone_img_size = drone_img.get_size()
        self.drone_img = pygame.transform.scale(drone_img, (self.drone_img_size[0] // 2, self.drone_img_size[1] // 2))
        self.drone_rect = self.drone_img.get_rect(center=center)

    def move(self, x_speed, y_speed) -> None:
        self.drone_rect.x += x_speed
        self.drone_rect.y += y_speed

    def draw(self, screen) -> None:
        screen.blit(self.drone_img, self.drone_rect)

    def rotate(self, angle, screen) -> None:
        self.drone_img = pygame.transform.rotate(self.drone_img, angle)
        self.drone_rect = self.drone_img.get_rect(center=self.drone_rect.center)
        self.draw(screen)
