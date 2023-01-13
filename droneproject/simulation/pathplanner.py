from pygame.math import Vector2
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class PathPlanner():
    def __init__(self, wall, distance_2w, distance_2p) -> None:
        self.wall = wall
        self.distance_2w = distance_2w
        self.checkpoints = []
        self.distance_2p = distance_2p

    def get_checkpoints(self):
        temp_checkpoints = []
        for idx, pt in enumerate(self.wall):
            if idx != len(self.wall)-1:
                vector = self.get_vector(pt, self.wall[idx+1])
                perp_vector = self.get_perpendicular(vector)
                p1 = [pt + perp_vector]
                p2 = [self.wall[idx+1]+perp_vector]
                temp_checkpoints = temp_checkpoints + p1 + p2
                if pt == self.wall[0]:
                    self.checkpoints = self.checkpoints + p1
                if pt == self.wall[-2] or len(self.wall) == 2:
                    temp_point = p2

        temp_lines = []
        for idx in range(0, len(temp_checkpoints)-1, 2):
            temp_lines = temp_lines + \
                [self.points_to_line(temp_checkpoints[idx],
                                     temp_checkpoints[idx+1])]

        for idx in range(0, len(temp_lines)-1):
            self.checkpoints = self.checkpoints + \
                [self.line_intercept(temp_lines[idx], temp_lines[idx+1])]
        self.checkpoints = self.checkpoints + temp_point

    def line_intercept(self, line1, line2):
        x = (line2[1] - line1[1]) / (line1[0] - line2[0])
        y = line1[0] * x + line1[1]
        return Vector2(x, y)

    def points_to_line(self, p1, p2):
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        y_intercept = p1[1] - slope * p1[0]
        line = (slope, y_intercept, 0)
        return line

    def get_vector(self, p1: Vector2, p2: Vector2):
        return Vector2(p1[0]-p2[0], p1[1]-p2[1])

    def get_points(self):
        points = []
        vectors = []
        for idx in range(0, len(self.checkpoints)-1):
            p1 = self.checkpoints[idx]
            p2 = self.checkpoints[idx+1]
            distance = p1.distance_to(p2)
            vector = self.get_vector(p2, p1)
            vector.scale_to_length(self.distance_2p)
            vectors = vectors + [vector]
            i = 0
            while i < distance:
                if i == 0:
                    points = points + [p1]
                if i + self.distance_2p > distance:
                    points = points + [p2]
                else:
                    points = points + [vector + points[-1]]
                i += self.distance_2p
        return points  # , vectors

    def get_perpendicular(self, vector: Vector2):
        perp_vector = Vector2(-vector[1], vector[0])
        perp_vector.scale_to_length(self.distance_2w)
        return perp_vector

    def get_parallel(self, vector: Vector2):
        perpendicular_vector = self.get_perpendicular(vector)
        parallel_vector = Vector2(
            perpendicular_vector.x,  perpendicular_vector.y)
        parallel_vector = self.get_perpendicular(perpendicular_vector)
        parallel_vector.scale_to_length(vector.length())
        return parallel_vector
