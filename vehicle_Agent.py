import math
import random
import numpy as np
from pyglet import shapes

class VehicleAgent:
    def __init__(self, x, y, width, height, color, batch1, batch2, isControlled = False):
        self.isControlled = isControlled
        self.shape = shapes.Rectangle(x=x, y=y, width=width, height=height, color=color, batch=batch1)
        self.width = width
        self.height = height
        self.turning_anchor_x = width // 4 #anchor for turning
        self.turning_anchor_y = height // 2 
        self.center_anchor_x = width // 2 #anchor for center
        self.center_anchor_y = height // 2
        self.deg_angle = 0
        self.car_angle = math.radians(self.deg_angle)

        self.vision_length = 2*width
        self.shape.anchor_position = self.turning_anchor_x, self.turning_anchor_y
        self.front_vehicle = shapes.Line(x, y, x + math.cos(self.car_angle) * self.vision_length, y + math.sin(self.car_angle) * self.vision_length, width=1, color=(255, 0, 0), batch=batch2)
        self.left_vehicle = shapes.Line(x, y, x + math.cos(self.car_angle + math.radians(45)) * self.vision_length, y + math.sin(self.car_angle + math.radians(45)) * self.vision_length, width=1, color=(255, 0, 0), batch=batch2)
        self.right_vehicle = shapes.Line(x, y, x + math.cos(self.car_angle - math.radians(45)) * self.vision_length, y + math.sin(self.car_angle - math.radians(45)) * self.vision_length, width=1, color=(255, 0, 0), batch=batch2)

        self.velocity = 0
        self.turning_speed = 0 #degree of turning for vehicle

        self.current_direction = [0, 0, 0, 0]  

    def getDirection(self):
        if not self.isControlled:
            return [random.choice([0, 1]) for _ in range(4)]

    def updateDirection(self): #vehicle only changes its desired direction after a chosen number of timesteps (longer than native refresh rate)
        self.current_direction = self.getDirection()

    def changeAnchor(self, anchor_x, anchor_y):
        # Calculate the difference in anchor position
        dx = anchor_x - self.shape.anchor_x
        dy = anchor_y - self.shape.anchor_y
        rotated_dx = dx * math.cos(self.car_angle) - dy * math.sin(self.car_angle)
        rotated_dy = dx * math.sin(self.car_angle) + dy * math.cos(self.car_angle)
        # Update the shape's position to account for the new anchor
        self.shape.x += rotated_dx
        self.shape.y += rotated_dy
        # Update the anchor position
        self.shape.anchor_x = anchor_x
        self.shape.anchor_y = anchor_y


class RoadTile:
    def __init__(self, start_x, start_y, end_x, end_y, width, color, batch):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.width = width
        self.color = color
        self.batch = batch
        self.road_line = shapes.Line(start_x, start_y, end_x, end_y, width, color, batch=batch)

    def is_on_road(self, object): #check whether the ANCHOR of a shape is inside the road, if yes then return true


        # def point_to_segment_distance(px, py, x1, y1, x2, y2):
        #     line_vec = np.array([x2 - x1, y2 - y1])
        #     point_vec = np.array([px - x1, py - y1])
        #     line_len = np.linalg.norm(line_vec)
        #     line_unitvec = line_vec / line_len
        #     point_vec_scaled = point_vec / line_len
        #     t = np.dot(line_unitvec, point_vec_scaled)
        #     t = np.clip(t, 0, 1)
        #     nearest = np.array([x1, y1]) + t * line_vec
        #     distance = np.linalg.norm(np.array([px, py]) - nearest)
        #     return distance
        
        # distance = point_to_segment_distance(object.shape.x, object.shape.y, self.start_x, self.start_y, self.end_x, self.end_y)
        # return distance <= self.width / 2 #distance from object to road line
        return object in self.road_line




