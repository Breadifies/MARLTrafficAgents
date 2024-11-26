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
        self.shape.anchor_position = self.turning_anchor_x, self.turning_anchor_y

        #LINES
        self.maxLen = 2*width
        self.num_vision_lines = 6
        self.Lines = []#6
        self.Angles = [0, 45, -45, 180, 135, 215]#3
        self.lineLengths = [self.maxLen]*self.num_vision_lines
        self.defineLines(x, y, batch2, self.Angles)

        self.velocity = 0
        self.turning_speed = 0 #degree of turning for vehicle
        self.current_direction = [0, 0, 0, 0]  

    def getDirection(self, action):
        if not self.isControlled:
            if action == 0:
                return [0, 0, 0, 0]
            elif action == 1:
                return [1, 0, 0, 0]
            elif action == 2:
                return [0, 1, 0, 0]
            elif action == 3:
                return [0, 0, 1, 0]
            elif action == 4:
                return [0, 0, 0, 1]
            else:
                print("Invalid action")
                return [0, 0, 0, 0]

        
    def updateDirection(self, action): #vehicle only changes its desired direction after a chosen number of timesteps (longer than native refresh rate)
        self.current_direction = self.getDirection(action)

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

    def defineLines(self, x, y, batch, angles):
        for i in range(self.num_vision_lines):
            line = shapes.Line(x, y, x + math.cos(self.car_angle + math.radians(angles[i])) * self.maxLen, y + math.sin(self.car_angle + math.radians(angles[i])) * self.maxLen, width=1, color=(255, 0, 0), batch=batch)
            self.Lines.append(line)

    def updateLines(self):
        for i in range(self.num_vision_lines):
            self.Lines[i].x = self.shape.x
            self.Lines[i].y = self.shape.y
            self.Lines[i].x2 = self.shape.x + math.cos(self.car_angle + math.radians(self.Angles[i])) * self.lineLengths[i]
            self.Lines[i].y2 = self.shape.y + math.sin(self.car_angle + math.radians(self.Angles[i])) * self.lineLengths[i]
    


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

        #checkpoints for vehicle to travel on track
        # self.checkpoints = []
        # num_checkpoints = 20
        # checkpoint_width = 10
        # checkpoint_height = width
        # interval = (end_x - start_x) / (num_checkpoints + 1)
        # for i in range(1, num_checkpoints + 1):
        #     checkpoint_x = start_x + i * interval
        #     checkpoint_y = start_y
        #     checkpoint = shapes.Rectangle(checkpoint_x - checkpoint_width / 2, checkpoint_y - checkpoint_height / 2, checkpoint_width, checkpoint_height, color=(0, 255, 0), batch=batch)
        #     self.checkpoints.append(checkpoint)


    def is_on_road(self, object_pos): #check whether the ANCHOR of a shape is inside the road, if yes then return true
        return object_pos in self.road_line
    
    def line_end_on_road(self, x2, y2):
        return self.is_on_road((x2, y2))#if ends of vision line NOT in contact with road

    # def is_in_checkpoint(self, object_pos):
    #     # Check if the object position is within the checkpoint rectangle
    #     for checkpoint in self.checkpoints:
    #         if (checkpoint.x <= object_pos[0] <= checkpoint.x + checkpoint.width and
    #             checkpoint.y <= object_pos[1] <= checkpoint.y + checkpoint.height):
    #             return True
    #     return False
            




