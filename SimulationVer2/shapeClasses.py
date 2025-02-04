import math
import random
import numpy as np
from pyglet import shapes

class VehicleAgent:
    def __init__(self, x, y, width, height, color, batch1, batch2, isControlled = False):
        self.shape = shapes.Rectangle(x=x, y=y, width=width, height=height, color=color, batch=batch1)
        self.turn_anch_y = height // 2 #center at all times
        self.turn_anch_x = width // 4
        self.followF_anch_x = width / 1.5
        self.followB_anch_x = 2*self.turn_anch_x - self.followF_anch_x #backwards driving (center of rotation outside shape)
        self.deg_angle = 0
        self.shape.anchor_position = self.turn_anch_x, self.turn_anch_y

        #vehicle movement properties
        self.velocity = 0
        self.turning_speed = 0
        self.isControlled = isControlled
        self.current_direction = [0, 0] #input direction

        #vehicle vision line properties
        self.maxLen = 1.5*width
        self.num_vision_lines = 0
        self.Lines = []
        self.Angles = [] #[0, 45, -45, 180, 135, 215]
        self.lineLengths = [self.maxLen]*self.num_vision_lines
        self.defineLines(x, y, batch2, self.Angles)

        #vehicle follow road properties
        self.curRoad = None #road segment currently on
        self.increment = 0 #rotation search for line
        self.clockwise = 1 
    
    def updateDirection(self, action): #change input direction after a certain no. timesteps
        self.current_direction = self.getDirection(action)

    def getDirection(self, action):
        if action == 0:
            return [0, 0]
        elif action == 1:
            return [1, 0]
        elif action == 2:
            return [0, 1]
        else:
            print("Invalid action")
            return [0, 0]
    
    def changeAnchor(self, anch_x, anch_y): #manually reconfigure pyglet shape anchor
        dx = anch_x - self.shape.anchor_x
        dy = anch_y - self.shape.anchor_y
        car_angle = math.radians(self.deg_angle)
        rotated_dx = dx*math.cos(car_angle) - dy*math.sin(car_angle)
        rotated_dy = dx*math.sin(car_angle) + dy*math.cos(car_angle)
        #update shape's position to account for new anchor
        self.shape.x += rotated_dx
        self.shape.y += rotated_dy 
        #update anchor position
        self.shape.anchor_position = (anch_x, anch_y)

    def defineLines(self, x, y, batch, angles): #create vehicle vision lines
        for i in range(self.num_vision_lines):
            car_angle = math.radians(self.deg_angle)
            line = shapes.Line(x, y, x + math.cos(car_angle + math.radians(angles[i])) * self.maxLen, y + math.sin(car_angle) + math.radians(angles[i]) * self.maxLen, width=1, color=(255, 0, 0), batch=batch)
            self.Lines.append(line)
    
    def updateLines(self):
        for i in range(self.num_vision_lines):
            car_angle = math.radians(self.deg_angle)
            self.Lines[i].x = self.shape.x
            self.Lines[i].y = self.shape.y
            self.Lines[i].x2 = self.shape.x + math.cos(car_angle + math.radians(self.Angles[i])) * self.lineLengths[i]
            self.Lines[i].y2 = self.shape.y + math.sin(car_angle + math.radians(self.Angles[i])) * self.lineLengths[i]

    def is_on_agent(self, object_pos):
        return object_pos in self.shape
    def line_end_on_agent(self, x2, y2):
        return self.is_on_agent((x2, y2))



class RoadTile:
    def __init__(self, start_x, start_y, end_x, end_y, width, color, batch1, batch2):
        self.roadShape = shapes.Line(start_x, start_y, end_x, end_y, width, color, batch=batch1) #shape
        self.roadFollow = shapes.Line(start_x, start_y, end_x, end_y, 5, (255, 50, 50), batch=batch2) #line for vehicle to follow
        self.roadCheckPoint = shapes.Line(end_x-5, end_y, end_x, end_y, width, (50, 255, 50), batch=batch2) #checkpoint rewards
        self.passed = False

    def is_on_road(self, object_pos): #check if vehicle on road shape
        return object_pos in self.roadShape
    def is_on_follow(self, object_pos): #check orientation matches on road
        return object_pos in self.roadFollow
    def passed_checkpoint(self, object_pos): #check if vehicle on checkpoint
        if not self.passed:
            if object_pos in self.roadCheckPoint:
                self.passed = True
                return True
        return False

    

# class PedestrianAgent:
#     def __init__(self, start_x, start_y, end_x, end_y, )
        
