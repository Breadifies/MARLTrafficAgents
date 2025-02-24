import math
from pyglet import shapes
from pyglet.gl import *

class VehicleAgent:
    def __init__(self, x, y, width, height, color, batch1, batch2, isControlled = False):
        self.shape = shapes.Rectangle(x=x, y=y, width=width, height=height, color=color, batch=batch1)
        self.turn_anch_y = height // 2 #center at all times
        self.turn_anch_x = width // 4
        self.followF_anch_x = width / 1.5
        self.followB_anch_x = 2*self.turn_anch_x - self.followF_anch_x #backwards driving (center of rotation outside shape)
        self.center_anch_x = width // 2 #anchor for center (vision lines)
        self.deg_angle = 0
        self.changedAngle = 0 #for rotating vision cones
        self.shape.anchor_position = self.turn_anch_x, self.turn_anch_y

        #vehicle movement properties
        self.velocity = 0
        self.turning_speed = 0
        self.isControlled = isControlled
        self.current_direction = [0, 0] #input direction

        #vehicle vision (sector) properties
        self.maxLenCone = 3*width
        self.Cones = []
        self.coneWidth = [math.sin(30)]
        self.startAnglesCones = [self.coneWidth[0]]
        self.num_cones = len(self.coneWidth)
        self.coneLengths = [self.maxLenCone]*self.num_cones
        self.defineCones(x, y, batch2)

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

    def defineCones(self, x, y, batch): #create vehicle vision cones
        for i in range(self.num_cones):
            cone = shapes.Sector(x=x, y=y, radius=self.maxLenCone, angle=self.coneWidth[i], start_angle=math.cos(self.deg_angle + self.startAnglesCones[i]), color=(255, 0, 0, 100), batch=batch)
            self.Cones.append(cone)

    def updateCones(self):
        for i in range(self.num_cones):
            self.Cones[i].x = self.shape.x
            self.Cones[i].y = self.shape.y
            self.Cones[i].radius = self.coneLengths[i]
            self.Cones[i].rotation = self.deg_angle/2 #don't know why its this value


    def is_on_agent(self, object_pos):
        return object_pos in self.shape
    
    def is_on_cones(self, x, y): #check if an object is inside a cone
        conesIndex = [False]*self.num_cones
        for i in range(self.num_cones):
            #print(self.shape.y)
            if (10, 300) in self.Cones[i]:
                print(conesIndex)
                conesIndex[i] = True
        return conesIndex


    def __deepcopy__(self, renderBatch, renderBatch2):
        # Create a new instance of VehicleAgent
        new_copy = VehicleAgent(
            self.shape.x, self.shape.y, self.shape.width, self.shape.height,
            self.shape.color, renderBatch, renderBatch2, self.isControlled
        )
        # Manually copy the attributes
        return new_copy




class RoadTile:
    def __init__(self, start_x, start_y, end_x, end_y, width, color, batch1, batch2):
        self.roadShape = shapes.Line(start_x, start_y, end_x, end_y, width, color, batch=batch1) #shape
        self.roadFollow = shapes.Line(start_x, start_y, end_x, end_y, 5, (255, 210, 220), batch=batch2) #line for vehicle to follow
        #self.roadCheckPoint = shapes.Line(end_x-5, end_y, end_x, end_y, width, (50, 255, 50), batch=batch2) #checkpoint rewards
        self.passed = False

    def is_on_road(self, object_pos): #check if vehicle on road shape
        return object_pos in self.roadShape
    def is_on_follow(self, object_pos): #check orientation matches on road
        return object_pos in self.roadFollow
    
   # def passed_checkpoint(self, object_pos): #check if vehicle on checkpoint
        if not self.passed:
            if object_pos in self.roadCheckPoint:
                self.passed = True
                return True
        return False

    

class PedestrianAgent:
    def __init__(self, x, y, target_x, target_y, radius, color, batch):
        self.shape = shapes.Circle(x=x, y=y, radius=radius, color=color, batch=batch)
        self.radius = radius
        self.timestep = 0
        self.start_x = x
        self.start_y = y
        self.target_x = target_x
        self.target_y = target_y
        self.reached_end = False
    
    def move(self, speed):
        if not self.reached_end:
            self.timestep += 1
            if self.timestep % 1 == 0:
                self.shape.x += (self.target_x - self.start_x) * speed
                self.shape.y += (self.target_y - self.start_y) * speed
                if self.shape.x == self.target_x and self.shape.y == self.target_y:
                    self.reached_end = True
        else:
            print("pedestrian reached target")

    def is_on_ped(self, object_pos):
        return object_pos in self.shape
    def line_end_on_ped(self, x2, y2): #check for contact with vehicle
        return self.is_on_ped((x2, y2))

    def __deepcopy__(self, renderBatch):
        # Create a new instance of PedestrianAgent
        new_copy = PedestrianAgent(
            self.start_x, self.start_y, self.target_x, self.target_y,
            self.shape.radius, self.shape.color, renderBatch
        )
        # Manually copy the attributes
        new_copy.timestep = self.timestep
        new_copy.reached_end = self.reached_end
        return new_copy