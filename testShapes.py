import pyglet
import numpy as np
import math
from pyglet.window import key, mouse
from pyglet import shapes
from pyglet import gl 

window = pyglet.window.Window(width=800, height=600, vsync=True)  # vsync, eliminate unnecessary variables
WINDOW_WIDTH, WINDOW_HEIGHT = window.get_size()

batch = pyglet.graphics.Batch()
batch2 = pyglet.graphics.Batch()

class RoadTile:
    def __init__(self, start_x, start_y, end_x, end_y, width, color, batch, batch2):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.width = width
        self.color = color
        self.batch = batch
        self.roadShape = shapes.Line(start_x, start_y, end_x, end_y, width, color, batch=batch) #shape
        self.roadFollow = shapes.Line(start_x, start_y, end_x, end_y, 5, (255, 0, 0), batch=batch2) #line for vehicle to follow

    def is_on_road(self, object_pos): #check on road itself
        return object_pos in self.roadShape
    
    def is_on_follow(self, object_pos): #check orientation matches on road
        return object_pos in self.roadFollow
    
    
class VehicleAgent:
    def __init__(self, x, y, width, height, color, batch1):
        self.shape = shapes.Rectangle(x=x, y=y, width=width, height=height, color=color, batch=batch1)
        self.width = width
        self.height = height
        self.turning_anchor_y = height // 2 #center at all times
        self.turning_anchor_x = width // 4
        self.followF_anchor_x = width / 1.5 #forwards driving
        self.followB_anchor_x = 2*self.turning_anchor_x - self.followF_anchor_x #backwards driving (center of rotation outside)
        self.deg_angle = 0
        self.shape.anchor_position = self.turning_anchor_x, self.turning_anchor_y

        self.velocity = 0
        self.turning_speed = 0
        self.current_direction = [0, 0, 0, 0]

        self.curRoad = None #check the road segment its on
        self.increment = 0
        self.clockwise = 1

    def updateDirection(self, action): #vehicle only changes its desired direction after a chosen number of timesteps (longer than native refresh rate)
        self.current_direction = self.getDirection(action)

    def changeAnchor(self, anchor_x, anchor_y): #has to be done manually due to pyglet shape properties
        #calculate difference in anchor position
        dx = anchor_x - self.shape.anchor_x
        dy = anchor_y - self.shape.anchor_y
        car_angle = math.radians(self.deg_angle)
        rotated_dx = dx * math.cos(car_angle) - dy * math.sin(car_angle)
        rotated_dy = dx * math.sin(car_angle) + dy * math.cos(car_angle)
        #Update shape's position to account for new anchor
        self.shape.x += rotated_dx
        self.shape.y += rotated_dy
        #Update anchor position
        self.shape.anchor_position = (anchor_x, anchor_y)





ROAD_WIDTH = 20
# Circle parameters
center_x = 250  # Center of the circle (x-coordinate)
center_y = 250  # Center of the circle (y-coordinate)
radius = 150  # Radius of the circle
num_tiles = 20  # Number of road tiles
# Angle between each tile
angle_increment = 2 * math.pi / num_tiles
# Creating road tiles along a circular path
roads = [
    RoadTile(
        start_x=center_x + radius * math.cos(angle_increment * i),
        start_y=center_y + radius * math.sin(angle_increment * i),
        end_x=center_x + radius * math.cos(angle_increment * (i + 1)),
        end_y=center_y + radius * math.sin(angle_increment * (i + 1)),
        width=ROAD_WIDTH,
        color=(50, 50, 50),
        batch=batch,
        batch2 = batch2
    )
    for i in range(num_tiles)
]


#allows for multiple presses
keys_pressed = [0]*4 
@window.event
def on_key_press(symbol, modifiers): #symbol is virtual key, modifiers area bitwise combination of present modifiers
    if symbol == key.LEFT:
        keys_pressed[0] = 1
    elif symbol == key.RIGHT:
        keys_pressed[1] = 1
    elif symbol == key.DOWN:
        keys_pressed[2] = 1
    elif symbol == key.UP:
        keys_pressed[3] = 1
@window.event
def on_key_release(symbol, modifiers):
    if symbol == key.LEFT:
        keys_pressed[0] = 0
    elif symbol == key.RIGHT:
        keys_pressed[1] = 0
    elif symbol == key.DOWN:
        keys_pressed[2] = 0
    elif symbol == key.UP:
        keys_pressed[3] = 0

MU = 7 #1 meter = n pixels (based on vehicle width) (METER UNIT)
CAR_LENGTH = MU*4.2 #4.2 meters, pixel per meter 21:9 ratio standard
CAR_WIDTH = MU*1.8  #= 1.8 meters to pixels is 24, therfore 1 meter = 13.33 pixels
TOP_SPEED = MU*50 #150 #48km/h  pixels per second (15)
ACCELERATION = MU * 70#5
DECELERATION = ACCELERATION * 2.5
FRICTION = MU * 12 #3 #FRICTION coefficient 
INITIAL_VELOCITY = 0 #initial velocity

SPEED_UP = 1 #speed up simulation

#SETUP ENVIRONMENT
vAgents = [
    VehicleAgent(x=250, y=100, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch),
    VehicleAgent(x=250, y=400, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch),
    VehicleAgent(x=100, y=250, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch),
    VehicleAgent(x=400, y=250, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch)
]

vAgents[0].velocity = INITIAL_VELOCITY #initial velocity
vAgents[0].deg_angle = 20 #initial angle

def update(dt):
    dt = dt * SPEED_UP
    global keys_pressed, ACCELERATION, DECELERATION, FRICTION
    for a in vAgents:
        #NO TURNING
        left, right, down, up = a.current_direction

        if up:
            a.velocity += ACCELERATION * dt 
        if down:
            a.velocity -= DECELERATION * dt 
        #FRICTION
        if not up and not down:
            if a.velocity > 0:
                a.velocity -= FRICTION * dt     
            elif a.velocity < 0:
                a.velocity += FRICTION * dt 



        #CHECK IF FOLLOWING ROAD LINE
        if a.curRoad == None: #if no road assigned
            a.curRoad = roads[0] #assign first road to bypass NoneType error
        curFollow = None
        if a.velocity >= 1: #determine which follow rotation to use
            curFollow = a.followF_anchor_x
            a.shape.color = (200, 0, 0)
        elif a.velocity <= 1:
            curFollow = a.followB_anchor_x
            a.shape.color = (0, 0, 200)

        a.changeAnchor(curFollow, a.turning_anchor_y) #road check based on follow anchor
        if not a.curRoad.is_on_road((a.shape.position)):#vehicle on a new road
            for road in roads: #check what new road its on
                if road.is_on_road((a.shape.position)): # new road found
                        a.curRoad = road  #reassign road
    
        onFollow = False
        changedAngle = 0
        if not onFollow:#check if vehicle is oriented correctly on road segment
            if a.curRoad.is_on_follow(a.shape.position): #check if oriented correctly
                a.increment = 0
                onFollow = True       
            else: #NOT ORIENTED CORRECTLY -> ROTATE
                a.increment += 0.5
                a.changeAnchor(a.turning_anchor_x, a.turning_anchor_y) #rotate based on c.o.r
                changedAngle += a.clockwise*a.increment
                a.shape.rotation += a.clockwise*a.increment
                a.changeAnchor(curFollow, a.turning_anchor_y) #reset anchor to follow anchor
                a.clockwise = -a.clockwise #next time rotate in opposite direction
                        
        a.changeAnchor(a.turning_anchor_x, a.turning_anchor_y)
        #rotate the vehicle
        a.deg_angle = (a.deg_angle - changedAngle)%360
        car_angle = math.radians(a.deg_angle)
        a.shape.x += a.velocity * dt * math.cos(car_angle) 
        a.shape.y += a.velocity * dt * math.sin(car_angle) 
        a.shape.rotation = -a.deg_angle



        if abs(a.velocity) < 0.1: #clamp minimum speed
            a.velocity = 0
        if a.velocity >= 0:
            if (a.velocity) > TOP_SPEED: #clamp top speed (max drag)
                a.velocity = TOP_SPEED*np.sign(a.velocity)
        else:
            if (a.velocity) < -TOP_SPEED/2: #clamp reverse speed
                a.velocity = TOP_SPEED/2*np.sign(a.velocity)
        

def update_user_direction(dt):
    for a in vAgents:
        a.current_direction = keys_pressed

@window.event
def on_draw():
    gl.glClearColor(240 / 255.0, 240 / 255.0, 240 / 255.0, 1.0) #colour light grey
    window.clear()
    batch.draw()

pyglet.clock.schedule(update)
pyglet.clock.schedule(update_user_direction)

pyglet.app.run()