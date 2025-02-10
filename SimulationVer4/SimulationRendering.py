import pyglet
import numpy as np
import math
from pyglet.window import key
from pyglet import shapes
from pyglet import gl
import copy
from shapeClasses import VehicleAgent, RoadTile, PedestrianAgent
import torch
import torch.nn as nn
from torch.nn import functional as F

window = pyglet.window.Window(width=800, height=600, vsync=True)
WINDOW_WIDTH, WINDOW_HEIGHT = window.get_size()

batch = pyglet.graphics.Batch()
batch_ = pyglet.graphics.Batch()
batch1 = pyglet.graphics.Batch()
batch2 = pyglet.graphics.Batch()

###################CONSTANTS#######################################
ROAD_WIDTH = 20

MU = 7 #1 meter = n pixels (based on vehicle width) (STANDARDIZED METER UNIT)
CAR_LENGTH = MU*4.2 #21:9 ratio
CAR_WIDTH = MU*1.8
TOP_SPEED = MU*30
TOP_REV_SPEED = TOP_SPEED/2
ACCELERATION = MU*70
DECELERATION = ACCELERATION * 2.5 #braking force is much stronger than accel
FRICTION = MU*12 #friction coefficient

SPEED_UP = 1 #speed up simulation
###################################################################


###################CONSTRUCTING ROADS##############################
# center_x = WINDOW_WIDTH // 2
# center_y = WINDOW_HEIGHT // 2
# radius = 150
# num_tiles = 10
# angle_increment = 2 * math.pi / num_tiles
startProads = [
    # RoadTile(
    #     start_x=center_x+ radius * math.cos(angle_increment * i),
    #     start_y=center_y + radius * math.sin(angle_increment * i),
    #     end_x=center_x + radius * math.cos(angle_increment * (i + 1)),
    #     end_y=center_y + radius * math.sin(angle_increment * (i + 1)), 
    #     width=ROAD_WIDTH,
    #     color=(50, 50, 50),
    #     batch1=batch,
    #     batch2 = batch2
    # )
    # for i in range(num_tiles)
    RoadTile(start_x = -100, start_y = WINDOW_HEIGHT//2, end_x = 150, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2),
    RoadTile(start_x = 150, start_y = WINDOW_HEIGHT//2, end_x = 200, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2),
    RoadTile(start_x = 200, start_y = WINDOW_HEIGHT//2, end_x = 250, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2),
    RoadTile(start_x = 250, start_y = WINDOW_HEIGHT//2, end_x = 300, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2),
    RoadTile(start_x = 300, start_y = WINDOW_HEIGHT//2, end_x = 350, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2),
    RoadTile(start_x = 350, start_y = WINDOW_HEIGHT//2, end_x = 400, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2),
    RoadTile(start_x = 400, start_y = WINDOW_HEIGHT//2, end_x = 450, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2),
    RoadTile(start_x = 450, start_y = WINDOW_HEIGHT//2, end_x = 500, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2),
    RoadTile(start_x = 500, start_y = WINDOW_HEIGHT//2, end_x = 550, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2),
    RoadTile(start_x = 550, start_y = WINDOW_HEIGHT//2, end_x = 600, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2),
    RoadTile(start_x = 600, start_y = WINDOW_HEIGHT//2, end_x = 650, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2),
    RoadTile(start_x = 650, start_y = WINDOW_HEIGHT//2, end_x = 700, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2)

]
###################################################################


###################CONSTRUCTING AGENTS#############################
startPvAgents = [
    # VehicleAgent(x=400, y=150, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2),
    # VehicleAgent(x=400, y=450, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2),
    # VehicleAgent(x=250, y=310, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2),
    # VehicleAgent(x=550, y=310, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2)
    VehicleAgent(x=120, y=300, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch_)
    ]
initVel = [0 * len(startPvAgents)]
initAngle = [0 * len(startPvAgents)]
###################################################################


#################CONSTRUCTINGPEDESTRIANS#############################
startPAgents = [
    PedestrianAgent(x=400, y=400, target_x = 400, target_y = 200, radius=10, color=(0, 0, 100), batch=batch)
]
###################################################################

roads = startProads
vAgents = []
pAgents = []
for item in startPvAgents:
    vAgents.append(item.__deepcopy__(batch1, batch2))
for item in startPAgents:
    pAgents.append(item.__deepcopy__(batch1))


###############PHYSICS RENDERING###################################
def update(dt):
    dt = dt * SPEED_UP
    global ACCELERATION, DECELERATION, FRICTION, TOP_SPEED, TOP_REV_SPEED, HIT_CHECKPOINT
    for p in pAgents:
        p.move(dt*0.1) #move towards target
    for a in vAgents:
        down, up = a.current_direction
        #forwards and backwards
        if up:
            a.velocity += ACCELERATION * dt
        if down:
            a.velocity -= DECELERATION * dt
        #friction
        if not up and not down:
            if a.velocity > 0:
                a.velocity -= FRICTION * dt
            elif a.velocity < 0:
                a.velocity += FRICTION * dt

        #ENSURE VEHICLE ORIENTED ON ROAD
        followAnchor = None
        onFollow = False
        #choose either forward or backward follow anchor
        if a.velocity > 0:
            followAnchor = a.followF_anch_x
        else:
            followAnchor = a.followB_anch_x

        a.changeAnchor(followAnchor, a.turn_anch_y) #switch to follow anchor (detecting road orientation)
        if not a.curRoad or not a.curRoad.is_on_road(a.shape.position):#check if on a different road
            for road in roads:
                if road.is_on_road(a.shape.position):
                    a.curRoad = road
        
        changedAngle = 0 #final angle change post-successful orient
        if a.curRoad and not onFollow: #if on a road, check orientation
            if a.curRoad.is_on_follow(a.shape.position):
                onFollow = True
                a.increment = 0
            else: #not oriented correctly -> ROTATE
                a.increment += 5 #sweep angle outwards till orient found
                a.changeAnchor(a.turn_anch_x, a.turn_anch_y)#rotate based on c.o.r
                changedAngle += a.clockwise*a.increment
                a.shape.rotation += a.clockwise*a.increment
                a.changeAnchor(followAnchor, a.turn_anch_y) #reset anchor to follow anchor
                a.clockwise = -a.clockwise #rotate in opposite direction
        a.changeAnchor(a.turn_anch_x, a.turn_anch_y) 


        #rotate vehicle
        a.deg_angle = (a.deg_angle - changedAngle)%360
        car_angle = math.radians(a.deg_angle)
        a.shape.x += a.velocity * dt * math.cos(car_angle)
        a.shape.y += a.velocity * dt * math.sin(car_angle)
        a.shape.rotation = -a.deg_angle
        
        a.updateLines()
        
        if abs(a.velocity) < 0.1: #clamp minimum speed
            a.velocity = 0
        if a.velocity >= 0:
            if a.velocity > TOP_SPEED: #clamp top speed (max drag)
                a.velocity = TOP_SPEED*np.sign(a.velocity)
        else:
            if a.velocity < -TOP_REV_SPEED: #clamp top reverse speed
                a.velocity = TOP_REV_SPEED*np.sign(a.velocity)

###################################################################


###################DRAWTOSCREEN####################################
@window.event
def on_draw():
    gl.glClearColor(1, 1, 1, 1.0)
    window.clear()
    batch1.draw()
    batch2.draw()
    
###################################################################

UPDATE_FREQUENCY = 0.05 #updating state
MAX_EP_LENGTH = 5/UPDATE_FREQUENCY
ep_len = 0

###################EPISODE-RENDER##################################
def calculate_reward(current_state):
    reward = 0
    if current_state[3] >=0: #encourage forward movement
        reward += current_state[3]*2
    return reward

def envReset():
    global vAgents, pAgents, ep_len, initVel, roads
    #vAgents = create New agents function ()
    vAgents.clear()
    pAgents.clear()
    for item in startPvAgents:
        vAgents.append(item.__deepcopy__(batch1, batch2))
    for item in startPAgents:
        pAgents.append(item.__deepcopy__(batch1))
    print("RESET")
    for i, a in enumerate(vAgents):
        a.velocity = initVel[i]
        a.shape.rotation = initAngle[i]
    ep_len = 0

def envStep(dt):
    global ep_len
    done = False
    currentState = []
    current_reward = 0
    ep_len += 1
    for a in vAgents:
        currentState = [a.shape.x, a.shape.y, a.deg_angle, a.velocity]
        current_reward = calculate_reward(currentState)
        a.updateDirection(selectAction())
    if ep_len >= MAX_EP_LENGTH: #if exceeds max length or last part of road passed
        envReset()
        done = True
    return currentState, current_reward, done
    #returns states of agents (array), actions taken (array), done (terminal flag) 
def selectAction():
    return 2 #forwards
###################################################################

pyglet.clock.schedule_interval(envStep, UPDATE_FREQUENCY/SPEED_UP)
pyglet.clock.schedule(update)#call update function according to system refresh rate

pyglet.app.run()



