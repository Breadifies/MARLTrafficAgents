import pyglet
import numpy as np
import math
from pyglet.window import key, mouse
from pyglet import shapes
from pyglet import gl #graphics drawing done via GPU than CPU, more optimised
from vehicle_Agent import VehicleAgent, RoadTile




window = pyglet.window.Window(width=1568, height=882, vsync=True)  # vsync, eliminate unnecessary variables
WINDOW_WIDTH, WINDOW_HEIGHT = window.get_size()
batch = pyglet.graphics.Batch()
batch2 = pyglet.graphics.Batch()
roadBatch = pyglet.graphics.Batch()

keys_pressed = [0]*4 #allows for multiple presses
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


#/////////////////////////////////////////////////////////////////////////////////////////#/////////////////////////////////////////////////////////////////////////////////////////  
# //CONVERSION OF REAL-WORLD UNITS to SIMULATION UNITS  (meters, km/h to pixels/frame)
MU = 16 #1 meter = n pixels (based on vehicle width) (METER UNIT)
TOP_SPEED = MU*15 #48km/h  pixels per second
TOP_TURNING_SPEED = MU*13
ACCELERATION = MU * 5
DECELERATION = ACCELERATION * 3
TURNING_ACCEL = MU * 15 #assume turning acceleration of 5m/s
FRICTION = MU * 3 #FRICTION coefficient
TURNING_FRICTION = MU * 20 #how quickly rotation stops
CAR_LENGTH = MU*4.2 #4.2 meters, pixel per meter 21:9 ratio standard
CAR_WIDTH = MU*1.8  #= 1.8 meters to pixels is 24, therfore 1 meter = 13.33 pixels

vAgents = [
    VehicleAgent(x=450, y=450, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2, isControlled=True),
    # VehicleAgent(x=300, y=450, width=CAR_LENGTH, height=CAR_WIDTH, color=(50, 225, 30), batch1=batch, batch2=batch2)
    # VehicleAgent(x=120, y=450, width=38, height=25, color=(214, 125, 67), batch1=batch, batch2=batch2),
    # VehicleAgent(x=400, y=450, width=42, height=30, color=(50, 225, 30), batch1=batch, batch2=batch2),
    # VehicleAgent(x=300, y=450, width=42, height=30, color=(50, 225, 30), batch1=batch, batch2=batch2),
    # VehicleAgent(x=120, y=450, width=38, height=25, color=(214, 125, 67), batch1=batch, batch2=batch2),
    # VehicleAgent(x=500, y=400, width=45, height=32, color=(100, 150, 200), batch1=batch, batch2=batch2),
    # VehicleAgent(x=200, y=400, width=40, height=28, color=(255, 0, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=350, y=420, width=44, height=31, color=(0, 255, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=250, y=430, width=41, height=29, color=(0, 0, 255), batch1=batch, batch2=batch2),
    # VehicleAgent(x=100, y=460, width=39, height=26, color=(255, 255, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=450, y=410, width=43, height=29, color=(0, 255, 255), batch1=batch, batch2=batch2),
    # VehicleAgent(x=150, y=470, width=37, height=24, color=(255, 0, 255), batch1=batch, batch2=batch2),
    # VehicleAgent(x=320, y=440, width=46, height=33, color=(128, 128, 128), batch1=batch, batch2=batch2),
    # VehicleAgent(x=220, y=460, width=40, height=27, color=(128, 0, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=370, y=430, width=42, height=28, color=(0, 128, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=270, y=450, width=44, height=30, color=(0, 0, 128), batch1=batch, batch2=batch2),
    # VehicleAgent(x=180, y=480, width=41, height=29, color=(128, 128, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=420, y=420, width=45, height=31, color=(0, 128, 128), batch1=batch, batch2=batch2),
    # VehicleAgent(x=320, y=460, width=43, height=28, color=(128, 0, 128), batch1=batch, batch2=batch2),
    # VehicleAgent(x=370, y=440, width=46, height=33, color=(128, 128, 128), batch1=batch, batch2=batch2),
    # VehicleAgent(x=270, y=460, width=40, height=27, color=(128, 0, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=220, y=430, width=42, height=28, color=(0, 128, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=170, y=450, width=44, height=30, color=(0, 0, 128), batch1=batch, batch2=batch2),
    # VehicleAgent(x=420, y=480, width=41, height=29, color=(128, 128, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=320, y=420, width=45, height=31, color=(0, 128, 128), batch1=batch, batch2=batch2),
    # VehicleAgent(x=370, y=460, width=43, height=28, color=(128, 0, 128), batch1=batch, batch2=batch2),
    # VehicleAgent(x=270, y=440, width=46, height=33, color=(128, 128, 128), batch1=batch, batch2=batch2),
    # VehicleAgent(x=220, y=460, width=40, height=27, color=(128, 0, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=170, y=430, width=42, height=28, color=(0, 128, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=420, y=450, width=44, height=30, color=(0, 0, 128), batch1=batch, batch2=batch2),
    # VehicleAgent(x=320, y=480, width=41, height=29, color=(128, 128, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=370, y=420, width=45, height=31, color=(0, 128, 128), batch1=batch, batch2=batch2)
]
#/////////////////////////////////////////////////////////////////////////////////////////#////////////////////////////////////////////////////////////////////////////////////////       
ROAD_WIDTH = CAR_WIDTH * 2 #should be roughly DOUBLE a saloon width
roads = []
road = RoadTile(start_x=(WINDOW_WIDTH*0.1), start_y=WINDOW_HEIGHT/2, end_x=(WINDOW_WIDTH*0.9), end_y=WINDOW_HEIGHT/2, width=ROAD_WIDTH, color=(50, 50, 50), batch=roadBatch)
roads.append(road)
 

#dt is the time elapsed since last call
def update(dt): #ensuring consistent framerate and game logic to be frame-rate indepependent
    
    global keys_pressed, ACCELERATION, DECELERATION, TURNING_ACCEL, FRICTION, TURNING_FRICTION
    for a in vAgents: #looping through every vehicle agent
        left, right, down, up = a.current_direction
        if left:
            a.turning_speed += TURNING_ACCEL * dt
        if right:
            a.turning_speed -= TURNING_ACCEL * dt

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
        if not right and not left:
            if a.turning_speed > 0:
                a.turning_speed -= TURNING_FRICTION * dt 
            elif a.turning_speed < 0:
                a.turning_speed += TURNING_FRICTION * dt 
        #rotate and translate the vehicle
        a.car_angle = math.radians(a.deg_angle)
        a.shape.x += a.velocity * dt * math.cos(a.car_angle) 
        a.shape.y += a.velocity * dt * math.sin(a.car_angle) 
        a.deg_angle += a.turning_speed * dt
        a.shape.rotation = -a.deg_angle

        if a.shape.x >= window.width or a.shape.x <= 0:
            a.velocity = -a.velocity+10
        if a.shape.y >= window.height or a.shape.y <= 0:
            a.velocity = -a.velocity+10

        if abs(a.velocity) < 0.1: #clamp minimum speed
            a.velocity = 0
        if abs(a.turning_speed) < 0.1:
            a.turning_speed = 0
        if abs(a.velocity) > TOP_SPEED: #clamp top speed (max drag)
            a.velocity = TOP_SPEED*np.sign(a.velocity)
        if abs(a.turning_speed) > TOP_TURNING_SPEED: #clamp top turning speed
           a.turning_speed = TOP_TURNING_SPEED*np.sign(a.turning_speed)

        # Check if the vehicle is on the road
        a.changeAnchor(a.center_anchor_x, a.center_anchor_y) #center anchor position temporarily
        for road in roads:
            if road.is_on_road(a.shape.position): #not in road
                a.shape.color = (255, 0, 0)  # Stop the vehicle if it's off the road
            else:
                a.shape.color = (50, 225, 30)

            for i in range(a.num_vision_lines):
                if not road.line_end_on_road(a.Lines[i].x2, a.Lines[i].y2):
                    #decrease the vision length until it is NOT on the road
                    a.lineLengths[i] = max(0, a.lineLengths[i] - 1)
                    a.updateLines()
                else:
                    a.lineLengths[i] = min(a.maxLen, a.lineLengths[i] + 1)
        a.updateLines()
        a.changeAnchor(a.turning_anchor_x, a.turning_anchor_y)

def update_direction(dt): #operates on longer timesteps than animation update
    for a in vAgents:
        if not a.isControlled:
            a.updateDirection()
        currentState = [a.shape.x, a.shape.y, a.deg_angle, a.velocity, a.turning_speed, a.lineLengths]
        print(currentState)


def update_user_direction(dt):
    for a in vAgents:
        if a.isControlled:
            a.current_direction = keys_pressed


pyglet.clock.schedule_interval(update_direction, 0.3)  # Update direction every xx seconds
pyglet.clock.schedule(update)
pyglet.clock.schedule(update_user_direction)


@window.event
def on_draw():
    gl.glClearColor(240 / 255.0, 240 / 255.0, 240 / 255.0, 1.0) #colour light grey
    window.clear()
    roadBatch.draw()
    batch2.draw() #vision lines
    batch.draw()
    

pyglet.app.run()        