import pyglet
import numpy as np
import math
from pyglet.window import key, mouse
from pyglet import shapes
from pyglet import gl #graphics drawing done via GPU than CPU, more optimised
from vehicle_Agent import VehicleAgent, RoadTile


window = pyglet.window.Window(vsync=True) #vysnc, eliminate unecessary variables

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
acceleration = 300
turning_accel = 500
friction = 500
turning_friction = 2500
# Create vehicle agents in an iterable -> loop through these in updates
vAgents = [
    VehicleAgent(x=450, y=450, width=42, height=30, color=(200, 225, 90), batch1=batch, batch2=batch2, isControlled=True),
    # VehicleAgent(x=300, y=450, width=42, height=30, color=(50, 225, 30), batch1=batch, batch2=batch2),
    # VehicleAgent(x=120, y=450, width=38, height=25, color=(214, 125, 67), batch1=batch, batch2=batch2),
    # VehicleAgent(x=400, y=450, width=42, height=30, color=(50, 225, 30), batch1=batch, batch2=batch2),
    # VehicleAgent(x=300, y=450, width=42, height=30, color=(50, 225, 30), batch1=batch, batch2=batch2),
    # VehicleAgent(x=120, y=450, width=38, height=25, color=(214, 125, 67), batch1=batch, batch2=batch2),
    # VehicleAgent(x=500, y=400, width=45, height=32, color=(100, 150, 200), batch1=batch, batch2=batch2),
    # VehicleAgent(x=200, y=400, width=40, height=28, color=(255, 0, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=350, y=420, width=44, height=31, color=(0, 255, 0), batch1=batch, batch2=batch2),
    # VehicleAgent(x=250, y=430, width=41, height=29, color=(0, 0, 255), batch1=batch, batch2=batch2)
]
#/////////////////////////////////////////////////////////////////////////////////////////#////////////////////////////////////////////////////////////////////////////////////////       
ROAD_WIDTH = 80
roads = []
road = RoadTile(start_x=(WINDOW_WIDTH*0.1), start_y=WINDOW_HEIGHT/2, end_x=(WINDOW_WIDTH*0.9), end_y=WINDOW_HEIGHT/2, width=ROAD_WIDTH, color=(50, 50, 50), batch=roadBatch)
roads.append(road)
 

#dt is the time elapsed since last call
def update(dt): #ensuring consistent framerate and game logic to be frame-rate indepependent
    global acceleration, turning_accel, keys_pressed, turning_friction, friction

    for a in vAgents: #looping through every vehicle agent
        left, right, down, up = a.current_direction
        if left:
            a.turning_speed += turning_accel * dt
        if right:
            a.turning_speed -= turning_accel * dt 
        if up:
            a.velocity += acceleration * dt 
        if down:
            a.velocity -= acceleration * dt 

        #friction when no key pressed
        if not up and not down:
            if a.velocity > 0:
                a.velocity -= friction * dt     
            elif a.velocity < 0:
                a.velocity += friction * dt 
        if not right and not left:
            if a.turning_speed > 0:
                a.turning_speed -= turning_friction * dt 
            elif a.turning_speed < 0:
                a.turning_speed += turning_friction * dt 
        #rotate and translate the vehicle
        a.car_angle = math.radians(a.deg_angle)
        a.shape.x += a.velocity * dt * math.cos(a.car_angle) 
        a.shape.y += a.velocity * dt * math.sin(a.car_angle) 
        a.deg_angle += a.turning_speed * dt
        a.shape.rotation = -a.deg_angle

        #rotate and translate the vision lines
        a.changeAnchor(a.center_anchor_x, a.center_anchor_y)#center anchor position temporarily
        print("ANCHOR BEFORE: ", a.shape.anchor_position, a.shape.position)

        a.front_vehicle.x = a.left_vehicle.x = a.right_vehicle.x = a.shape.x
        a.front_vehicle.y = a.left_vehicle.y = a.right_vehicle.y = a.shape.y
        print("CHECKING front vehicle a: ", a.front_vehicle.position)
        a.front_vehicle.rotation = a.left_vehicle.rotation = a.right_vehicle.rotation = -a.deg_angle
        if a.shape.x >= window.width or a.shape.x <= 0:
            a.velocity = -a.velocity+10
        if a.shape.y >= window.height or a.shape.y <= 0:
            a.velocity = -a.velocity+10
            
        if abs(a.velocity) > 500: #clamp top speed (max drag)
            a.velocity = 500*np.sign(a.velocity)
        if abs(a.turning_speed) > 300:
           a.turning_speed = 300*np.sign(a.turning_speed)
        # Check if the vehicle is on the road
        if road.is_on_road(a): #not in road
            a.shape.color = (255, 0, 0)  # Stop the vehicle if it's off the road
        else:
            a.shape.color = (50, 225, 30)

        
        a.changeAnchor(a.turning_anchor_x, a.turning_anchor_y)
        print("ANCHOR AFTER: ", a.shape.anchor_position, a.shape.position)
        print("CHECKING front vehicle a AFTER: ", a.front_vehicle.position)

def update_direction(dt): #operates on longer timesteps than animation update
    for a in vAgents:
        if not a.isControlled:
            a.updateDirection()

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
    batch2.draw()
    batch.draw()
    

pyglet.app.run()        