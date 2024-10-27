import pyglet
import numpy as np
import math
from pyglet.window import key, mouse
from pyglet import shapes
from pyglet import gl #graphics drawing done via GPU than CPU, more optimised
from vehicle_Agent import VehicleAgent


window = pyglet.window.Window(vsync=True) #vysnc, eliminate unecessary variables

window_width, window_height = window.get_size()

batch = pyglet.graphics.Batch()
batch2 = pyglet.graphics.Batch()

keys_pressed = set() #allows for multiple presses
@window.event
def on_key_press(symbol, modifiers): #symbol is virtual key, modifiers area bitwise combination of present modifiers
    keys_pressed.add(symbol)
@window.event
def on_key_release(symbol, modifiers):
    keys_pressed.discard(symbol)


#standardise vehicles
CAR_WIDTH = 42
CAR_HEIGHT = 30
vAgent = shapes.Rectangle(x=400, y=450, width=CAR_WIDTH, height=CAR_HEIGHT, color=(50, 225, 30), batch=batch)
vAgent.anchor_x = CAR_WIDTH//4
vAgent.anchor_y = CAR_HEIGHT//2

degCar_Angle = 0
car_angle = math.radians(degCar_Angle) #in radians
VISION_LENGTH = 2*CAR_WIDTH
frontVehicle = shapes.Line(vAgent.x, vAgent.y, vAgent.x+math.cos(car_angle)*VISION_LENGTH, vAgent.y+math.sin(car_angle)*VISION_LENGTH, width=1, color=(255, 0, 0),batch=batch2) 
leftVehicle = shapes.Line(vAgent.x, vAgent.y, vAgent.x+math.cos(car_angle+math.radians(45))*VISION_LENGTH, vAgent.y+math.sin(car_angle+math.radians(45))*VISION_LENGTH, width=1, color=(255, 0, 0),batch=batch2) 
rightVehicle = shapes.Line(vAgent.x, vAgent.y, vAgent.x+math.cos(car_angle-math.radians(45))*VISION_LENGTH, vAgent.y+math.sin(car_angle-math.radians(45))*VISION_LENGTH, width=1, color=(255, 0, 0),batch=batch2) #determine front of vehicle


velocity = 0
acceleration= 300
friction = 500
turning_speed = 0 #degree of turning for vehicle
turning_accel = 500
turning_friction = 2500


def update(dt): #ensuring consistent framerate and game logic to be frame-rate indepependent
    global velocity, turning_speed, degCar_Angle, car_angle, turning_accel
    if key.LEFT in keys_pressed:
        turning_speed += turning_accel * dt
    if key.RIGHT in keys_pressed:
        turning_speed -= turning_accel * dt

    if key.UP in keys_pressed:
        velocity += acceleration * dt
    if key.DOWN in keys_pressed:
        velocity -= acceleration * dt

    #friction when no key pressed
    if key.UP not in keys_pressed and key.DOWN not in keys_pressed:
        if velocity > 0:
            velocity -= friction * dt
        elif velocity < 0:
            velocity += friction * dt
    if key.RIGHT not in keys_pressed and key.LEFT not in keys_pressed:
        if turning_speed > 0:
            turning_speed -= turning_friction * dt
        elif turning_speed < 0:
            turning_speed += turning_friction * dt


    car_angle = math.radians(degCar_Angle)
    vAgent.x += velocity * dt * math.cos(car_angle) 
    vAgent.y += velocity * dt * math.sin(car_angle) 
    degCar_Angle += turning_speed * dt
    
    frontVehicle.x = leftVehicle.x = rightVehicle.x = vAgent.x
    frontVehicle.y = leftVehicle.y = rightVehicle.y = vAgent.y
    frontVehicle.rotation = leftVehicle.rotation = rightVehicle.rotation = -degCar_Angle
    vAgent.rotation = -degCar_Angle

    if vAgent.x >= window.width or vAgent.x <= 0:
        velocity = -velocity
    if vAgent.y >= window.height or vAgent.y <= 0:
        velocity = -velocity
        
    if abs(velocity) > 500: #clamp top speed (max drag)
        velocity = 500*np.sign(velocity)
    if abs(turning_speed) > 300:
        turning_speed = 300*np.sign(turning_speed)

    


pyglet.clock.schedule(update)

@window.event
def on_draw():
    gl.glClearColor(240 / 255.0, 240 / 255.0, 240 / 255.0, 1.0) #colour light grey
    window.clear()
    batch2.draw()
    batch.draw()
    



pyglet.app.run()        