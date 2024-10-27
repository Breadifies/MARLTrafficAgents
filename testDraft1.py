import pyglet
import math
from pyglet.window import key, mouse
from pyglet import shapes
from pyglet import gl #graphics drawing done via GPU than CPU, more optimised

window = pyglet.window.Window(vsync=True)

window_width, window_height = window.get_size()


batch = pyglet.graphics.Batch()
batch2 = pyglet.graphics.Batch()

rectangle = shapes.Rectangle(x=400, y=450, width=100, height=50, color=(50, 225, 30), batch=batch)
rectangle.anchor_x = 50
rectangle.anchor_y = 25

#default anchor point is center

@window.event
def on_key_press(symbol, modifiers): #symbol is virtual key, modifiers area bitwise combination of present modifiers
    keys_pressed.add(symbol)
@window.event
def on_key_release(symbol, modifiers):
    keys_pressed.discard(symbol)


keys_pressed = set() #allows for multiple presses

velocity = 200
acceleration = 1000 #changed depending on cars
#again keeping things mostly fixed, same car speed and friction, keep these values non-variable 
friction = 2000 #can be changed depending on environment
car_angle = 0 #angle of rotation


line = shapes.Line(rectangle.x, rectangle.y, 50, 50, width=1, color=(255, 0, 0), batch=batch2)

def update(dt): #ensuring consistent framerate, game logic and object movement will be frame-rate independent, so the behaviour is consistent on monitors with different refresh rates, or frame rate drops due to performance inssues, -> conssitent results esp important for when handling detection
    global velocity, car_angle

    rad_angle = math.radians(car_angle)
    rectangle.rotation = rectangle.rotation % 360

    if key.LEFT in keys_pressed:
        car_angle += 1
    if key.RIGHT in keys_pressed:
        car_angle -= 1
    if key.UP in keys_pressed:
        velocity += acceleration * dt
    if key.DOWN in keys_pressed:
        velocity -= acceleration * dt

    #friction when no key pressed
    if key.LEFT not in keys_pressed and key.RIGHT not in keys_pressed and key.UP not in keys_pressed and key.DOWN not in keys_pressed:
        if velocity > 0:
            velocity -= friction * dt
        elif velocity < 0:
            velocity += friction * dt


    rectangle.x += velocity * dt * math.cos(rad_angle)
    rectangle.y += velocity * dt * math.sin(rad_angle)
    rectangle.rotation = car_angle
    line.x = rectangle.x
    line.y = rectangle.y
    line.x2 = car_angle
    line.y2 = math.sin(car_angle)

    if rectangle.x >= window.width or rectangle.x <= 0:
        velocity = -velocity
    if rectangle.y >= window.height or rectangle.y <= 0:
        velocity = -velocity

    

    #line for direction of car
    
    #clamping velocity to prevent needless values for pixels


    

pyglet.clock.schedule(update) #ensures calling a funtion ONCE per frame

@window.event
def on_draw():
    gl.glClearColor(240 / 255.0, 240 / 255.0, 240 / 255.0, 1.0)
    window.clear()
    batch.draw()
    batch2.draw()



pyglet.app.run()        