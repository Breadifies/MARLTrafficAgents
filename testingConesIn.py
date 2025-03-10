import pyglet
import math
from pyglet import shapes

# Window setup
WIDTH, HEIGHT = 600, 600
window = pyglet.window.Window(WIDTH, HEIGHT)
batch = pyglet.graphics.Batch()


center_x, center_y = WIDTH // 2, HEIGHT // 2  
radius = 150  
start_angle = 30  
angle = 300  


# sector = shapes.Sector(
#     x=center_x, 
#     y=center_y, 
#     radius=radius, 
#     angle=angle, 
#     start_angle=start_angle, 
#     color=(255, 100, 100),
#     batch=batch
# ) 

# circle = shapes.Circle(
#     x=center_x, 
#     y=center_y, 
#     radius=radius, 
#     color=(255, 100, 100, 200), 
#     batch=batch
# )

# Coordinates for the triangle's three vertices
x1, y1 = 100, 100  # First vertex
x2, y2 = 250, 400  # Second vertex
x3, y3 = 400, 100  # Third vertex

# Color of the triangle (RGB format, red color)
color = (100, 100, 255, 100)  # Red color (no transparency)

# Create the triangle shape
triangle = shapes.Triangle(x1, y1, x2, y2, x3, y3, color=color, batch=batch)


@window.event
def on_mouse_press(x, y, button, modifiers):
    inside = False
    if (x, y) in triangle:    
        inside = True
    print(f"Point ({x}, {y}) is inside sector: {inside}")
    
    
@window.event
def on_draw():
    window.clear()
    batch.draw() 

pyglet.app.run()
