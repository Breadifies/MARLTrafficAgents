import pyglet
from pyglet import shapes
from pyglet import gl

# Create a window
window = pyglet.window.Window(800, 600)

# Create a batch for efficient rendering
batch = pyglet.graphics.Batch()

# Define road parameters
ROAD_RADIUS = 500
ROAD_WIDTH = 50

# Create the outer circle for the road
outer_circle = shapes.Circle(x=window.width // 2, y=window.height // 2, radius=ROAD_RADIUS, color=(50, 50, 50), batch=batch)

# Create the inner circle to represent the inner boundary of the road
inner_circle = shapes.Circle(x=window.width // 2, y=window.height // 2, radius=ROAD_RADIUS - ROAD_WIDTH, color=(240, 240, 240), batch=batch)

@window.event
def on_draw():
    gl.glClearColor(1, 1, 1, 1)
    window.clear()
    batch.draw()

# Run the application
pyglet.app.run()