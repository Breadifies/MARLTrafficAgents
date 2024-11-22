import pyglet
import numpy as np
import math
from pyglet.window import key, mouse
from pyglet import shapes
from pyglet import gl #graphics drawing done via GPU than CPU, more optimised
from vehicle_Agent import VehicleAgent, RoadTile
import torch
import torch.nn as nn
from torch.nn import functional as F



window = pyglet.window.Window(width=800, height=600, vsync=True)  # vsync, eliminate unnecessary variables
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
SPEED_UP = 1 #speed up simulation

vAgents = [
    VehicleAgent(x=450, y=450, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2, isControlled=True),
    VehicleAgent(x=300, y=450, width=CAR_LENGTH, height=CAR_WIDTH, color=(50, 225, 30), batch1=batch, batch2=batch2)
]
#/////////////////////////////////////////////////////////////////////////////////////////#////////////////////////////////////////////////////////////////////////////////////////       
ROAD_WIDTH = CAR_WIDTH * 2 #should be roughly DOUBLE a saloon width
roads = []
road = RoadTile(start_x=(WINDOW_WIDTH*0.1), start_y=WINDOW_HEIGHT/2, end_x=(WINDOW_WIDTH*0.9), end_y=WINDOW_HEIGHT/2, width=ROAD_WIDTH, color=(50, 50, 50), batch=roadBatch)
roads.append(road)
 

#dt is the time elapsed since last call
def update(dt): #ensuring consistent framerate and game logic to be frame-rate indepependent
    dt = dt * SPEED_UP #speed up simulation

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

def update_user_direction(dt):
    for a in vAgents:
        if a.isControlled:
            a.current_direction = keys_pressed

@window.event
def on_draw():
    gl.glClearColor(240 / 255.0, 240 / 255.0, 240 / 255.0, 1.0) #colour light grey
    window.clear()
    roadBatch.draw()
    batch2.draw() #vision lines
    batch.draw()

def update_direction(dt): #operates on longer timesteps than animation update
    global current_len
    current_len = current_len + 1
    for a in vAgents:
        currentState = [a.shape.x, a.shape.y, a.deg_angle, a.velocity, a.turning_speed, a.lineLengths[0], a.lineLengths[1], a.lineLengths[2]]
        #store current state
        global_states.append(currentState)
        action = pick_action(currentState) #process state and get action out of it
        if not a.isControlled:
            a.updateDirection(action) #do the determined action
        global_actions.append(action)
        global_rewards.append(0) #no rewards implemented yet

    print(current_len)
    if current_len >= MAX_EP_LENGTH:
        update_episode() #get cumulative rewards
        exit() #should reset the environment back to the beginning


pyglet.clock.schedule_interval(update_direction, 0.3/SPEED_UP)  # Update direction every xx seconds
pyglet.clock.schedule(update)
pyglet.clock.schedule(update_user_direction)











#ACTOR_CRITIC_NETWORK.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTOR_INPUTS = 8 #based on test environment
class ActorNet(nn.Module): #our ACTOR network
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden = nn.Linear(ACTOR_INPUTS, hidden_dim) #input layer has 8 units, hidden layer has 16 units
        self.output = nn.Linear(hidden_dim, 5) #output layer has 2 units
    def forward(self, s): #the NN forward pass
        outs = self.hidden(s)
        outs = F.relu(outs) #applies ReLU activation
        logits = self.output(outs) #computes logits -> passed through e.g. a softmax function to convert to probabilities
        return logits

class ValueNet(nn.Module): #our VALUE network
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden = nn.Linear(ACTOR_INPUTS, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1) #1 unit to predict value of the state
    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        value = self.output(outs)
        return value

#instantiate the actor and value networks and move to GPU if available (CPU otherwise)
actor_func = ActorNet().to(device) 
value_func = ValueNet().to(device)

def pick_action(state):
    with torch.no_grad():
        state_batch = np.expand_dims(state, axis=0) #add extra dimension to state for input to NN
        state_batch = torch.tensor(state_batch, dtype=torch.float).to(device) #expanded state converted to PyTorch tensor and moved to GPU or CPU
        logits = actor_func(state_batch) #pass state tensor through actor network to get logits
        logits = logits.squeeze(dim=0)
        probs = F.softmax(logits, dim=-1) #convert logits to probabilities
        a = torch.multinomial(probs, num_samples=1) #sample an action from probabilities
        return a.tolist()[0] #return the action
        #recall that action space is 4 so will be from 0 to 3
        #[0, 0, 0, 0] = 0
        #[1, 0, 0, 0] = 1
        #[0, 1, 0, 0] = 2
        #[0, 0, 1, 0] = 3
        #[0, 0, 0, 1] = 4

reward_records = []
optActor = torch.optim.AdamW(actor_func.parameters(), lr=0.001) #set up AdamW optimisers for Value & Actor networks
optCritic = torch.optim.AdamW(value_func.parameters(), lr=0.001) #these are the OPTIMISER ALGORITHMS to find best weights

global_states = []
global_actions = []
global_rewards = []
MAX_EP_LENGTH = 50
current_len = 0
gamma = 0.99 #discount factor for future rewards (later timesteps)

def update_episode():
    #get cumulative rewards by iterating backwards through rewards and applying discount factor
    global global_states, global_actions, global_rewards
    cum_rewards = np.zeros_like(global_rewards)
    reward_len = len(global_rewards)
    for j in reversed(range(reward_len)):
        cum_rewards[j] = global_rewards[j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0)
    
    #Train (optimise parameters)
    #optimisng value LOSS (critic)
    optCritic.zero_grad()
    global_states = torch.tensor(global_states, dtype=torch.float).to(device)
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
    values = value_func(global_states)
    values = values.squeeze(dim=1)
    vf_loss = F.mse_loss(
        values,
        cum_rewards,
        reduction="none")
    vf_loss.sum().backward()
    optCritic.step()

    #otimising policy loss (Actor)
    with torch.no_grad():
        values = value_func(global_states)
    optActor.zero_grad()
    global_actions = torch.tensor(global_actions, dtype=torch.int64).to(device)
    advantages = cum_rewards - values
    logits = actor_func(global_states)
    log_probs = F.cross_entropy(logits, global_actions, reduction="none")
    pi_loss = -log_probs * advantages
    pi_loss.sum().backward()
    optActor.step()
    
    #OUTPUT total rewards in current episode
    print("Run episode{} with rewards {}".format(0, 0), end="\r")
    reward_records.append(sum(global_rewards))


pyglet.app.run()        