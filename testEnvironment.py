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
MU = 13 #1 meter = n pixels (based on vehicle width) (METER UNIT)
TOP_SPEED = MU*150 #48km/h  pixels per second (15)
TOP_TURNING_SPEED = MU*13 #13
ACCELERATION = MU * 5
DECELERATION = ACCELERATION * 3
TURNING_ACCEL = MU * 15 #15 #assume turning acceleration of 5m/s
FRICTION = MU * 3 #FRICTION coefficient
TURNING_FRICTION = MU * 20 #how quickly rotation stops
CAR_LENGTH = MU*4.2 #4.2 meters, pixel per meter 21:9 ratio standard
CAR_WIDTH = MU*1.8  #= 1.8 meters to pixels is 24, therfore 1 meter = 13.33 pixels
SPEED_UP = 1 #speed up simulation

INITIAL_VELOCITY = 300 #initial velocity

vAgents = [
    VehicleAgent(x=(WINDOW_WIDTH*0.1), y=WINDOW_HEIGHT/2, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2, isControlled=False)
]
vAgents[0].velocity = INITIAL_VELOCITY #initial velocity
vAgents[0].deg_angle = 20 #initial angle
#/////////////////////////////////////////////////////////////////////////////////////////#////////////////////////////////////////////////////////////////////////////////////////       
ROAD_WIDTH = CAR_WIDTH * 4 #should be roughly DOUBLE a saloon width
roads = []
road = RoadTile(start_x=(WINDOW_WIDTH*0.1), start_y=WINDOW_HEIGHT/2, end_x=(WINDOW_WIDTH*0.9), end_y=WINDOW_HEIGHT/2, width=ROAD_WIDTH, color=(50, 50, 50), batch=roadBatch)
roads.append(road)
 

#HITCHECKPOINT = False


#dt is the time elapsed since last call
def update(dt): #ensuring consistent framerate and game logic to be frame-rate indepependent
    dt = dt * SPEED_UP #speed up simulation

    global keys_pressed, ACCELERATION, DECELERATION, TURNING_ACCEL, FRICTION, TURNING_FRICTION #HITCHECKPOINT
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
            a.velocity = 0
        if a.shape.y >= window.height or a.shape.y <= 0:
            a.velocity = 0

        if abs(a.velocity) < 0.1: #clamp minimum speed
            a.velocity = 0
        if abs(a.turning_speed) < 0.1:
            a.turning_speed = 0
        if abs(a.velocity) > TOP_SPEED*2: #clamp top speed (max drag)
            a.velocity = TOP_SPEED*2*np.sign(a.velocity)
        if abs(a.turning_speed) > TOP_TURNING_SPEED: #clamp top turning speed
           a.turning_speed = TOP_TURNING_SPEED*2*np.sign(a.turning_speed)

        # Check if the vehicle is on the road
        a.changeAnchor(a.center_anchor_x, a.center_anchor_y) #center anchor position temporarily
        for road in roads:
            if road.is_on_road(a.shape.position): #not in road
                a.shape.color = (255, 0, 0)  # Stop the vehicle if it's off the road
            else:
                a.shape.color = (50, 225, 30)
            #if road.is_in_checkpoint(a.shape.position):
                #HITCHECKPOINT = True

            for i in range(a.num_vision_lines):
                if not road.line_end_on_road(a.Lines[i].x2, a.Lines[i].y2):
                    #decrease the vision length until it is NOT on the road
                    a.lineLengths[i] = max(0, a.lineLengths[i] - 1)
                    a.updateLines()
                elif not road.line_end_on_road(a.Lines[i].x, a.Lines[i].y):
                    a.lineLengths[i] = 0
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



















#/////////////////////////////////////////////////////////////////////////////////////////#/////////////////////////////////////////////////////////////////////////////////////////
### HYPERPARAMETERS
GAMMA = 0.99 #discount factor for future rewards (higher means more long-term oriented)
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001
UPDATE_FREQUENCY = 0.1 #how often agent direction is updated (lower is more frequent)
MAX_EP_LENGTH = 5/UPDATE_FREQUENCY
ACTOR_EPSILON = 0.5 #exploration rate chance
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.99



def calculate_reward(current_state):
   # global HITCHECKPOINT
    reward = 0
    max_distance = 134 #max length of vision line
    for i in range(6):  #for every vision line
        distance = current_state[i+5]
        if distance <= 0:
            reward += -5
        else:
            reward += ((distance-(max_distance/2)) / max_distance) * 10 
    #if HITCHECKPOINT:
        #HITCHECKPOINT = False
       # reward += 10
    return reward

def resetEnvironment():
    global vAgents, global_states, global_actions, global_rewards, current_len, INITIAL_VELOCITY
    vAgents = [
    VehicleAgent(x=(WINDOW_WIDTH*0.1), y=WINDOW_HEIGHT/2, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2, isControlled=False),
    ]
    vAgents[0].velocity = INITIAL_VELOCITY #initial velocity
    vAgents[0].deg_angle = 20 #initial angle
    global_states = []
    global_actions = []
    global_rewards = []
    current_len = 0

def update_direction(dt): #operates on longer timesteps than animation update
    global current_len, global_states
    current_len = current_len + 1
    for a in vAgents:
        currentState = [a.shape.x, a.shape.y, a.deg_angle, a.velocity, a.turning_speed, a.lineLengths[0], a.lineLengths[1], a.lineLengths[2], a.lineLengths[3], a.lineLengths[4], a.lineLengths[5]]
        #store current state
        global_states.append(currentState)
        action = pick_action(currentState, actor_func) #process state and get action out of it
        if not a.isControlled:
            a.updateDirection(action) #do the determined action
        global_actions.append(action)
        current_reward = calculate_reward(currentState)
        global_rewards.append(current_reward)#add the rewards to list for the timestep

    if current_len >= MAX_EP_LENGTH or current_reward < -1000: #off road
        update_agent() #get cumulative rewards
        resetEnvironment() #should reset the environment back to the beginning


#ACTOR_CRITIC_NETWORK.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTOR_INPUTS = 11 #based on test environment

class ActorNet(nn.Module): #our ACTOR network
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden = nn.Linear(ACTOR_INPUTS, hidden_dim) #input layer has 8 units, hidden layer has 16 units
        self.output = nn.Linear(hidden_dim, 5) #output layer has 2 units
        self.epsilon = ACTOR_EPSILON #epsilon value
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

global_decisions = []

def pick_action(state, actor):
    if np.random.rand() < ACTOR_EPSILON:
        action = np.random.choice(range(actor.output.out_features))
        print("ACTION TAKEN: ", action)
        return action
    with torch.no_grad():
        state_batch = np.expand_dims(state, axis=0) #add extra dimension to state for input to NN
        state_batch = torch.tensor(state_batch, dtype=torch.float).to(device) #expanded state converted to PyTorch tensor and moved to GPU or CPU
        #print("STATE BATCH: ", state_batch)
        logits = actor_func(state_batch) #pass state tensor through actor network to get logits
        #print("LOGITS: ", logits)
        logits = logits.squeeze(dim=0)
        probs = F.softmax(logits, dim=-1) #convert logits to probabilities
        a = torch.multinomial(probs, num_samples=1) #sample an action from probabilities
        print("ACTION TAKEN: ", a.tolist(), probs)
        #print(a.tolist())
        return a.tolist()[0] #return the action
        #recall that action space is 4 so will be from 0 to 3
        #[0, 0, 0, 0] = 0 -> nothing
        #[1, 0, 0, 0] = 1 -> left
        #[0, 1, 0, 0] = 2 -> right
        #[0, 0, 1, 0] = 3 -> backwards
        #[0, 0, 0, 1] = 4 -> forwards


optActor = torch.optim.AdamW(actor_func.parameters(), lr=ACTOR_LEARNING_RATE) #set up AdamW optimisers for Value & Actor networks
optCritic = torch.optim.AdamW(value_func.parameters(), lr=CRITIC_LEARNING_RATE) #these are the OPTIMISER ALGORITHMS to find best weights
#optcritic has faster learning rate
reward_records = [] #keep record for analysis later
global_states = []
global_actions = []
global_rewards = []
current_len = 0
episode = 0

def update_agent():
    global episode
    #get cumulative rewards by iterating backwards through rewards and applying discount factor
    global global_states, global_actions, global_rewards, reward_records, ACTOR_EPSILON
    cum_rewards = np.zeros_like(global_rewards)
    reward_len = len(global_rewards)
    for j in reversed(range(reward_len)):
        cum_rewards[j] = global_rewards[j] + (cum_rewards[j+1]*GAMMA if j+1 < reward_len else 0)
    
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
    log_probs = -F.cross_entropy(logits, global_actions, reduction="none")
    pi_loss = -log_probs * advantages
    pi_loss.sum().backward()
    optActor.step()
    

    
    #OUTPUT total rewards in current episode
    episode += 1
    print("Run episode{} with rewards {}".format(episode, sum(global_rewards)))
    reward_records.append(sum(global_rewards))

    #decay epsilon rate
    ACTOR_EPSILON  = max(MIN_EPSILON, ACTOR_EPSILON * EPSILON_DECAY)
    
    if episode > 1000:
        print("\nDone")
        pyglet.app.exit()




pyglet.clock.schedule_interval(update_direction, UPDATE_FREQUENCY/SPEED_UP)  # Update direction every xx seconds
pyglet.clock.schedule(update)

pyglet.clock.schedule(update_user_direction)

pyglet.app.run()        

import matplotlib.pyplot as plt
average_reward = []
for idx in range(len(reward_records)):
    if idx < 50:
        avg_list = reward_records[:idx+1]
    else:
        avg_list = reward_records[idx-49:idx+1]
    average_reward.append(np.average(avg_list))


plt.plot(reward_records, label='Reward per Episode')
plt.plot(average_reward, label='Average Reward (Last 50 Episodes)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards and Average Rewards Over Episodes')
plt.legend()
plt.show()


# Display plot
plt.show()