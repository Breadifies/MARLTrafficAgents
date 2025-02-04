import pyglet
import numpy as np
import math
from pyglet.window import key
from pyglet import shapes
from pyglet import gl
from shapeClasses import VehicleAgent, RoadTile
import torch
import torch.nn as nn
from torch.nn import functional as F

window = pyglet.window.Window(width=800, height=600, vsync=True)
WINDOW_WIDTH, WINDOW_HEIGHT = window.get_size()

batch = pyglet.graphics.Batch()
batch2 = pyglet.graphics.Batch()

####################USER_INPUT#####################################
keys_pressed = [0]*2
@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.DOWN:
        keys_pressed[0] = 1
    elif symbol == key.UP:
        keys_pressed[1] = 1
@window.event
def on_key_release(symbol, modifiers):
    if symbol == key.DOWN:
        keys_pressed[0] = 0
    elif symbol == key.UP:
        keys_pressed[1] = 0
##################################################################


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
roads = []
center_x = WINDOW_WIDTH // 2
center_y = WINDOW_HEIGHT // 2
radius = 150
num_tiles = 10
angle_increment = 2 * math.pi / num_tiles
roads = [
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
    RoadTile(start_x = 100, start_y = WINDOW_HEIGHT//2, end_x = 200, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch, batch2=batch2),
    RoadTile(start_x = 200, start_y = WINDOW_HEIGHT//2, end_x = 300, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch, batch2=batch2),
    RoadTile(start_x = 300, start_y = WINDOW_HEIGHT//2, end_x = 400, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch, batch2=batch2),
    RoadTile(start_x = 400, start_y = WINDOW_HEIGHT//2, end_x = 500, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch, batch2=batch2),
    RoadTile(start_x = 500, start_y = WINDOW_HEIGHT//2, end_x = 600, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch, batch2=batch2),
    RoadTile(start_x = 600, start_y = WINDOW_HEIGHT//2, end_x = 700, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch, batch2=batch2)

]
###################################################################


###################CONSTRUCTING AGENTS#############################
vAgents = [
    # VehicleAgent(x=400, y=150, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2),
    # VehicleAgent(x=400, y=450, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2),
    # VehicleAgent(x=250, y=310, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2),
    # VehicleAgent(x=550, y=310, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2)
    VehicleAgent(x=120, y=300, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2)
    ]
vAgents[0].velocity = 500
initVel = [0 * len(vAgents)]
initAngle = [0 * len(vAgents)]
###################################################################

HIT_CHECKPOINT = False
###############PHYSICS RENDERING###################################
def update(dt):
    dt = dt * SPEED_UP
    global keys_pressed, ACCELERATION, DECELERATION, FRICTION, TOP_SPEED, TOP_REV_SPEED, HIT_CHECKPOINT
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

        for road in roads:
            if road.passed_checkpoint(a.shape.position):
                HIT_CHECKPOINT = True

        # for v in vAgents: #check distance of vision lines from other vehicles
        #     intersect = False #check if its intersecting WITH ANY object
        #     if not (v == a):
        #         for i in range(a.num_vision_lines):
        #             if not v.line_end_on_agent(a.Lines[i].x2, a.Lines[i].y2):
        #                 a.lineLengths[i] = min(a.maxLen, a.lineLengths[i] + 1)
                        
        #             else:
        #                 intersect = True
        #                 a.lineLengths[i] = max(0, a.lineLengths[i] - 1)
        #     if intersect: #already intersecting with an agent, don't check others
        #         break
        # a.updateLines()
        
        if abs(a.velocity) < 0.1: #clamp minimum speed
            a.velocity = 0
        if a.velocity >= 0:
            if a.velocity > TOP_SPEED: #clamp top speed (max drag)
                a.velocity = TOP_SPEED*np.sign(a.velocity)
        else:
            if a.velocity < -TOP_REV_SPEED:
                a.velocity = TOP_REV_SPEED*np.sign(a.velocity)


def update_user_direction(dt):
    for a in vAgents:
        if a.isControlled:
            a.current_direction = keys_pressed
###################################################################


###################DRAWTOSCREEN####################################
@window.event
def on_draw():
    gl.glClearColor(1, 1, 1, 1.0)
    window.clear()
    batch.draw()
    batch2.draw()
    
###################################################################


####################ACTOR-CRITIC-NETWORK###########################
GAMMA = 0.90
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001
UPDATE_FREQUENCY = 0.05
ACTOR_EPSILON = 0.5
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.99
MAX_EP_LENGTH = 5/UPDATE_FREQUENCY


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTOR_INPUTS = 4 #a.shape.x, a.shape.y, a.deg_angle ,shape.a.velocity
class ActorNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden = nn.Linear(ACTOR_INPUTS, hidden_dim)
        self.output = nn.Linear(hidden_dim, 3) #no action, backwards, forwards
        self.epsilon = ACTOR_EPSILON
    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        logits = self.output(outs)
        return logits
class ValueNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden = nn.Linear(ACTOR_INPUTS, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        value = self.output(outs)
        return value
    
actor_func = ActorNet().to(device)
value_func = ValueNet().to(device)

def pick_action(state, actor):
    if np.random.rand() < ACTOR_EPSILON:
        action = np.random.choice(range(actor.output.out_features))
        print("ACTION TAKEN: ", action)
        return action
    with torch.no_grad():
        state_batch = np.expand_dims(state, axis=0)
        state_batch = torch.tensor(state_batch, dtype=torch.float).to(device)
        logits = actor_func(state_batch)
        logits = logits.squeeze(dim=0)
        probs = F.softmax(logits, dim=-1)
        a = torch.multinomial(probs, num_samples=1)
        print("ACTION TAKEN: ", a.tolist(), probs)
        return a.tolist()[0]
    
optActor = torch.optim.AdamW(actor_func.parameters(), lr=ACTOR_LEARNING_RATE)
optCritic = torch.optim.AdamW(value_func.parameters(), lr=CRITIC_LEARNING_RATE)

reward_records = []
agent_states = []
agent_actions = []
agent_rewards = []
current_len = 0
episode = 0

def update_agent():
    global agent_states, agent_actions, agent_rewards, episode, ACTOR_EPSILON
    cum_rewards = np.zeros_like(agent_rewards)
    reward_len = len(agent_rewards)
    for j in reversed(range(reward_len)):
        cum_rewards[j] = agent_rewards[j] + (cum_rewards[j+1]*GAMMA if j+1<reward_len else 0)
    optCritic.zero_grad()
    agent_states = torch.tensor(agent_states, dtype=torch.float).to(device)
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
    values = value_func(agent_states)
    values = values.squeeze(dim=1)
    vf_loss = F.mse_loss(values, cum_rewards, reduction="none")
    vf_loss.sum().backward()
    optCritic.step()
    with torch.no_grad():
        values = value_func(agent_states)
    optActor.zero_grad()
    agent_actions = torch.tensor(agent_actions, dtype=torch.int64).to(device)
    advantages = cum_rewards - values #CUM_REWARDS - VALUES
    logits = actor_func(agent_states)
    log_probs = -F.cross_entropy(logits, agent_actions, reduction="none")
    pi_loss = -log_probs*advantages
    pi_loss.sum().backward()
    optActor.step()

    episode += 1
    print("Run episode{} with rewards {}".format(episode, sum(agent_rewards)))
    print("ACTOR EPSILON: ", ACTOR_EPSILON)
    reward_records.append(sum(agent_rewards))
    ACTOR_EPSILON = max(MIN_EPSILON, ACTOR_EPSILON*EPSILON_DECAY)

    if episode > 4000:
        print("\nDONE")
        pyglet.app.exit()

###################################################################


###################EPISODE-RENDER##################################
def calculate_reward(current_state):
    global HIT_CHECKPOINT
    reward = 0
    if HIT_CHECKPOINT:
        reward += 100
        HIT_CHECKPOINT = False
    if vAgents[0].velocity <=0:
        reward -= 1
    reward -= 1
    return reward

def resetEnvironment():
    global vAgents, agent_states, agent_actions, agent_rewards, current_len, initVel, roads
    for road in roads:
        road.passed = False
    #vAgents = create New agents function ()
    vAgents = [VehicleAgent(x=120, y=300, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2)]
    for i, a in enumerate(vAgents):
        a.velocity = 500
        a.shape.rotation = initAngle[i]
    agent_states = []
    agent_actions = []
    agent_rewards = []
    current_len = 0

def update_direction(dt):
    global current_len, agent_states
    current_len = current_len + 1
    for a in vAgents:
        currentState = [a.shape.x, a.shape.y, a.deg_angle, a.velocity]
        agent_states.append(currentState)
        action = pick_action(currentState, actor_func)
        if not a.isControlled:
            a.updateDirection(action)
        agent_actions.append(action)
        current_reward = calculate_reward(currentState)
        agent_rewards.append(current_reward)
    if current_len >= MAX_EP_LENGTH or roads[5].passed: #if exceeds max length or last part of road passed
        update_agent()
        resetEnvironment()

###################################################################


pyglet.clock.schedule_interval(update_direction, UPDATE_FREQUENCY/SPEED_UP)
pyglet.clock.schedule(update)#call update function according to system refresh rate
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
plt.show()