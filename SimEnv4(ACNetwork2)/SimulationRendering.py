import pyglet
import numpy as np
import math
from pyglet import gl
from shapeClasses import VehicleAgent, RoadTile, PedestrianAgent

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import namedtuple

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

SPEED_UP = 1 #speed up simulation (do not abouse, accumulate numerical errors, update missing and accuracy of physics)
###################################################################


###################CONSTRUCTING ROADS##############################
# center_x = WINDOW_WIDTH // 2
# center_y = WINDOW_HEIGHT // 2
# radius = 150
# num_tiles = 10
# angle_increment = 2 * math.pi / num_tiles
startProads = [
#     RoadTile(
#         start_x=center_x+ radius * math.cos(angle_increment * i),
#         start_y=center_y + radius * math.sin(angle_increment * i),
#         end_x=center_x + radius * math.cos(angle_increment * (i + 1)),
#         end_y=center_y + radius * math.sin(angle_increment * (i + 1)), 
#         width=ROAD_WIDTH,
#         color=(50, 50, 50),
#         batch1=batch1,
#         batch2 = batch2
#     )
#     for i in range(num_tiles)
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
    RoadTile(start_x = 650, start_y = WINDOW_HEIGHT//2, end_x = 800, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2)

]
###################################################################


###################CONSTRUCTING AGENTS#############################
startPvAgents = [
    # VehicleAgent(x=400, y=150, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2),
    # VehicleAgent(x=400, y=450, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2),
    # VehicleAgent(x=250, y=310, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2),
    # VehicleAgent(x=550, y=310, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch2)
    VehicleAgent(x=120, y=300, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch_),
    # VehicleAgent(x=50, y=300, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch_),
    # VehicleAgent(x=100, y=300, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch_),
    # VehicleAgent(x=150, y=300, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch_),
    # VehicleAgent(x=200, y=300, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch_),
    # VehicleAgent(x=250, y=300, width=CAR_LENGTH, height=CAR_WIDTH, color=(200, 225, 90), batch1=batch, batch2=batch_)
    ]
initVel = [0]*len(startPvAgents)
initAngle = [0]*len(startPvAgents)
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

#HIT_CHECKPOINT = False
###############PHYSICS RENDERING###################################
def update(dt):
    dt = dt * SPEED_UP
    global ACCELERATION, DECELERATION, FRICTION, TOP_SPEED, TOP_REV_SPEED
    for p in pAgents:
        p.move(dt*0.35) #move towards target
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
    
        a.changeAnchor(a.center_anch_x, a.turn_anch_y) 

        for p in pAgents:
            #check for vision line distance
            for i in range(a.num_vision_lines):
                if p.line_end_on_ped(a.Lines[i].x2, a.Lines[i].y2):
                    a.lineLengths[i] = max(0, a.lineLengths[i] - 1)
                elif (a.lineLengths[i] < a.maxLen):
                    a.lineLengths[i] = min(a.maxLen, a.lineLengths[i] + 1)
                    

        a.updateLines() 
        a.changeAnchor(a.turn_anch_x, a.turn_anch_y) 

        #rotate vehicle
        a.deg_angle = (a.deg_angle - changedAngle)%360
        car_angle = math.radians(a.deg_angle)
        a.shape.x += a.velocity * dt * math.cos(car_angle)
        a.shape.y += a.velocity * dt * math.sin(car_angle)
        a.shape.rotation = -a.deg_angle


        # for road in roads:
        #     if road.passed_checkpoint(a.shape.position):
        #         print("PASSED CHECKPOINT")
        #         HIT_CHECKPOINT = True

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


###################EPISODE-RENDER##################################
UPDATE_FREQUENCY = 0.05 #HOW OFTEN AGENT UPDATES ITS STATE AND ACTION #0.05
MAX_EP_LENGTH = 5/UPDATE_FREQUENCY 
MAX_EPS = 1000 #run how many eps.
#GREEDY EXPLORATION
STARTING_EPSILON = 0.2 #0.5
ACTOR_EPSILON = STARTING_EPSILON
MIN_EPSILON = 0.01 #0.05
EPSILON_DECAY = 0.95 #0.99

num_eps = 0 
ep_len = 0

running_reward = 0
episode_rewards = [] #maintained over every episode (tracking purposes)
running_averages = [] #same but for running_average
ep_reward = 0 #cumulative reward for episode

runningS_reward = 0
episodeS_rewards = [] #maintained over every episode (tracking purposes)
runningS_averages = [] #same but for running_average
safety_reward = 0


#TEMP SHOWCASING REWARD FUNCTION ACTIVATION
efficiency_rewards = []
safety_rewards = []


def calculate_reward(current_state):
    #global HIT_CHECKPOINT
    # if HIT_CHECKPOINT:
    #     reward += 10
    #     HIT_CHECKPOINT = False
    reward1 = 0
    reward1 += current_state[3]
    r1W = 0.5
    reward2 = 0
    reward2 += current_state[4]
    r2W = 0.5
    if reward2 == 1:#if no obstacles in vision
        r1W = 1
        r2W = 0
    else:
        r1W = 0.5
        r2W = 0.5
        reward2 = min(-10* ((current_state[4] - 1) ** 2) + 1, 1)#adjusted reward range (-10, 1), quadratic function to penalise distances away from 1
        print(f"CONTACT: {reward2}")
    
    #TEMP FOR REWARD SHOWCASE
    efficiency_rewards.append(reward1)
    safety_rewards.append(reward2)

    reward1 = r1W*reward1 + r2W*reward2 #linear scalarisation
    return [reward1, reward2]

def envReset():
    global vAgents, pAgents, ep_len, initVel, roads, num_eps
    global ep_reward, running_reward, episode_rewards, running_averages
    global safety_reward, runningS_reward, episodeS_rewards, runningS_averages
    global ACTOR_EPSILON
    print("RESET")
    #vAgents = create New agents function ()
    vAgents.clear()
    pAgents.clear()
    for item in startPvAgents:
        vAgents.append(item.__deepcopy__(batch1, batch2))
    for item in startPAgents:
        pAgents.append(item.__deepcopy__(batch1))
    for i, a in enumerate(vAgents):
        a.velocity = initVel[i]
        a.shape.rotation = initAngle[i]
    ep_len = 0


    #update cumulative reward
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward #exponential moving average
    episode_rewards.append(ep_reward)
    running_averages.append(running_reward)

    #update safety reward (vision line distance)
    runningS_reward = 0.05 * safety_reward + (1 - 0.05) * runningS_reward #exponential moving average
    episodeS_rewards.append(safety_reward)
    runningS_averages.append(runningS_reward)

    #performing backpropagation
    finish_episode()

    num_eps += 1
    #LOGGING
    print("Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}".format(num_eps, ep_reward, running_reward))
    ep_reward = 0
    print("Also with Safety Score: {:.2f}\tAverage reward: {:.2f}".format(safety_reward, runningS_reward))
    safety_reward = 0


    #GREEDY EPSILON UPDATE
    print("ACTOR EPSILON: ", ACTOR_EPSILON)
    ACTOR_EPSILON = max(MIN_EPSILON, ACTOR_EPSILON*EPSILON_DECAY)

    if num_eps >= 1: #MAX_EPS
        print("Completed training... DONE")
        pyglet.app.exit()


def envStep(dt):
    global ep_len, ep_reward, safety_reward
    currentState = []
    current_reward = 0
    dist_reward = 0 #vision line dist
    ep_len += 1

    for a in vAgents:
        visionLength = 0
        for i in range(a.num_vision_lines): 
            visionLength += a.lineLengths[i]
        visionLength = (visionLength/a.num_vision_lines) #get average visionLengths from all lines

        currentState = [a.shape.x, a.shape.y, a.deg_angle, a.velocity, visionLength] 
        #WILL CHANGE TO ARRAY OF ARRAYS FOR MULTI-AGENT
        #NORMALISE STATES IMPORTANT
        nomCState = [currentState[0]/WINDOW_WIDTH, currentState[1]/WINDOW_HEIGHT, currentState[2]/360, currentState[3]/(TOP_SPEED + TOP_REV_SPEED), currentState[4]/(a.maxLen)]
        #print(f"NORM STATE {nomCState}")
        current_reward, dist_reward = calculate_reward(nomCState)
        a.updateDirection(selectAction(nomCState))
    #cross reference to MAIN from actor_critic.py
    model.rewards.append(current_reward)
    ep_reward += current_reward
    safety_reward += dist_reward

    if ep_len >= MAX_EP_LENGTH: #if exceeds max length or last part of road passed
        envReset()

def selectAction(currentState):
    currentState = np.array(currentState)
    action = select_action(currentState) #A-C func
    #print(f"ACTION: {action}")
    action = 2 #forwards automatically
    return action #forwards
###################################################################

#######A-C_NETWORK##########################
INPUTS = 5 #number of state inputs (x, y, angle, velocity, visionLength)

OPTIM_LR = 0.006 #optimiser learning rate # 0.006
GAMMA = 0.90 #discount factor for future rewards #0.99

#ONE DNN for both actor and critic
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(INPUTS, 128) #input fully connected layer
        self.action_head = nn.Linear(128, 3) #actor output (3 OUTPUTS)
        self.value_head = nn.Linear(128, 1) #critic output
        #action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x): 
        #forward pass for both actor and critic
        x = F.relu(self.affine(x))
        #actor output chooses action from state, probability distribution therefore softmax
        action_prob = F.softmax(self.action_head(x), dim=-1)
        #critic output evaluates being in state
        state_values = self.value_head(x)
        return action_prob, state_values

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=OPTIM_LR)
eps = np.finfo(np.float32).eps.item() #epsilon, const to add numerical stability

SavedAction = namedtuple('SavedAction', ['log_prob', 'value']) #store log prob of selected action and state value when action was taken -> used to compute loss for policy and value function (log_prob is used due to stability and policy gradient reasons)

def select_action(state):

    if np.random.rand() < ACTOR_EPSILON: #GREEDY EPSILON EXPLORATION
        action = np.random.choice([0, 1, 2])
        #print("RANDOM ACTION TAKEN: ", action)
        return action

    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    #create categorical distr over list of probabilities
    
    m = Categorical(probs)
    action = m.sample() #sampling allows us to select actions basd on probabilities
    print(f"PROBS : {probs}, ACTION: {action}")
    #save action to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item() #forwards or backwards

def finish_episode():
    #code for training network -> calculate loss for actor and critic and then backpropagate
    R = 0 #reward at timestep
    saved_actions = model.saved_actions
    policy_losses = [] #save actor losses
    value_losses = [] #save critic losses
    returns = [] #save true values of returns

    for r in model.rewards[::-1]: #calculate discounted  (reverse order of timesteps)
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps) #normalization of returns tensor -> to calculate advantage and help ensure consistent updates
    
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item() #advantage calculated as actual return (R) - critic's estimated value (value.item())
        policy_losses.append(-log_prob * advantage) #actor loss (gradient ascent)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]))) #critic loss via L1 smooth loss (Huber loss)
    optimizer.zero_grad() #reset gradients
    #TD learning, so sum up all policy and value losses calculated
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    #perform backpropagation
    loss.backward()
    optimizer.step()
    #reset buffers for next ep
    del model.rewards[:]
    del model.saved_actions[:]

############MAIN#######################################

pyglet.clock.schedule_interval(envStep, UPDATE_FREQUENCY/SPEED_UP)
pyglet.clock.schedule(update)#call update function according to system refresh rate

pyglet.app.run()

import matplotlib.pyplot as plt

# Plotting Efficiency and Safety Rewards
plt.plot(efficiency_rewards, label='Efficiency Reward', color='b') 
plt.plot(safety_rewards, label='Safety Reward', color='r')

# Adding labels and title
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Efficiency and Safety Rewards Over Episodes')

# Displaying the legend
plt.legend()

# Show the plot
plt.show()











# # Plotting the main data
# plt.plot(episode_rewards, label='Reward per Episode', color='b') 
# plt.plot(running_averages, label='Average Reward (Last 50 Episodes)', color='g')

# plt.plot(episodeS_rewards, label='EpisodeS Rewards', color='gray', alpha=0.5) 
# plt.plot(runningS_averages, label='RunningS Averages', color='orange', alpha=0.5)  

# # Adding labels and title
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title('Rewards and Average Rewards Over Episodes')

# # Adding the hyperparameters as a text box in the plot
# hyperparameters = f"""
# STARTING_EPSILON: {STARTING_EPSILON}
# EPSILON_DECAY: {EPSILON_DECAY}
# OPTIM_LR: {OPTIM_LR}
# GAMMA: {GAMMA}
# """

# # Position the text box (adjust the coordinates as needed)
# plt.text(0.96, 0.5, hyperparameters, transform=plt.gca().transAxes,
#          fontsize=5, verticalalignment='top', horizontalalignment='right',
#          bbox=dict(facecolor='white', alpha=0.4, edgecolor='black', boxstyle='round,pad=0.3'))


# # Displaying the legend
# plt.legend()

# # Show the plot
# plt.show()

#[ACTOR_EPSILON, EPSILON_DECAY, OPTIM_LR, GAMMA]