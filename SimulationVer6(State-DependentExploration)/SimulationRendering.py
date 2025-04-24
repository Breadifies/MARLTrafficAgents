import pyglet
import numpy as np
import math
from pyglet import gl
from shapeClasses import VehicleAgent, RoadTile, PedestrianAgent
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions.normal import Normal
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
TOP_SPEED = MU*30#MU*30
TOP_REV_SPEED = TOP_SPEED/2
ACCELERATION = MU*70
DECELERATION = ACCELERATION * 1 #2.5 #braking force is much stronger than accel 
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
    # RoadTile(
    #     start_x=center_x+ radius * math.cos(angle_increment * i),
    #     start_y=center_y + radius * math.sin(angle_increment * i),
    #     end_x=center_x + radius * math.cos(angle_increment * (i + 1)),
    #     end_y=center_y + radius * math.sin(angle_increment * (i + 1)), 
    #     width=ROAD_WIDTH,
    #     color=(50, 50, 50),
    #     batch1=batch1,
    #     batch2 = batch2
    # )
    # for i in range(num_tiles)
    RoadTile(start_x = -400, start_y = WINDOW_HEIGHT//2, end_x = 150, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2),
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
    RoadTile(start_x = 650, start_y = WINDOW_HEIGHT//2, end_x = 3000, end_y = WINDOW_HEIGHT//2, width=ROAD_WIDTH, color=(50, 50, 50), batch1=batch1, batch2=batch2)
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
    # PedestrianAgent(x=450, y=400, target_x = 450, target_y = 200, radius=10, color=(0, 0, 100), batch=batch),
    # PedestrianAgent(x=300, y=400, target_x = 300, target_y = 200, radius=10, color=(0, 0, 100), batch=batch)
    PedestrianAgent(x=400, y=350, target_x = 400, target_y = 300, radius=10, color=(0, 0, 100), batch=batch)
]
###################################################################





#################INITIALISE###########################################
roads = startProads
vAgents = []
pAgents = []
for item in startPvAgents:
    vAgents.append(item.__deepcopy__(batch1, batch2))
for item in startPAgents:
    pAgents.append(item.__deepcopy__(batch1))
###################################################################





###############PHYSICS RENDERING###################################
def update(dt):
    dt = dt * SPEED_UP
    global ACCELERATION, DECELERATION, FRICTION, TOP_SPEED, TOP_REV_SPEED
    for index, p in enumerate(pAgents):
        if index == 0:
            p.move(dt*0.5) #move towards target
        if index == 1:
            p.move(dt*0.35)
        p.move(dt*0.35)

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
        
        a.changedAngle = 0 #final angle change post-successful orient
        if a.curRoad and not onFollow: #if on a road, check orientation
            if a.curRoad.is_on_follow(a.shape.position):
                onFollow = True
                a.increment = 0
            else: #not oriented correctly -> ROTATE
                a.increment += 5 #sweep angle outwards till orient found
                a.changeAnchor(a.turn_anch_x, a.turn_anch_y)#rotate based on c.o.r
                a.changedAngle += a.clockwise*a.increment
                a.shape.rotation += a.clockwise*a.increment
                a.changeAnchor(followAnchor, a.turn_anch_y) #reset anchor to follow anchor
                a.clockwise = -a.clockwise #rotate in opposite direction
    
        a.changeAnchor(a.center_anch_x, a.turn_anch_y) 


        #update vision cones
        checkContact = [False]*a.num_tris #represents each individual status of cone, resets to false each time
        for p in pAgents:
            coneStatus = a.is_on_cones(p.shape.x, p.shape.y) #returns a tuple of each vision cone's status
            for i in range(len(coneStatus)): #for every vision cone
                if coneStatus[i]:
                    checkContact[i] = True #IF ANY in contact then set that to true

        for i in range(a.num_tris):
            if checkContact[i]: #if any ped on cone   # getting False then true
                a.triLengths[i] = max(0, a.triLengths[i] - 3)
            elif (a.triLengths[i] < a.maxLenTri):
                a.triLengths[i] = min(a.maxLenTri, a.triLengths[i] + 3)

        # for p in pAgents:
        #     #check for vision line distance
        #     for i in range(a.num_vision_lines):
        #         if p.line_end_on_ped(a.Lines[i].x2, a.Lines[i].y2):
        #             a.lineLengths[i] = max(0, a.lineLengths[i] - 1)
        #         elif (a.lineLengths[i] < a.maxLen):
        #             a.lineLengths[i] = min(a.maxLen, a.lineLengths[i] + 1)    
                    
        a.updateTris()
        #a.updateCones()
        
        a.changeAnchor(a.turn_anch_x, a.turn_anch_y) 

        #rotate vehicle
        a.deg_angle = (a.deg_angle - a.changedAngle)%360
        car_angle = math.radians(a.deg_angle)
        a.shape.x += a.velocity * dt * math.cos(car_angle)
        a.shape.y += a.velocity * dt * math.sin(car_angle)
        a.shape.rotation = -a.deg_angle

        if abs(a.velocity) < 0.1: #clamp minimum speed
            a.velocity = 0
        if a.velocity >= 0:
            if a.velocity > TOP_SPEED: #clamp top speed (max drag)
                a.velocity = TOP_SPEED*np.sign(a.velocity)
        else:
            if a.velocity < -TOP_REV_SPEED: #clamp top reverse speed
                a.velocity = TOP_REV_SPEED*np.sign(a.velocity)

###################################################################










































###################EPISODE-RENDER####################################################
UPDATE_FREQUENCY = (1/60) #HOW OFTEN AGENT UPDATES ITS STATE AND ACTION #0.05
MAX_EP_LENGTH = 3.5/UPDATE_FREQUENCY  
MAX_EPS = 1000 #run how many eps.
#GREEDY EXPLORATION
STARTING_EPSILON = 0.5 #0.5
ACTOR_EPSILON = STARTING_EPSILON
MIN_EPSILON = 0.01 #0.05
EPSILON_DECAY = 0.95 #0.99
#DYNAMIC DECAY
DSTART_EPSILON = 1
DANGER_EPSILON = DSTART_EPSILON
D_MIN_EPSILON = 0.1
D_EPSILON_DECAY = 0.9995


#SOFTMAX EXPLORATION
STARTING_TEMP = 1.0
ACTOR_TEMP = STARTING_TEMP
MIN_TEMP = 0.01
TEMP_DECAY = 0.95


num_eps = 1
ep_len = 0

running_reward = 0
episode_rewards = [] #maintained over every episode (tracking purposes)
running_averages = [] #same but for running_average
ep_reward = 0 #cumulative reward for episode

runningS_reward = 0
episodeS_rewards = [] #maintained over every episode (tracking purposes)
runningS_averages = [] #same but for running_average
safety_reward = 0 


#TEMP SHOWCASING REWARD FUNCTION ACTIVATION PER EPISODE
efficiency_rewards = []
ep_safety_rewards = []
avg_ep_safety_rewards = []*200

saveModel = True
loadModel = False
useEpsilonGreedy = True
useSoftMax = False




def calculate_reward(current_state):
    inDanger = False
    r1W = 0.5
    r2W = 0.5
    reward1 = 0
    reward1 += current_state[3]
    reward2 = 0
    reward2 += current_state[4]
    if reward2 >= 1:#if no obstacles in vision
        r1W = 1
        r2W = 0
    elif reward2 > 0.5: #implement the piecewise-function
        r1W = 0.1 #0.01
        r2W = 0.9 #0.99
        if current_state[4] > 0.5: #implement > 0.2 piece-wise function
            reward2 = current_state[4]**0.2 # x^0.2
        elif current_state[4] > 0.05:
            reward2 = 1.94*current_state[4] - 0.1  # 4.8x - 0.24 
        else:   
            reward2 = 0 # 0 < x < 0.05
    else: #danger zone
        inDanger = True
        r1W = 0
        r2W = 1
        reward2 = 0 #must get out of danger zone immediately
    reward2 *= 1  
    #TEMP FOR REWARD SHOWCASE
    efficiency_rewards.append(reward1)
    ep_safety_rewards.append(reward2)
    utility = r1W*reward1 + r2W*reward2 #linear scalarisation
    #print(f"REWARD1: {reward1}, REW2: {reward2}")
    return [utility, reward2, inDanger]

def envReset():
    global vAgents, pAgents, ep_len, initVel, roads, num_eps
    global ep_reward, running_reward, episode_rewards, running_averages
    global safety_reward, runningS_reward, episodeS_rewards, runningS_averages
    global ACTOR_EPSILON
    global ACTOR_TEMP
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

    #SOFTMAX TEMP UPDATE
    print("ACTOR TEMP: ", ACTOR_TEMP)
    ACTOR_TEMP = max(MIN_TEMP, ACTOR_TEMP*TEMP_DECAY)

    if num_eps >= MAX_EPS: #MAX_EPS
        print("Completed training... DONE")
        if saveModel:
            save_model(model)
        pyglet.app.exit()

    statUpdates() #execute statistical updates

#check for if window is closed prematurely
def on_close():
    print("Window closed by user")
    if saveModel:
        save_model(model)
    pyglet.app.exit() 
window.push_handlers(on_close=on_close)

def envStep(dt):
    global ep_len, ep_reward, safety_reward
    currentState = []
    current_reward = 0
    dist_reward = 0 #vision line dist
    ep_len += 1

    for a in vAgents:
        visionLength = 0
        for i in range(a.num_tris): 
            visionLength += a.triLengths[i]
        visionLength = (visionLength/a.num_tris) #get average visionLengths from all lines

        currentState = [a.shape.x, a.shape.y, a.deg_angle, a.velocity, visionLength] 
        #WILL CHANGE TO ARRAY OF ARRAYS FOR MULTI-AGENT
        #NORMALISE STATES IMPORTANT
        nomVelocity = 0
        if currentState[3] > 0: #normalise velocity dependent on direction, ensure that max efficiency matches max safety
            nomVelocity = (currentState[3]/TOP_SPEED)/2 + 0.5
        else:
            nomVelocity = (currentState[3]/TOP_REV_SPEED)/2 + 0.5
        nomCState = [currentState[0]/WINDOW_WIDTH, currentState[1]/WINDOW_HEIGHT, currentState[2]/360, nomVelocity, currentState[4]/(a.maxLenTri)]
        current_reward, dist_reward, inDanger = calculate_reward(nomCState)
        if inDanger:
            a.shape.color = (255, 0, 0)
        else:
            a.shape.color = (200, 225, 90)
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
    #action = 2 #forwards automatically
    return action #forwards
#####################################################################################

#################DEFINING_MODELS##############################################

def save_model(model):
    torch.save(model.state_dict(), 'saved_models/actor_critic.pth')
    print("Model saved to 'saved_models/actor_critic.pth'")

def load_model(model):
    model_path = 'saved_models/actor_critic.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from '{model_path}'")
    else:
        print(f"Model not found, training from scratch")

############################################################################





#######A-C_NETWORK##########################################################
INPUTS = 5 #number of state inputs (x, y, angle, velocity, visionLength)

OPTIM_LR = 0.006 #optimiser learning rate # 0.006
GAMMA = 0.98 #discount factor for future rewards #0.99

#GRAPHING AGENT POLICY
actionProbs = []
totalAP = []
criticValue = []
totalCV = []



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

if loadModel:
    load_model(model)

optimizer = optim.Adam(model.parameters(), lr=OPTIM_LR)
eps = np.finfo(np.float32).eps.item() #epsilon, const to add numerical stability

SavedAction = namedtuple('SavedAction', ['log_prob', 'value']) #store log prob of selected action and state value when action was taken -> used to compute loss for policy and value function (log_prob is used due to stability and policy gradient reasons)




def select_action(state):
    global num_eps, ACTOR_EPSILON, ACTOR_TEMP, DANGER_EPSILON, D_MIN_EPSILON, D_EPSILON_DECAY

    if state[4] < 0.5: #in danger zone
        print("DANGER_EPSILON: ", DANGER_EPSILON)
        if np.random.rand() < DANGER_EPSILON:
            DANGER_EPSILON = max(D_MIN_EPSILON, DANGER_EPSILON*D_EPSILON_DECAY) #decay this over time as well
            action = 1 #guide the agent to go backwards/coast
            return action
             

    if useEpsilonGreedy:
        if np.random.rand() < ACTOR_EPSILON: #GREEDY EPSILON EXPLORATION
            action = np.random.choice([0, 1, 2])
            return action

    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    if useSoftMax:
        probs = F.softmax(probs / ACTOR_TEMP, dim=-1) #use softmax for action selection
    #create categorical distr over list of probabilities

    m = Categorical(probs)
    action = m.sample() #sampling allows us to select actions based on probabilities

    #GRAPHING AGENT POLICY
    actionProbs.append(probs)
    criticValue.append(state_value)
    totalAP.append(probs)
    totalCV.append(state_value)

    #save action to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value)) #store the entropy of the probability distribution -> tells us the "uncertainty" of policy at that decision
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
        policy_losses.append(-log_prob * advantage)  #actor loss (gradient ascent)
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
#####################################################################################





###############STAT_TRACKING###############################################

def statUpdates():
    episode_label.text = f'Episode: {num_eps}'
    temp_label.text = f'Temp: {ACTOR_TEMP}'
    update_graph()

#statistics for simulation tracking
episode_label = pyglet.text.Label( 
    f'Episode: {num_eps}',
    font_name='Arial',
    font_size=14,
    x=WINDOW_WIDTH - 10,  # Position near the top right corner
    y=WINDOW_HEIGHT - 10,
    anchor_x='right',
    anchor_y='top',
    color=(0, 0, 0, 255)  # Black color
)
temp_label = pyglet.text.Label(
    f'Temp: {ACTOR_TEMP}',
    font_name='Arial',
    font_size=14,
    x=WINDOW_WIDTH - 10,  # Position near the top right corner
    y=WINDOW_HEIGHT - 30,
    anchor_x='right',
    anchor_y='top',
    color=(255, 0, 0, 255) # Red color
)

#########################################################################################





############MAIN################################################################

pyglet.clock.schedule_interval(envStep, UPDATE_FREQUENCY/SPEED_UP)
pyglet.clock.schedule(update)#call update function according to system refresh rate

################################################################################





############GRAPHING#################################################################

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

graph_window = pyglet.window.Window(width=800, height=600, caption="Graph Window")

cached_pyglet_image = None

def update_graph():
    global cached_pyglet_image, num_eps
    ax1.cla()
    ax2.cla()
    if actionProbs:
        probs = torch.stack(actionProbs).detach().numpy()
        ax1.stackplot(range(probs.shape[0]), probs.T, labels=['Coast', 'Brake', 'Accelerate'], colors=['#FF6666', '#66FF66', '#6666FF'], edgecolor='black')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Probability')
        ax1.set_title("Action Probability")
        ax1.legend(loc='upper left')
    if criticValue:
        losses = torch.stack(criticValue).detach().numpy()
        ax2.plot(range(losses.shape[0]), losses, color='black')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.set_title("Critic Losses")
        ax2.set_ylim(-2, 2)
    fig.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.tostring_argb()
    width, height = fig.canvas.get_width_height()
    cached_pyglet_image = pyglet.image.ImageData(width, height, 'RGBA', buf, pitch=-4 * width)

    actionProbs.clear() 
    criticValue.clear()
    ax1.set_xlim(left=0)  # Reset the x-axis to start from 0
    ax2.set_xlim(left=0)  # Reset the x-axis to start from 0

    #plotting avg rewards
    avg_ep_safety_rewards.append(ep_safety_rewards[:200])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4)) 

@graph_window.event
def on_draw():
    gl.glClearColor(1, 1, 1, 1.0)
    graph_window.clear()

    global cached_pyglet_image
    if cached_pyglet_image:
        x = (graph_window.width - cached_pyglet_image.width) // 2
        y = (graph_window.height - cached_pyglet_image.height) // 2
        cached_pyglet_image.blit(x, y)

################################################################






###################DRAWTOSCREEN####################################
@window.event
def on_draw():
    gl.glClearColor(1, 1, 1, 1.0)
    window.clear()

    # Draw the Simulation
    batch1.draw()
    batch2.draw()
    episode_label.draw()
    temp_label.draw()

@graph_window.event
def on_draw():
    gl.glClearColor(1, 1, 1, 1.0)
    graph_window.clear()

    global cached_pyglet_image
    if cached_pyglet_image:
        x = (graph_window.width - cached_pyglet_image.width) // 2
        y = (graph_window.height - cached_pyglet_image.height) // 2
        cached_pyglet_image.blit(x, y)
###################################################################

pyglet.app.run()






####################POST-SIMULATION PLOTTING############################################
plt.close() #close old plots

output_dir = 'saved_models'



# Plotting the main data -> OVER EVERY EPOCH
plt.plot(episode_rewards, label='Reward per Episode', color='b') 
plt.plot(running_averages, label='Average Reward (Last 50 Episodes)', color='g')
plt.plot(episodeS_rewards, label='EpisodeS Rewards', color='gray', alpha=0.5) 
plt.plot(runningS_averages, label='RunningS Averages', color='orange', alpha=0.5)  
# Adding labels and title
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards and Average Rewards Over Episodes')
# Adding the hyperparameters as a text box in the plot
hyperparameters = f"""
STARTING_EPSILON: {STARTING_EPSILON}
EPSILON_DECAY: {EPSILON_DECAY}
OPTIM_LR: {OPTIM_LR}
GAMMA: {GAMMA}
"""
# Position the text box (adjust the coordinates as needed)
plt.text(0.96, 0.5, hyperparameters, transform=plt.gca().transAxes,
         fontsize=5, verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.4, edgecolor='black', boxstyle='round,pad=0.3'))

plt.legend()
if saveModel: #if saving model then save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving plot to 'saved_models/rewards_plot.png'")
    plt.savefig(f'{output_dir}/rewards_plot.png') #save plot as png
plt.show()



# Plotting totalAP
totalActionProbs = torch.stack(totalAP).detach().numpy()
fig, ax1 = plt.subplots(figsize=(8, 5))  # Create a figure and axis
ax1.stackplot(
    range(totalActionProbs.shape[0]),  # X-axis range
    totalActionProbs.T,  # Transpose for correct shape
    labels=['Coast', 'Brake', 'Accelerate'],
    colors=['#FF6666', '#66FF66', '#6666FF'],
    edgecolor='black'
)
ax1.set_xlabel('Step')
ax1.set_ylabel('Probability')
ax1.set_title("Action Probability")
ax1.legend(loc='upper left')
if saveModel: #if saving model then save the plot
    print("Saving plot to 'saved_models/actionProbs_plot.png'")
    plt.savefig(f'{output_dir}/actionProbs_plot.png') #save plot as png
plt.show()



# Plotting totalCV
losses = torch.stack(totalCV).detach().numpy()
fig, ax2 = plt.subplots(figsize=(8, 5))  # Create a figure and axis
ax2.plot(range(losses.shape[0]), losses, color='black')
ax2.set_xlabel('Step')
ax2.set_ylabel('Loss')
ax2.set_title("Critic Losses")
ax2.set_ylim(-2, 2)
if saveModel: #if saving model then save the plot
    print("Saving plot to 'saved_models/totalCV_plot.png'")
    plt.savefig(f'{output_dir}/totalCV_plot.png') #save plot as png
plt.show()

#Plotting avg_safety_rewards_variance
avg_ep_safety_rewards = np.array(avg_ep_safety_rewards)
mean_ESR = np.mean(avg_ep_safety_rewards, axis=0)
std_ESR = np.std(avg_ep_safety_rewards, axis=0)
timesteps = np.arange(200)
plt.figure(figsize=(10, 5))
plt.plot(timesteps, mean_ESR, label='Average Safety Reward', color='b')
plt.fill_between(timesteps, mean_ESR - std_ESR, mean_ESR + std_ESR, color='b', alpha=0.2, label="Standard Deviation")
plt.xlabel('Timesteps')
plt.ylabel('Average Safety Reward')
plt.title('Average Safety Reward over an Episode')
plt.legend()
plt.grid()
plt.show()
print(f"AVG_EP: {avg_ep_safety_rewards}")
print(f"MEAN: {mean_ESR}")
print(f"stdERROR: {std_ESR}")