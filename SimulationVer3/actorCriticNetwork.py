#DEFINE HPYERPARAMETERS

#DEF REWARD CALCULATION

#DEF how and when to RESET ENVIRONMENT

#DEF how to process direction update for agent

#DEF NEURAL NETWORK Actor and Critic

#DEF pick_action SELECTION

#DEF optimisers for actor_critic

#DEF VISUALISSER INFO FOR ANALYSIS

#Def update_agent for when updating after the end of an episode

import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

ACTOR_EPSILON = 0.5


class ACNetwork:

    class ActorNet(nn.Module): #ACTOR network
        def __init__(self, hidden_dim=16):
            super().__init__()
            self.hidden = nn.Linear(self.actor_inputs, hidden_dim) #input layer, 16 units in hidden layer
            self.output = nn.Linear(hidden_dim, self.actor_outputs) #output layer has 3 units (forwards, backwards or no action)
            self.epsilon = ACTOR_EPSILON 
        def forward(self, s): #NN forward pass
            outs = self.hidden(s)
            outs - F.relu(outs) #applies ReLU activation
            logits = self.output(outs) #logit computation -> passed through e.g. softmax function to convert to probabilities
            return logits
    
    class ValueNet(nn.Module): #VALUE network
        def __init__(self, hidden_dim=16):
            super().__init__()
            self.hidden = nn.Linear(self.actor_inputs, hidden_dim)
            self.output = nn.Linear(hidden_dim, 1) #1 unit predicting value of state
        def forward(self, s):
            outs = self.hidden(s)
            outs = F.relu(outs)
            value = self.output(outs)
            return value
    
    def __init__(self, num_inputs, num_outputs, agentState):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_inputs = num_inputs
        self.actor_outputs = num_outputs
        self.agentState = agentState

        #instantiating actor and value network and move to GPU if available
        self.actor_func = self.ActorNet().to(self.device)
        self.value_func = self.ValueNet().to(self.device)

    def pick_action(state, actor):
        if np.random.rand() < ACTOR_EPSILON: #random action
            action = np.random.choice(range(actor.output.out_features))
            return action

        with torch.no_grad():
            state_batch = np.expand_dims(state, axis=0)
            state_batch = torch.tensor(state_batch, dtype=torch.float).to(self.device)
            
        


     



#DEFINE ACTOR CRITIC network -> action selection -> optimisers for actor_critic -> define update_agent -> 