import numpy as np
import utils
import random


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self._train = True
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None

    def update_n(self, state, action):
        if self._train:  
            self.N[state][action] = self.N[state][action] + 1


    def update_q(self, s, a, r, s_prime):
          if self._train:
            alpha = self.C / (self.C + self.N[s][a])
            self.Q[s][a] = self.Q[s][a] + (alpha * (r + self.gamma * np.max(self.Q[s_prime] ) - self.Q[s][a]))
         
               
    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)
        '''
        i_Q = [0,0,0,0,0,0,0,0]
        
        if state[0] == 40: 
            i_Q[0] = 1 
        elif state[0] == 480:
            i_Q[0] = 2 
        if state[1] == 40:
            i_Q[1] = 1
        elif state[1] == 480:
            i_Q[1] = 2
        if state[0] > state[3]: 
            i_Q[2] = 1
        elif state[0] < state[3]:
            i_Q[2] = 2
        if state[1] > state[4]:
            i_Q[3] = 1
        elif state[1] < state[4]:
            i_Q[3] = 2
        if (state[0],state[1]-40) in state[2]:
            i_Q[4] = 1
        if (state[0],state[1]+40) in state[2]:
            i_Q[5] = 1
        if (state[0]-40,state[1]) in state[2]:
            i_Q[6] = 1
        if (state[0]+40,state[1]) in state[2]:
            i_Q[7] = 1
        
        if self._train: 
            max_Q = -1000000
            max_a = 3
            reward = -0.1

            if points-self.points == 1:
                reward = 1
            if dead:
                reward = -1
            if self.s != None and self.a != None:
                for action in range(3, -1, -1):
                    if self.Q[i_Q[0],i_Q[1],i_Q[2],i_Q[3],i_Q[4],i_Q[5],i_Q[6],i_Q[7],action] > max_Q:
                        max_Q = self.Q[i_Q[0],i_Q[1],i_Q[2],i_Q[3],i_Q[4],i_Q[5],i_Q[6],i_Q[7],action]
                        max_a = action
                self.N[self.s[0],self.s[1],self.s[2],self.s[3],self.s[4],self.s[5],self.s[6],self.s[7],self.a] += 1 
                alpha = self.C/(self.C + self.N[self.s[0],self.s[1],self.s[2],self.s[3],self.s[4],self.s[5],self.s[6],self.s[7],self.a])
                
                self.Q[self.s[0],self.s[1],self.s[2],self.s[3],self.s[4],self.s[5],self.s[6],self.s[7],self.a] += \
                    alpha*(reward-self.Q[self.s[0],self.s[1],self.s[2],self.s[3],self.s[4],self.s[5],self.s[6],self.s[7],self.a] + self.gamma*max_Q)
            if dead:
                self.reset()
            else:
                self.s = i_Q
                self.points = points
                max_q_n = -1000000
                for action in range(3, -1, -1):
                    if self.N[i_Q[0],i_Q[1],i_Q[2],i_Q[3],i_Q[4],i_Q[5],i_Q[6],i_Q[7],action] < self.Ne:
                        self.a = action 
                        break
                    if self.Q[i_Q[0],i_Q[1],i_Q[2],i_Q[3],i_Q[4],i_Q[5],i_Q[6],i_Q[7],action] > max_q_n:
                        self.a = action 
                        max_q_n = self.Q[i_Q[0],i_Q[1],i_Q[2],i_Q[3],i_Q[4],i_Q[5],i_Q[6],i_Q[7],action]
            
        else: 
            max_Q = -100000
            for action in range(3, -1, -1):
                if self.Q[i_Q[0],i_Q[1],i_Q[2],i_Q[3],i_Q[4],i_Q[5],i_Q[6],i_Q[7],action] > max_Q:
                    max_Q = self.Q[i_Q[0],i_Q[1],i_Q[2],i_Q[3],i_Q[4],i_Q[5],i_Q[6],i_Q[7],action]
                    self.a = action

        return self.a