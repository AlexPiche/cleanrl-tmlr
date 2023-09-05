import numpy as np
import emdp.emdp.gridworld as gw
from emdp.emdp import actions
import copy



def build_MDP(size=19):
    """
    Example 3.5 from (Sutton and Barto, 2018) pg 60 (March 2018 version).
    A rectangular Gridworld representation of size 5 x 5.

    Quotation from book:
    At each state, four actions are possible: north, south, east, and west, which deterministically
    cause the agent to move one cell in the respective direction on the grid. Actions that
    would take the agent off the grid leave its location unchanged, but also result in a reward
    of −1. Other actions result in a reward of 0, except those that move the agent out of the
    special states A and B. From state A, all four actions yield a reward of +10 and take the
    agent to A'. From state B, all actions yield a reward of +5 and take the agent to B'
    """
    P = gw.build_simple_grid(size=size, p_success=1)
    # modify P to match dynamics from book.

    ##------ States with reward loop back to beginning?
    # P[size**2-1, :, :] = 0
    # P[size**2-1, :, 0] = 1
    
    # P[2*size-1, :, :] = 0
    # P[2*size-1, :, 0] = 1

    # P[size**2-size+1, :, :] = 0
    # P[size**2-size+1, :, 0] = 1
    ##------
    
    # Here R will represent a log density
    ##------ Attribution of the rewards
    R = np.zeros((P.shape[0], P.shape[1])) # initialize a matrix of size |S|x|A|

    walls = np.ones(P.shape[0])#/P.shape[0] # Start in the upper left corner

    #p0 = p0*0
    #p0[0]=1
    gamma = 0.99
    
    ##------ Make 4 rooms
    for j in range(size**2):
      for i in range(size**2):
        # States in the horizontal wall except doors
        is_line = lambda x: x> (size//2)*size-1 and x < (size//2)*size+size and x != (size//2)*size+size//4 and x != (size//2)*size+3*size//4
        k = size*(i%size) + i//size
        if is_line(i):
          walls[i] = 0
          walls[k] = 0
          for a in range(4):
            if P[j, a, i] == 1:
              P[j, a, i] = 0
              P[j, a, j] = 1
            elif P[j, a, k] == 1:
              P[j, a, k] = 0
              P[j, a, j] = 1
      ##------ 
    p0 = np.zeros_like(walls)
    p0[-1] = 1.
    #p0 = copy.deepcopy(walls)
    #p0 /= p0.sum()
    idx = 0
    R[idx,:] = 1

    terminal_states = []
    return gw.GridWorldMDP(P, R, gamma, p0, terminal_states, size), walls


class Wrapper():
  def __init__(self, build_mdp, size=19):
    self.build_mdp = build_mdp
    self.size = size

  def get_walls(self):
    _, walls = self.build_mdp(self.size)
    return walls
  
  def reset(self):
    self.mdp, _ = self.build_mdp(self.size)
    self.eps_step = 0
    state = self.mdp.reset()
    state = state.reshape((1, self.size, self.size))
    timer = np.array([self.eps_step/self.size**2]).reshape((1, 1))
    state = np.concatenate([state.reshape(1, -1), timer], -1)
    return state 

  def step(self, action):
    self.eps_step += 1
    state, reward, done, _ = self.mdp.step(action)
    done = True if reward else False #or (self.eps_step > self.size**2) else False
    #reward = 0 if reward else -1 # always return -1 except when done
    reward -= 0.01
    state = state.reshape((1, self.size, self.size))
    timer = np.array([self.eps_step/self.size**2]).reshape((1, 1))
    timer *= 0
    state = np.concatenate([state.reshape(1, -1), timer], -1)
    return state, reward, done, {}
