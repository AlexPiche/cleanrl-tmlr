import numpy as np


def compute_Q_pi_well(mdp, Q0=None, eps=0):
  '''
  INPUTS:
  Q0 is the Q-value the policy will be derived on.
  Q0.shape: Ns x Na

  eps is the greedy coeff

  OUTPUTS:
  Q_pi: size Ns x Na
  '''

  Ns = mdp.P.shape[0]
  Na = mdp.P.shape[1]

  if Q0 is None:
    pi = eps_greedy(np.random.randn(Ns+1, Na), eps=eps) # very random pi
  else:
    assert Q0.shape == np.zeros((Ns,Na)).shape
    Q0well = np.zeros((Ns+1, Na))
    Q0well[:Ns] = np.copy(Q0)
    pi = eps_greedy(Q0well, eps=eps)
  
  Qwell = compute_Qwell(mdp, pi)
  return Qwell.reshape((Ns+1, Na))[:-1]


def extend_PR_well(mdp, Psasa, R):
  Ns = mdp.P.shape[0]
  Na = mdp.P.shape[1]
  Rwell = np.zeros((Ns+1, Na))
  R = R.reshape((Ns, Na))
  Rwell[0:Ns, 0:Na] = np.copy(R)

  Psasa = Psasa.reshape((Ns, Na, Ns, Na))
  P = np.zeros((Ns+1, Na, Ns+1, Na))

  P[0:Ns, 0:Na, 0:Ns, 0:Na] = np.copy(Psasa)
  final_state = np.where(mdp.R == 1)[0][0]
  P[final_state, :, :, :] = 0
  P[final_state, :, -1, :] = 1/Na
  P[-1, :, -1, :] = 1/Na
  return P.reshape(((Ns+1)*Na, (Ns+1)*Na)), Rwell

def eps_greedy(Q, eps=0.05):
  Na = Q.shape[1]
  greedy = (Q==np.max(Q, axis=1, keepdims=True))
  greedy = greedy/greedy.sum(1, keepdims=True)
  pi = (1-eps)*greedy + eps*np.ones_like(Q)/Na
  pi = pi/pi.sum(1, keepdims=True)
  return pi

def compute_Qwell(mdp, pi):
    Ns = mdp.P.shape[0]
    Na = mdp.P.shape[1]
    Nsa = Ns*Na
    Psasa = compute_Psasa(mdp, pi[:-1, :])
    Pwell, Rwell = extend_PR_well(mdp, Psasa, mdp.R)
    # mu_sa = (mdp.p0[:, np.newaxis]*pi).reshape(Nsa)
    Qwell = np.linalg.inv((np.eye(Nsa+Na)-mdp.gamma*Pwell))@(Rwell.ravel())
    return Qwell

def compute_Psasa(mdp, pi):
    Ns = mdp.P.shape[0]
    Na = mdp.P.shape[1]
    size = int(np.sqrt(mdp.p0.size))
    Nsa = Ns*Na
    return (mdp.P[:, :, :, np.newaxis]*pi[np.newaxis, np.newaxis, :, :]).reshape(Nsa, Nsa)


from collections import defaultdict

def compute_optimal_Q_values(env, discount):
    alpha = 0.1
    Q = defaultdict(lambda: np.ones(4))
    transitions = []  
    done = True
    for i in range(1500):
        if done:
            state = env.reset()
            done = False      
        eps_step = 0
        while eps_step < 11**2:
            eps_step += 1
            action = np.random.randint(4)
            next_state, reward, done, _ = env.step(action)
            transitions.append((state[0, :-1], action, reward, next_state[0, :-1], done))
            if done:
                break
            state = next_state  
    for _ in range(300):
        all_td_delta = 0
        for transition in transitions:
            state, action, reward, next_state, done = transition
            state = np.where(state.reshape(-1))[0][0]
            next_state = np.where(next_state.reshape(-1))[0][0]
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + (1-done) * discount * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            all_td_delta += td_delta**2
            Q[state][action] += alpha * td_delta      
        print(all_td_delta/len(transitions))
        if all_td_delta/len(transitions) < 0.0000001:
            break  
    return np.stack([Q[i] for i in range(11**2)])          