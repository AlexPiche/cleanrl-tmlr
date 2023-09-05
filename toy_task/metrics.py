import jax.numpy as jnp
import chex
from rl import network
from tabular import compute_Q_pi_well


def get_true_error(agent_state, walls, true_q, dims, mdp):
  obs = jnp.eye(dims**2)
  timer = jnp.zeros((dims**2, 1))
  obs = jnp.concatenate([obs, timer], -1)
  q_values = network.apply(agent_state.q_params, obs)
  #max_q = jnp.max(q_values, -1)
  #true_q = compute_Q_pi_well(mdp, q_values)
  #max_q = jnp.reshape(max_q, (dims, dims))
  chex.assert_equal_shape([true_q, q_values])
  diff_q = (q_values - true_q)
  metrics = {'mean_true_error': jnp.mean(diff_q**2),
          'max_true_error': jnp.max(diff_q**2),
          'max_q': jnp.max(jnp.abs(q_values)),
          'mean_q': jnp.mean(q_values),
          #'q_values': q_values,
          #'true_error': diff_q**2
          }
  
  #diff_q = jnp.reshape(diff_q, -1)
  #q_values = jnp.reshape(q_values, -1)
  #for i in range(diff_q.shape[0]-1):
  #  #metrics[f'q_value_{i}'] = q_values[i]
  #  metrics[f'diff_q_{i}'] = diff_q[i]**2
  
  return metrics
