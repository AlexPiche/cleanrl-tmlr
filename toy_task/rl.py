import jax
import jax.numpy as jnp
import haiku as hk
import collections
import optax
import chex

AgentState = collections.namedtuple('agent_state', ['q_params',
                                                    'target_q_params',
                                                    'q_opt_state',])

def q_forward(inputs):
    flat_inputs = hk.Flatten()(inputs)
    action_values = hk.nets.MLP([128, 128, 4])(flat_inputs)
    return action_values

network = hk.without_apply_rng(hk.transform(q_forward))
optimizer = optax.chain(optax.scale_by_adam(), 
                        optax.scale(-1e-4))

def init_agent_state(rng, input):
  q_params = network.init(rng, input)
  q_opt_state = optimizer.init(q_params)
  agent_state = AgentState(q_params=q_params,
                           target_q_params=q_params,
                           q_opt_state=q_opt_state,)
  return agent_state

@jax.jit
def act(agent_state, obs):
  action_value = network.apply(agent_state.q_params, obs)
  action_value = jnp.squeeze(action_value)
  return jnp.argmax(action_value)

def DQN(params, target_params, state, action, reward, next_state, done, reg_weight, mellow_temp, use_mellow_op, discount):
  del reg_weight, mellow_temp, use_mellow_op
  action = jnp.reshape(action, (-1, 1))
  reward = jnp.reshape(reward, (-1, 1))
  done = jnp.reshape(done, (-1, 1))
  q_value = jnp.take_along_axis(network.apply(params, state), action, -1)
  next_q_value = jnp.max(network.apply(target_params, next_state), -1, keepdims=True)

  chex.assert_equal_shape([done, reward, next_q_value, q_value])
  td_error = reward + discount*(1 - done)*jax.lax.stop_gradient(next_q_value) - q_value
  loss = 0.5 * jnp.mean(td_error**2)
  return loss, {'td_loss': loss}

def FRDQN(params, target_params, state, action, reward, next_state, done, reg_weight, mellow_temp, use_mellow_op, discount):
  action = jnp.reshape(action, (-1, 1))
  reward = jnp.reshape(reward, (-1, 1))
  done = jnp.reshape(done, (-1, 1))
  q_value = jnp.take_along_axis(network.apply(params, state), action, -1)
  
  next_q = network.apply(params, next_state)
  next_q_value = jax.lax.select(use_mellow_op, 
                                mellow_op(next_q, mellow_temp, keepdims=True),
                                jnp.max(next_q, -1, keepdims=True)
                                )

  prior_q_value = jnp.take_along_axis(network.apply(target_params, state), action, -1)
  chex.assert_equal_shape([done, reward, next_q_value, q_value, prior_q_value])
  td_error = reward + discount*(1 - done)*jax.lax.stop_gradient(next_q_value) - q_value
  prior_error = q_value - prior_q_value
  loss = 0.5 * jnp.mean(td_error**2 + reg_weight*prior_error**2)
  
  return loss, {'loss': loss,
                'prior_loss': jnp.mean(prior_error**2),
                'td_loss': jnp.mean(td_error**2)}

def HYBRID_DQN(params, target_params, state, action, reward, next_state, done, reg_weight, mellow_temp, use_mellow_op, discount):
  prior_target_next_q = network.apply(target_params, next_state)
  prior_target_next_action = jnp.argmax(prior_target_next_q, -1)
  prior_target_next_action = jnp.reshape(prior_target_next_action, (-1, 1))

  action = jnp.reshape(action, (-1, 1))
  reward = jnp.reshape(reward, (-1, 1))
  done = jnp.reshape(done, (-1, 1))
  q_value = jnp.take_along_axis(network.apply(params, state), action, -1)
  
  next_q = network.apply(params, next_state)
  next_q_value = jnp.take_along_axis(next_q, prior_target_next_action, -1)
  chex.assert_equal_shape([done, reward, next_q_value, q_value])
  td_error = reward + discount*(1 - done)*jax.lax.stop_gradient(next_q_value) - q_value

  prior_target_next_q_value = jnp.take_along_axis(prior_target_next_q, prior_target_next_action, -1)
  prior_next_q_value = jnp.take_along_axis(network.apply(params, next_state), prior_target_next_action, -1)
  prior_error = jax.lax.stop_gradient(prior_next_q_value - prior_target_next_q_value)
  chex.assert_equal_shape([prior_error, q_value])
  loss = jnp.mean(0.5*td_error**2 + reg_weight*prior_error*q_value)
  
  return loss, {'loss': loss,
                'prior_loss': jnp.mean(prior_error**2),
                'td_loss': jnp.mean(td_error**2)}


def update(agent_state, batch, reg_weight, mellow_temp, use_mellow_op, loss_fn):
    state, action, reward, next_state, done = batch
    g, stats = jax.grad(loss_fn, 0, has_aux=True)(agent_state.q_params,
                                                  agent_state.target_q_params, 
                                                  state, 
                                                  action, 
                                                  reward, 
                                                  next_state, 
                                                  done,
                                                  reg_weight, 
                                                  mellow_temp,
                                                  use_mellow_op)
    
    q_update, new_q_opt_state = optimizer.update(g,
                                                 agent_state.q_opt_state)
    
    new_q_params = jax.tree_multimap(lambda p, u: p + u,
                                     agent_state.q_params,
                                     q_update)
    
    new_agent_state = AgentState(q_params=new_q_params,
                                 target_q_params=agent_state.target_q_params,
                                 q_opt_state=new_q_opt_state)
    
    return new_agent_state, stats


@jax.jit
def update_target(agent_state, tau):
  new_q_params = jax.tree_multimap(lambda p1, p2: tau * p1 + (1-tau) * p2, 
                                   agent_state.q_params,
                                   agent_state.target_q_params)
  agent_state = agent_state._replace(target_q_params=new_q_params)
  return agent_state
          

def mellow_op(x, temp, keepdims=False):
  mm_x = (jax.nn.logsumexp(x * temp, -1, keepdims=keepdims) - jnp.log(x.shape[-1]))/temp
  return jax.lax.stop_gradient(mm_x)