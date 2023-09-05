import wandb
from buffer import ReplayBuffer
import jax
import numpy as np
import haiku as hk
import functools
from absl import app
from absl import flags
from jax.config import config

from mdp import Wrapper, build_MDP
from rl import FRDQN, DQN, act, init_agent_state, update, update_target, HYBRID_DQN
from metrics import get_true_error
from tabular import compute_optimal_Q_values

FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 42, '')
flags.DEFINE_integer('use_target_net', 0, '')
flags.DEFINE_integer('use_hybrid', 0, '')
flags.DEFINE_integer('use_mellow_op', 0, '')
flags.DEFINE_integer('batch_size', 512, '')
flags.DEFINE_integer('size', 19, '')
flags.DEFINE_integer('target_update_freq', 10, '')
flags.DEFINE_float('epsilon', 0.05, '')
flags.DEFINE_float('reg_weight', 0., '')
flags.DEFINE_float('tau', 1., '')
flags.DEFINE_float('discount', 0.99, '')
flags.DEFINE_float('mellow_temp', 1., '')

def main(argv):
    wandb.login(key='246c8f672f0416b12172d64574c12d8a7ddae387')
    from rl import update
    
    loss_fn = DQN if FLAGS.use_target_net else FRDQN
    loss_fn = HYBRID_DQN if FLAGS.use_hybrid else loss_fn
    loss_fn = functools.partial(loss_fn, discount=FLAGS.discount)
    update = functools.partial(update, loss_fn=loss_fn)
    update = jax.jit(update)
    my_mdp, _ = build_MDP(size=FLAGS.size)
    
    env = Wrapper(build_MDP, size=FLAGS.size)
    test_env = Wrapper(build_MDP, size=FLAGS.size)
    walls = env.get_walls()
    true_q = walls
    
    def get_avg_reward(rng_seq, agent_state, env):
      returns = []
      for _ in range(100):
        done = False
        state = env.reset()
        return_eps = 0
        eps_step = 0
        while eps_step < 11**2:
          eps_step += 1
          if np.random.random() <= 0.1:
            action = np.random.randint(4)
          else:
            action = act(agent_state, state)
          next_state, reward, done, info = env.step(int(action))
          return_eps += reward
          if done:
            break
          state = next_state
    
        returns.append(return_eps)
      
      return np.array(returns).mean()
    
    #true_q, walls = compute_value(env, FLAGS.size)
    true_q = compute_optimal_Q_values(env, FLAGS.discount)
    for seed in range(40):
      wandb.init(project="fr_tmlr29_timer",
                config=FLAGS,
                settings=wandb.Settings(start_method='fork')) 
      buffer = ReplayBuffer(100000)
      agent_state = None
      rng_seq = hk.PRNGSequence(FLAGS.seed + seed)
      iteration = 0
      while iteration < 1e4:
        state = env.reset()
        if agent_state is None:
          agent_state = init_agent_state(next(rng_seq), state)
        done = False
        eps_return = 0
        eps_step = 0
        while eps_step < 11**2:
          eps_step += 1
          if np.random.random() <= FLAGS.epsilon:
            action = np.random.randint(4)
          else:
            action = act(agent_state, state)
          action = int(action)
          next_state, reward, done, _ = env.step(action)
          buffer.push(state, action, reward, next_state, done)
          if done:
            break
          state = next_state 
          if buffer.can_sample():
            batch = buffer.sample(FLAGS.batch_size)
            agent_state, stats = update(agent_state, batch, FLAGS.reg_weight,
                                        FLAGS.mellow_temp, bool(FLAGS.use_mellow_op))

            if iteration % FLAGS.target_update_freq == 0:
              agent_state = update_target(agent_state, FLAGS.tau)
    
            iteration += 1

      metrics = get_true_error(agent_state, walls, true_q, FLAGS.size, my_mdp)
      est_return = get_avg_reward(rng_seq, agent_state, test_env)
      eps_return = {'return': est_return, 'iteration': iteration}
      wandb.log({**eps_return, **stats, **metrics})
      wandb.finish()
    
    
if __name__ == '__main__':
  config.update('jax_platform_name', 'gpu')  # Default to GPU.
  config.update('jax_numpy_rank_promotion', 'raise')
  config.config_with_absl()
  app.run(main)