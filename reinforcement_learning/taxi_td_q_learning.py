import random
import gym

env = gym.make('Taxi-v3')

alpha = 0.4
gamma = 0.999
epsilon = 0.017

q = {}
for s in range(env.observation_space.n):
  for a in range(env.action_space.n):
    q[(s, a)] = 0

def update_q_table(prev_state, action, reward, next_state, aplha, gamma):
  qa = max([q[(next_state, a)] for a in range(env.action_space.n)])
  q[(prev_state, action)] += alpha * (reward + gamma * qa - q[(prev_state, action)])

def epsilon_greedy_policy(state, epsilon):
  if random.uniform(0, 1) < epsilon:
    return env.action_space.sample()
  else:
    return max(list(range(env.action_space.n)), key = lambda x : q[(state, x)])

for i in range(8000):
  r = 0
  prev_state = env.reset()
  
  while True:
    action = epsilon_greedy_policy(prev_state, epsilon)

    next_state, reward, done, _ = env.step(action)

    update_q_table(prev_state, action, reward, next_state, alpha, gamma)

    prev_state = next_state

    r += reward

    if done:
      break

env.close()