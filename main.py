import gymnasium as gym
from agent import Agent
import numpy as np

env = gym.make("Acrobot-v1")
num_states = 6
num_actions = 1
my_agent = Agent(num_states=num_states, num_actions=num_actions)
episodes = 700
scores = []
#Fill the memory with samples (warm-up)
current_state, info = env.reset()
warm_up_session = my_agent.memory_size
# warm_up_session = 10
for _ in range(warm_up_session):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    my_agent.memory.memorize(state=current_state, action=action, reward=reward,
                             next_state=observation, term=terminated, trunc=truncated)
    if terminated or truncated:
        current_state, info = env.reset()
    current_state = observation

for episode in range(episodes):
    observation, info = env.reset()
    done = False
    score = 0
    steps = 0
    while not done:
        action, _ = my_agent.actor_net.sample_actions(observation)
        # print(action)
        observation_, reward, terminated, truncated, info = env.step(action.cpu().data.numpy()[0])
        my_agent.memory.memorize(state=observation, action=action.cpu().data.numpy(), next_state=observation_,
                                 reward=reward, trunc=truncated, term=terminated)
        my_agent.train_step()
        score += reward
        observation = observation_
        steps += 1
        if terminated or truncated:
            done = True
    scores.append(score / steps)
    print("Episode: {} --- Mean score: {}".format(episode, np.mean(scores[-100:])))




