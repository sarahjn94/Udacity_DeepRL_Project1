import numpy as np
from unityagents import UnityEnvironment
from collections import deque
from dqn_agent import Agent
import torch
import matplotlib.pyplot as plt
%matplotlib inline

env = UnityEnvironment(file_name="Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print(brain_name)

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)        
        
env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

numEpisodes = 2000
scores = []
scores_window = deque(maxlen=100)  # last 100 scores
eps = 1.0
epsDecay = 0.995
epsEnd = 0.01

for i in range (1, numEpisodes+1):
    env_info = env.reset(train_mode=True)[brain_name] #reset env at start of each episode
    state = env_info.vector_observations[0] 
    score = 0
    while True:
        action = agent.act(state, eps)        # select an action
        env_info = env.step(np.array([action]))[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        agent.step(state, action, reward, next_state, done) # agent learning from this experience
        score += reward                                # roll over the state to next time step
        state = next_state                             # update the score
        if done:                                       # exit loop if episode finished
            break
    scores.append(score)
    scores_window.append(score)
    eps = (max(epsEnd, epsDecay*eps))
    
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)), end="")
    if i % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
    if np.mean(scores_window)>=13.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), 'model.pth')
        break

env.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()