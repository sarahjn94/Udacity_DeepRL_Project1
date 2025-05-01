# Udacity_DeepRL_Project1

# Project 1: Navigation

### Learning Algorithm

For this Navigation project, the implementation used a Deep Q Reinforcemnt Learning algorithm.

	- Used a replay buffer to store past experiences
	- Q-value updated with Bellman equation
	- Agent uses an epsilon-greedy policy
	- Neural network trained with stochastic gradient descent
	- Neural network is fully-connected feedforward network
		- Input layer of size 37
		- Two hidden layers of size 64 each
		- Output layer of size 4 (for each of the 4 actions)
	- Hyperparameters:
		- Replay buffer size: 1e5
		- Batch size: 64
		- Learning rate: 5e-4
		- Gamma (discount factor): 0.99
		- Epsilon start: 1.0
		- Epsilon decay: 0.995
		- Epsilon end: 0.01
		- Tau: 1e-3
		- Update frequency: 4
		- Training episodes: 2000

### Plot of Rewards

![Graph](/outputGraph.png)

Environment solved in 516 episodes!	Average Score: 13.02


### Ideas for Future Work

	- Try out different RL algorithms such as: Double DQN, Dueling DQN, or Prioritized Experience Replay
	- Further turning of hyperparameters
	- Use visual raw pixel inputs
	- Use a continuous action space
