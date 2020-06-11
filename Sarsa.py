import random
random.seed(10)
import numpy as np
import matplotlib.pyplot as plt
from racetrack_env import RacetrackEnv
from collections import defaultdict 

class SarsaLearning:
	"""
	For the Racetrack Environment, the agent learns the optimal policy using SARSA On Policy Control.
	For each episode, the mean return for 20 agents is plotted.
	
	"""

	def __init__(self,environment,policy='epsilon_greedy',epsilon=0.1,alpha=0.2,gamma=0.9,episodes=150):

		self.environment=environment
		self.policy=policy
		self.epsilon=epsilon
		self.alpha=alpha
		self.gamma=gamma
		self.episodes=episodes 
		self.reward_per_episode=np.zeros(self.episodes)
		self.q_table=defaultdict(lambda: np.zeros(len(environment.get_actions()))) 

		self.reward_list=[]

	def Sarsa_learning_algorithm(self):

		for episode in range(self.episodes):

			terminal=False
			cumulative_reward=0
			state=self.environment.reset()

			if self.policy=='epsilon_greedy' and np.random.uniform(0, 1) < self.epsilon:

				action=np.random.randint(0,len(self.environment.get_actions()))

			else:

				maximum_q_values = np.nonzero(self.q_table[state] == np.max(self.q_table[state]))[0]
				action = np.random.choice(maximum_q_values)

			while terminal!=True:

				next_state,reward,terminal= self.environment.step(action) 


				if self.policy=='epsilon_greedy' and np.random.uniform(0, 1) < self.epsilon:

					action_prime=np.random.randint(0,len(self.environment.get_actions()))

				else:

					maximum_q_values = np.nonzero(self.q_table[next_state] == np.max(self.q_table[next_state]))[0]
					action_prime = np.random.choice(maximum_q_values)


				self.q_table[state][action]= self.q_table[state][action]+self.alpha*(reward+self.gamma*self.q_table[next_state][action_prime]-self.q_table[state][action])
				state=next_state
				action=action_prime
				cumulative_reward+=reward
				if terminal==True:
					for i in range(9):
						self.q_table[next_state][i]=0
					break


			self.reward_per_episode[episode]=cumulative_reward

		return self.reward_per_episode



reward_per_episode=[]
for i in range(20):
    learning=SarsaLearning(RacetrackEnv()) 
    reward_per_episode.append(learning.Sarsa_learning_algorithm())

average_return_per_episode=np.matrix(reward_per_episode).mean(0) 
average_return_per_episode_SARSA=average_return_per_episode.tolist() 

plt.plot(average_return_per_episode_SARSA[0])
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.title("SARSA Average Return-20 Agents")
plt.show()