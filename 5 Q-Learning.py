import numpy as np
import matplotlib.pyplot as plt
from racetrack_env import RacetrackEnv
from collections import defaultdict 



class OffPolicyQLearning:
	"""
	For the Racetrack Environment, the agent learns the optimal policy using Q-Lerning Off Policy Control.
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



	def Q_learning_algorithm(self):	


		
		for episode in range(self.episodes):

			terminal=False
			cumulative_reward=0
			state=self.environment.reset() 

			while terminal!=True:

				if np.random.uniform(0, 1) < self.epsilon:
					action=np.random.randint(0,len(self.environment.get_actions()))

				else:
					maximum_q_values = np.nonzero(self.q_table[state] == np.max(self.q_table[state]))[0]
					action = np.random.choice(maximum_q_values)

				next_state, reward, terminal= self.environment.step(action)
				
				self.q_table[state][action]= self.q_table[state][action]+self.alpha*(reward+self.gamma*max(self.q_table[next_state])-self.q_table[state][action])
				cumulative_reward+=reward
				state=next_state
				if terminal==True:
					for i in range(9):
						self.q_table[next_state][i]=0
					break

			self.reward_per_episode[episode]=cumulative_reward
		return self.reward_per_episode



reward_per_episode=[]
for i in range(20):
    learning=OffPolicyQLearning(RacetrackEnv())
    reward_per_episode.append(learning.Q_learning_algorithm())

average_return_per_episode=np.matrix(reward_per_episode).mean(0)
average_return_per_episode_list_Q_Learning=average_return_per_episode.tolist() 

plt.plot(average_return_per_episode_list_Q_Learning[0])
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.title("Q Learning Average Return-20 Agents")
plt.show()