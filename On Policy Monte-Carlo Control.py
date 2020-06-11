from collections import defaultdict 
from racetrack_env import RacetrackEnv
import numpy as np
import matplotlib.pyplot as plt


class MonteCarloControl:
	
	"""
	For the Racetrack Environment, the agent learns the optimal policy using First Visit Monte-Carlo On Policy Control.
	For each episode, the mean return for 20 agents is plotted.
	"""


	def __init__(self, environment,policy="epsilon_greedy",epsilon=0.1,gamma=0.9,episodes=150):

		self.environment=environment
		self.policy=policy
		self.epsilon=epsilon
		self.gamma=gamma
		self.episodes=episodes
		self.rewards_per_episode=np.zeros(self.episodes)
		self.returns=defaultdict(float)
		self.count=defaultdict(float)
		self.q_table= defaultdict(lambda: np.zeros(len(self.environment.get_actions()))) 


	def MonteCarlo(self):


		def epsilon_greedy_policy(state): 

			if np.random.uniform(0, 1) < self.epsilon:
					action=np.random.randint(0,len(self.environment.get_actions()))

			else:
				maximum_q_values = np.nonzero(self.q_table[state] == np.max(self.q_table[state]))[0] 
				action = np.random.choice(maximum_q_values)

			return action


		def state_action_pair():
			state_action_list=[]
			for element in episode_list:
				state_action_list.append((element[0],element[1]))

			return set(state_action_list)


		for episode in range(self.episodes):

			episode_list=[]
			cumulative_reward=0
			terminal=False
			state=self.environment.reset()
			total_G=0

			while terminal!=True:

				action=epsilon_greedy_policy(state)
				next_state,reward,terminal=self.environment.step(action)

				episode_list.append((state,action,reward))
				cumulative_reward+=reward
				state=next_state

				if terminal==True:
					break

			for state,action in state_action_pair():

				pair=(state,action)

				first_occurence=next(index for index,element in enumerate(episode_list) if element[0]==state and element[1]==action)

				total_G=0

				for index,element in enumerate(episode_list[first_occurence:]) :
					total_G=total_G+(element[2]*(self.gamma**index))

				self.returns[pair]+=total_G  
				self.count[pair] += 1.0
				self.q_table[state][action]=self.returns[pair]/self.count[pair]

			self.rewards_per_episode[episode]=cumulative_reward
		return self.rewards_per_episode

    
reward_per_episode=[]
for i in range(20):
	learning=MonteCarloControl(RacetrackEnv()) 
	reward_per_episode.append(learning.MonteCarlo())

average_return_per_episode=np.matrix(reward_per_episode).mean(0) 
average_return_per_episode_Monte_Carlo=average_return_per_episode.tolist() 

plt.plot(average_return_per_episode_Monte_Carlo[0])
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.title("Monte Carlo Average Return-20 Agents")
plt.show()