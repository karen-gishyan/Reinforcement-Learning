import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

class Environment:

	def __init__(self, gold_state,bomb_state,
		gold_reward=9,bomb_reward=-11,actions=["n","s","w","e"],num_rows=5,num_columns=5,gamma=1):

		self.actions=actions
		self.length_actions=len(self.actions)
		self.gamma=gamma
		self.gold_state=gold_state
		self.bomb_state=bomb_state
		self.gold_reward=gold_reward
		self.bomb_reward=bomb_reward
		self.num_rows=num_rows
		self.num_columns=num_columns
		self.number_of_states=self.num_rows*self.num_columns

		self.action_rewards=np.full(self.number_of_states,-1) 
		self.action_rewards[gold_state]=gold_reward    
		self.action_rewards[bomb_state]=bomb_reward	   

		self.policy=np.zeros(self.number_of_states)
		self.policy=np.array(['s']*25) #terminal states labeled as south.
		self.state_values=np.zeros(self.number_of_states)
		self.states=np.array([i for i in range(25)])

	def one_step_lookahead(self,action_index,agent_position):
		self.agent_position=int(agent_position)
		position_checker=[]

		def check_if_directions_exist():
			nonlocal position_checker 
			"""
			For a given state, checks if neighbouring positions are accesible.
			If yes, returns these positions, if not, returns the current position.
			All are stored as integers. 
			Returns a list which contains the positions of the noughboring cells.	
			"""
			# Checks if the agent can move n.
			n_position=self.agent_position+self.num_columns
			if n_position<self.number_of_states:
				position_checker.append(n_position) 
			else:
				position_checker.append(self.agent_position)

			# Checks if the agent can move s.
			s_position=self.agent_position-self.num_columns	
			if s_position>=0:
				position_checker.append(s_position)
			else:
				position_checker.append(self.agent_position)

			# Checks if the agent can move w.
			w_position=self.agent_position-1
			if w_position % self.num_columns < (self.num_columns-1):
				position_checker.append(w_position)
			else:
				position_checker.append(self.agent_position)

			# Checks if the agent can move e.
			e_position=self.agent_position+1 
			if e_position % self.num_columns>0:
				position_checker.append(e_position)
			else:
				position_checker.append(self.agent_position)

			return position_checker


		def find_state_value(position_list):
			"""
			For each action, calculates the weighted average of mooving to another state.
			The neighbouring states are passed as a position list.
			For a given state, iterates over the respective probabilities and the list, returning the value by
			multipltying each state with its reward, probability, and value.
			"""
			primary_action=self.actions[action_index]
			if primary_action=="n":
				checker=position_list 
				cumulative_value=0
				list_of_probabilities_n=[0.85,0.05,0.05,0.05]
				for i,j in zip(range(len(list_of_probabilities_n)),range(len(checker))):
					cumulative_value=cumulative_value + list_of_probabilities_n[i]*((self.action_rewards[checker[j]]+self.gamma*self.state_values[checker[j]]))
				return cumulative_value

			elif primary_action=="s":
				cumulative_value=0
				checker=position_list 
				list_of_probabilities_s=[0.05,0.85,0.05,0.05]
				for i,j in zip(range(len(list_of_probabilities_s)),range(len(checker))):
					cumulative_value=cumulative_value + list_of_probabilities_s[i]*((self.action_rewards[checker[j]]+self.gamma*self.state_values[checker[j]]))
				return cumulative_value

			elif primary_action=="w":
				cumulative_value=0
				checker=position_list 
				list_of_probabilities_w=[0.05,0.05,0.85,0.05]
				for i,j in zip(range(len(list_of_probabilities_w)),range(len(checker))):
					cumulative_value=cumulative_value + list_of_probabilities_w[i]*((self.action_rewards[checker[j]]+self.gamma*self.state_values[checker[j]]))
				return cumulative_value

			elif primary_action=="e":
				cumulative_value=0
				checker=position_list 
				list_of_probabilities_e=[0.05,0.05,0.05,0.85]
				for i,j in zip(range(len(list_of_probabilities_e)),range(len(checker))):
					cumulative_value=cumulative_value + list_of_probabilities_e[i]*((self.action_rewards[checker[j]]+self.gamma*self.state_values[checker[j]]))
				return cumulative_value

			else:
				raise ValueError("Unrecognized action index")

		return find_state_value(check_if_directions_exist())

class Value_Iteration:

	def __init__(self,epsilon=1e-10):
		self.epsilon=epsilon
	def value_iteration(self,environment,terminal_states=[18,23]):
		action_list=np.zeros(environment.length_actions,dtype=float) 

		while True:
			change=0 
			for state in list(environment.states):
				if state in terminal_states:   
					continue

				state_value=environment.state_values[state] 
				
				for i in range(environment.length_actions):
					action_list[i]=environment.one_step_lookahead(i,state)

				best_value=np.max(action_list)
				best_policy=np.argmax(action_list)

				environment.state_values[state]=best_value 
				environment.policy[state]=environment.actions[best_policy]

				change=max(change,abs(state_value-best_value))

			if change<self.epsilon:  
				break
		return environment.policy, environment.state_values

environment=Environment(gold_state=23,bomb_state=18)

iteration=Value_Iteration()
policy,v=iteration.value_iteration(environment) 
print("Optimal Policy:")
print(policy)
print("Optimal Values:")
print(v)