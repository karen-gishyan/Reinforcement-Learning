import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

class Environment:

	#def __init__(self, gold_state,bomb_state,
	#	gold_reward=9,bomb_reward=-11,actions=["n","s","w","e"],num_rows=5,num_columns=5,gamma=1):		
	
	def __init__(self, evn_number, bomb_state, second_environment=None,third_environment=None,gold1_state=None,gold2_state=None, 
		gold_reward=9, bomb_reward=-11, actions=["n","s","w","e"],num_rows=5,num_columns=5,gamma=1):		
	

		self.actions=actions
		self.evn_number=evn_number
		self.second_environment=second_environment
		self.third_environment=third_environment
		self.length_actions=len(self.actions)
		self.gamma=gamma
		self.gold1_state=gold1_state
		self.gold2_state=gold2_state
		self.bomb_state=bomb_state
		self.gold_reward=gold_reward
		self.bomb_reward=bomb_reward
		self.num_rows=num_rows
		self.num_columns=num_columns
		self.number_of_states=self.num_rows*self.num_columns
		self.action_rewards=np.full(self.number_of_states,-1) 		
		self.action_rewards[bomb_state]=bomb_reward	   		
		self.policy=np.zeros(self.number_of_states)
		self.policy=np.array(['s']*25) 
		self.state_values=np.zeros(self.number_of_states)
		self.states=np.array([i for i in range(25)])

		if  self.evn_number==1:  
			#self.action_rewards[17]=4.01		
			self.action_rewards[gold1_state]=gold_reward 
			self.action_rewards[gold2_state]=gold_reward
			#self.action_rewards[17]=0.001
		if  self.evn_number==2:
			self.action_rewards[gold2_state]=gold_reward
		if  self.evn_number==3:
			self.action_rewards[gold1_state]=gold_reward

	def one_step_lookahead(self,action_index,agent_position):	
		self.agent_position=int(agent_position)		
		position_checker=[]	

		def check_if_directions_exist():
			nonlocal position_checker 
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
	
	def value_iteration(self,environment1,environment2,environment3,terminal_states1=[12,18,23],terminal_states2=[18,23],terminal_states3=[12,18]):		
		action_list1=np.zeros(environment1.length_actions,dtype=float) 
		action_list2=np.zeros(environment2.length_actions,dtype=float) 
		action_list3=np.zeros(environment3.length_actions,dtype=float) 
		
		while True:
			change1=0
			change2=0
			change3=0

			for state in list(environment2.states) :
				if state in terminal_states2:   
					continue 				 		
				state_value=environment2.state_values[state] 
				
				for i in range(environment2.length_actions):
					action_list2[i]=environment2.one_step_lookahead(i,state)
					
				best_value=np.max(action_list2)
				best_policy=np.argmax(action_list2)

				environment2.state_values[state]=best_value
				environment2.policy[state]=environment2.actions[best_policy]
				change2=max(change2,abs(state_value-best_value))


			for state in list(environment3.states) :
				if state in terminal_states3:   
					continue 				 

				state_value=environment3.state_values[state] 
				
				for i in range(environment3.length_actions):
					action_list3[i]=environment3.one_step_lookahead(i,state)
					
				best_value=np.max(action_list3)
				best_policy=np.argmax(action_list3)

				environment3.state_values[state]=best_value
				environment3.policy[state]=environment3.actions[best_policy]
				change3=max(change3,abs(state_value-best_value))
				
			environment1.state_values[12]=environment2.state_values[12]
			environment1.state_values[23]=environment3.state_values[23]
			
			for state in list(environment1.states) :
				if state in terminal_states1:   
					continue 				 
				state_value=environment1.state_values[state] 
				
				for i in range(environment1.length_actions):
					action_list1[i]=environment1.one_step_lookahead(i,state)
					
				best_value=np.max(action_list1)
				best_policy=np.argmax(action_list1)

				environment1.state_values[state]=best_value
				environment1.policy[state]=environment1.actions[best_policy]
				change1=max(change1,abs(state_value-best_value))
				
			for state in [environment1.states[12],environment1.states[23]]:

				state_value=environment1.state_values[state] 
				
				for i in range(environment1.length_actions):
					action_list1[i]=environment1.one_step_lookahead(i,state)
					
				best_value=np.max(action_list1)
				best_policy=np.argmax(action_list1)

				environment1.state_values[state]=best_value
				environment1.policy[state]=environment1.actions[best_policy]
				change1=max(change1,abs(state_value-best_value))



			if change3<self.epsilon:  
				break
		return environment1.policy, environment1.state_values


environment2=Environment(evn_number=2, gold2_state=23, bomb_state=18)
environment3=Environment(evn_number=3, gold1_state=12, bomb_state=18)	
environment1=Environment(evn_number=1, gold1_state=12, gold2_state=23, bomb_state=18, second_environment=environment2,third_environment=environment3)	
iteration=Value_Iteration()

policy,v=iteration.value_iteration(environment1,environment2,environment3)
print("Optimal Policy")
print(policy)
print("Optimal Values:")
print(v) 