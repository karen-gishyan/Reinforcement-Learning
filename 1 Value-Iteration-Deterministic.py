import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

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
        self.policy=np.array([' ']*25)
        self.state_values=np.zeros(self.number_of_states,dtype='double')
        self.states=np.array([i for i in range(25)])


    def one_step_lookahead(self,action_index,agent_position):

        action=self.actions[action_index]
        self.agent_position=int(agent_position)

        if action=="n":
            temporary_position=self.agent_position+self.num_columns 		

            if temporary_position<self.number_of_states:
                new_position=int(temporary_position)
                return self.action_rewards[new_position]+(self.gamma*self.state_values[new_position])

            else:
                return self.action_rewards[self.agent_position]+self.gamma*self.state_values[self.agent_position]   # the negative reward assoicated with it, should be -1.

        elif action=="s":
            temporary_position=self.agent_position-self.num_columns

            if temporary_position>=0:
                new_position=int(temporary_position)
                return self.action_rewards[new_position]+self.gamma*self.state_values[new_position]
            
            else:
                return self.action_rewards[self.agent_position]+self.gamma*self.state_values[self.agent_position]

        elif action=="w":
            temporary_position=self.agent_position-1

            if temporary_position % self.num_columns < (self.num_columns-1):
                new_position=int(temporary_position)
                return self.action_rewards[new_position]+self.gamma*self.state_values[new_position]

            else:
                return self.action_rewards[self.agent_position]+self.gamma*self.state_values[self.agent_position]

        elif action=="e":
            temporary_position=self.agent_position+1 # if it is able to move to the e.

            if temporary_position % self.num_columns >0:
                new_position= int(temporary_position) 
                return self.action_rewards[new_position]+self.gamma*self.state_values[new_position] #-1 plus value of that action.

            else: 
                return  self.action_rewards[self.agent_position]+self.gamma*self.state_values[self.agent_position]



class Value_Iteration:
    def value_iteration(self,environment,epsilon=1e-10,terminal_states=[18,23]):

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

            if change <epsilon: 
                break
        print("Optimal Policy")
        print(environment.policy)
        print("Optimal Value Function")
        print(environment.state_values)

class Action:
    environment=Environment(gold_state=23,bomb_state=18)
    iteration=Value_Iteration()
    iteration.value_iteration(environment)
