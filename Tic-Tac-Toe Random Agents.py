import numpy as np
import matplotlib.pyplot as plt


class Board:

	def __init__(self,number_of_rows=3,number_of_columns=3,agent1="X",agent2="O"):
		"""
		Initializes a board object.
		"""

		self.rows=number_of_rows
		self.columns=number_of_columns
		self.board=np.ones(shape=(self.rows,self.columns),dtype=str) 
		self.agent1=agent1
		self.agent2=agent2
		
	def getboard(self):

		return self.board

	def showboard(self):

		for row in range(self.rows):
			print(' _______________________')
			print("|  ",end= '')
			for column in range(self.columns):
				print(self.board[row][column],"  |  ", end= '',sep='  ')
			print(end="\n")
		print(' _______________________')


	def check_vertical_win(self):

		for row in range(self.rows):

			if self.board[row][0]=="X" and self.board[row][0]==self.board[row][1] and self.board[row][0]==self.board[row][2]:
				return 0

			if self.board[row][0]=="O" and self.board[row][0]==self.board[row][1] and self.board[row][0]==self.board[row][2]:
				return 1


	def horizontal_check_win(self):

		for column in range(self.columns):

			if self.board[0][column]=="X" and self.board[0][column]==self.board[1][column] and self.board[0][column]==self.board[2][column]:
				return 0

			if self.board[0][column]=="O" and self.board[0][column]==self.board[1][column] and self.board[0][column]==self.board[2][column]:
				return 1

	def top_bottom_diagonal_check_win(self):

		if self.board[0][0]== "X" and  self.board[0][0] ==self.board[1][1] and  self.board[0][0]==self.board[2][2]:
			return 0

		if self.board[0][0]== "O" and  self.board[0][0] ==self.board[1][1] and  self.board[0][0]==self.board[2][2]:
			return 1



	def bottom_up_diagonal_check_win(self):

		if self.board[0][2]=="X" and self.board[0][2]==self.board[1][1] and self.board[0][2]==self.board[2][0]:
			return 0
		if self.board[0][2]=="O" and self.board[0][2]==self.board[1][1] and self.board[0][2]==self.board[2][0]:
			return 1
	
	def checkwin(self):

		return self.check_vertical_win(), self.horizontal_check_win(),self.bottom_up_diagonal_check_win(),self.top_bottom_diagonal_check_win()


	def place_a_symbol(self,agent_number):
		available_positions=np.argwhere((self.board!="X") & (self.board!="O"))

		random_row_2D=(available_positions[np.random.choice(available_positions.shape[0], 1, replace=False), :])
		random_row_1D=tuple(random_row_2D.flatten())
		if agent_number==0:

			self.board[random_row_1D]="X"
		elif agent_number==1:
			self.board[random_row_1D]="O"


	def game_over(self):

		if len(np.argwhere(self.board=="1"))==0: 
			return True

        
class Game:
	"""
	Simulates a Tic-Tac-Toe Game for 2 random agents, for 100 episodes.
	The cumulative returns for both agents then plotted.
	"""

	episode_list1=np.zeros(100)
	episode_list2=np.zeros(100)

	cumulative_reward1=0
	cumulative_reward2=0

	for episode in range(100):

		board=Board()
		win=False
		draw=False
		choice=[0,1] 
		counter=0
		agent1_reward=0
		agent2_reward=0
		global agent_number

		while draw==False and win==False: 

			if board.game_over():
				draw=True
				#print("Draw !!")
				break

			if counter==0: 

				agent_number=np.random.choice(choice)
				
				board.place_a_symbol(agent_number)
				#print("Agent",agent_number+1,"starts")
				counter=1
				if agent_number==0:
					agent_number=1
				elif agent_number==1:
					agent_number=0
					

			else:

				if  agent_number==0:
					board.place_a_symbol(agent_number)
					agent_number=1


				elif agent_number==1:
					 board.place_a_symbol(agent_number)
					 agent_number=0

			vert_win,hor_win,top_bottom_win,bottom_up_win=board.checkwin() 

			if any([vert_win==0,hor_win==0,top_bottom_win==0,bottom_up_win==0]):
				agent1_reward=1
				agent2_reward=-1
				win=True
				#print("Agent 1 won!")
				break

			if any([vert_win==1,hor_win==1,top_bottom_win==1,bottom_up_win==1]):
				
				agent2_reward=1
				agent1_reward=-1
				win=True
				#print("Agent 2 won!!")
				break

		#board.showboard()
		cumulative_reward1+=agent1_reward
		cumulative_reward2+=agent2_reward

		episode_list1[episode]=cumulative_reward1
		episode_list2[episode]=cumulative_reward2

	plt.plot(episode_list1,label="Agent1")
	plt.plot(episode_list2,label='Agent2')
	plt.xlabel("Episodes")
	plt.ylabel("Return")
	plt.legend()
	plt.title("Cumulative Return for two Random Agents-100 Episodes")
	plt.show()