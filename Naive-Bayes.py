
#https://www.youtube.com/watch?v=CPqOCI0ahss&t=1s
import numpy as np
training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
#print("Shape of the spam training data set:", training_spam.shape)
#print(training_spam)


#Part A: Estimate class priors (20 marks)

def estimate_log_class_priors(data):
	"""
	Given a data set with binary response variable (0s and 1s) in the
	left-most column, calculate the logarithm of the empirical class priors,
	that is, the logarithm of the proportions of 0s and 1s:
	log(P(C=0)) and log(P(C=1))

	:param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
				 the first column contains the binary response (coded as 0s and 1s).

	:return log_class_priors: a numpy array of length two
	"""
	### YOUR CODE HERE...

	
	target_variable=data[:,[0]]
	

	class_0_count,class_1_count=calculate_proportion(target_variable)
	
	log_class_0= np.log(class_0_count/len(target_variable))
	#print(np.log(log_class_0))
	log_class_1= np.log(class_1_count/len(target_variable))
	log_class_priors=np.array([log_class_0,log_class_1])
	return log_class_priors



def calculate_proportion(y_values):
	class_0_count=0
	class_1_count=0   
	
	for i in range(len(y_values)):
		
		if y_values[i]==0:
			class_0_count+=1
		elif y_values[i]==1:
			class_1_count+=1
   
	return class_0_count,class_1_count 
	


log_class_priors=estimate_log_class_priors(training_spam)


def estimate_log_class_conditional_likelihoods(data, alpha=1.0):
	"""
	Given a data set with binary response variable (0s and 1s) in the
	left-most column and binary features (words), calculate the empirical
	class-conditional likelihoods, that is,
	log(P(w_i | c)) for all features w_i and both classes (c in {0, 1}).

	Assume a multinomial feature distribution and use Laplace smoothing
	if alpha > 0.

	:param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]

	:return theta:
		a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
		logarithm of the probability of feature i appearing in a sample belonging 
		to class j.
	"""
	### YOUR CODE HERE...

	target_variable=data[:,[0]]
	number_of_hams,number_of_spams = calculate_proportion(target_variable)
	#print(number_of_spams)
	features=data.shape[1]-1

	theta=np.empty((2,0),dtype=float) # the columns being 0 allowed to work succesfully.np.empty((2,54) appends at the end of the array.
									  # same for all other methods.
	

	# ete column wise append kenes, bdi arrayd erku row unena anpayman.
	#Nc_ham=0
	#Nc_spam=0


	
	# for column in range(data.shape[1]):

	# 	if column==0:
	# 		continue

	# 	independent_var=data[:,[column]]

	# 	for i in range(len(target_variable)): # lave or ashxatel e normal, vorovhtev target variable 2d shape uni, 1 d keneir uxaki data[:,0]

	# 		if target_variable[i]==0 and independent_var[i]==1:
	# 			Nc_ham+=1
			
	# 		elif target_variable[i]==1 and independent_var[i]==1:
	# 			Nc_spam+=1	

	def count_N_c():
		
		Nc_ham, Nc_spam=0,0
		
		for column in range(data.shape[1]):

			if column==0:
				continue

			independent_var=data[:,[column]]

			for i in range(len(target_variable)): # lave or ashxatel e normal, vorovhtev target variable 2d shape uni, 1 d keneir uxaki data[:,0]

				if target_variable[i]==0 and independent_var[i]==1:
					Nc_ham+=1
				
				elif target_variable[i]==1 and independent_var[i]==1:
					Nc_spam+=1	


		return Nc_ham, Nc_spam



	Nc_ham,Nc_spam=count_N_c()

	for column in range(data.shape[1]):
		"""
		Iterate over the whole dataframe except the first target variable.
		Keep on calculating, and appending to the new dataframe.
		"""

		count_wi_given_ham,count_wi_given_spam= 0,0
		
		if column==0:
			continue
		
		independent_var=data[:,[column]] # this is 2d, lav e vor iterate kexni.
		

		for i in range(len(target_variable)):	

			if target_variable[i]==0 and independent_var[i]==1:
				count_wi_given_ham+=1

			
			elif target_variable[i]==1 and independent_var[i]==1:
				count_wi_given_spam+=1    	
		

			
		log_theta_class0_word=np.log((count_wi_given_ham+alpha)/(Nc_ham+features*alpha))	
		log_theta_class1_word=np.log((count_wi_given_spam+alpha)/(Nc_spam+features*alpha))	
		

		#a = np.array([[1], [2], [3]]) --this has 3 rows and 1 column..
		#a = np.array([[1,2,3]]) -- 2D array, matrix form, 1 row and three columns.
		#a = np.array([[1,2,3],[4,5,6]]) -- 2D array, matrix form, 2 rows and three columns.

		myarray=np.array([[log_theta_class0_word],[log_theta_class1_word]]) #aysinqn 2d array with 1 column and 2 rows. amen nersi listy 1 hat row e.
		theta=np.append(theta,myarray,axis=1)
		
		#theta=np.hstack((theta,myarray)) this also works.
		#theta=np.column_stack((theta,myarray)) this also works.
		#theta=np.concatenate((theta,myarray),axis=1) # this also works.


	print(theta)
	return theta


log_class_conditional_likelihoods=estimate_log_class_conditional_likelihoods(training_spam)

def predict(new_data, log_class_priors, log_class_conditional_likelihoods):
	"""
	Given a new data set with binary features, predict the corresponding
	response for each instance (row) of the new_data set.

	:param new_data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].
	:param log_class_priors: a numpy array of length 2.
	:param log_class_conditional_likelihoods: a numpy array of shape = [2, n_features].
		theta[j, i] corresponds to the logarithm of the probability of feature i appearing
		in a sample belonging to class j.
	:return class_predictions: a numpy array containing the class predictions for each row
		of new_data.
	"""
	### YOUR CODE HERE...

	
	def myfunction(vector):
		
		ham_label_prob=  log_class_priors[0] +  np.dot(vector,log_class_conditional_likelihoods[0])
		spam_label_prob= log_class_priors[1]  +  np.dot(vector,log_class_conditional_likelihoods[1])

		my_list=[]

		my_list.append(0) if ham_label_prob>spam_label_prob else my_list.append(1)

		myarray=np.array(my_list)

		return myarray

		

	# sovorakan array-y row u column chuni.
	class_predictions=np.apply_along_axis(myfunction,1, new_data) # aranc flattern enelu el khashve
	class_predictions=class_predictions.flatten()
	print(class_predictions.shape)	
	
	return class_predictions

def accuracy(y_predictions, y_true):
	"""
	Calculate the accuracy.
	
	:param y_predictions: a one-dimensional numpy array of predicted classes (0s and 1s).
	:param y_true: a one-dimensional numpy array of true classes (0s and 1s).
	
	:return acc: a float between 0 and 1 
	"""
	### YOUR CODE HERE...
	count=0
	for i,j in zip(y_predictions,y_true):
		if i==j:
			count+=1

	acc=count/len(y_true)
	
	return acc


class_predictions=predict(training_spam[:, 1:], log_class_priors, log_class_conditional_likelihoods)
a=accuracy(class_predictions,training_spam[:, 0])
print(a)



testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")
print("Shape of the testing spam data set:", testing_spam.shape)
# testing_set_accuracy = ...
class_predictions=predict(testing_spam[:, 1:], log_class_priors, log_class_conditional_likelihoods)
testing_set_accuracy=accuracy(class_predictions,testing_spam[:, 0])
print(testing_set_accuracy)

