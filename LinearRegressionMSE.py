import numpy as np

class LinearRegression:
	def __init__(self):
		this.theta = np.empty(0)
		this.bias = 0
	
	def fit(self, X, y, alpha = 0.1, epochs = 100):
	 '''
		This function is used to fit the data using MSE loss.
		The function updates self.theta which will pe used to predict the values
		in predict function. 

		Parameters:
		----------
		X - Features
		y - Labels 
		alpha - Learning rate
		epochs - Number of epochs

		Returns: self
	'''
		# No of data points in dataset
        n = len(y)   

        # Initializing the theta and bias
        self.theta = np.zeros(X.shape[1])
        self.bias = 0
		
		for i in range(epochs):
			# Predicted value = X matrix multiplied with theta
			# Difference between the predicted and true value for training set
			error = X @ self.theta + self.bias - y

			# Gradint descent : theta = theta - (learning_rate/n)*(partial derivative of cost function)
			self.theta -= alpha*(1/n)*(np.sum([error[j] * X[j] for j in range(n)]))
			self.bias -= alpha*(1/n)*(np.sum(error))
		
		return self
	
	def predict(self, X):
	'''
        Predicting values using the trained linear model.

        Parameters
        ----------
        X - 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y - 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
   	'''
		return X @ self.theta + self.bias
