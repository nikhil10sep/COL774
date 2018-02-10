import numpy as np

# Normalize the training examples
# Returns normalized examples, mean vector and variance vector
def normalize(X):
	mean = X.mean(axis=0, keepdims=1)
	sigma = X.std(axis=0, keepdims=1)
	return ((X - mean) / sigma, mean[0], sigma[0])

# Cost function J(X, y, theta)
# X assumed to have extra column for x0 
def cost(X, y, theta):
	error = X.dot(theta) - y
	return (np.dot(error.T, error)[0][0] / 2)

# Gradient for cost function J(X, y theta)
# X assumed to have extra column for x0
def gradient(X, y, theta):
	m = X.shape[0]
	return np.dot(X.T, np.dot(X, theta) - y)

# Gradient descent
def linear_reg(X, y, init_theta, eta, epsilon):
	m, n = X.shape
	Xp = np.ones((m, n + 1))
	Xp[:, 1:] = X
	theta = np.array(init_theta)
	prev_error = 0.0
	iterations = 0
	while True:
		theta = theta - eta * gradient(Xp, y, theta)
		error = cost(Xp, y, theta)
		if abs(error - prev_error) < epsilon:
			break
		prev_error = error
		iterations += 1

	return theta, iterations