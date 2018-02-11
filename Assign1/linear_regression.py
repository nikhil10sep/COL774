import numpy as np

# Retuens hypothesis for m examples given training examples and theta
# x0 column is not present in X
def pred(X, theta):
	m, n = X.shape
	Xp = np.ones((m, n + 1))
	Xp[:, 1:] = X
	return Xp.dot(theta)

# Cost function J(X, y, theta)
# X assumed to have extra column for x0 
def cost(X, y, theta):
	error = X.dot(theta) - y
	return np.dot(error.T, error) / 2

# Gradient for cost function J(X, y theta)
# X assumed to have extra column for x0
def gradient(X, y, theta):
	return np.dot(X.T, X.dot(theta) - y)

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

# Normal equation method
# W = Default[None] for unweighted parameters
# TODO check @ for multiplication
def analytical_sol(X, y, W=None):
	m, n = X.shape
	Xp = np.ones((m, n + 1))
	Xp[:, 1:] = X
	if W is None:
		return (np.linalg.pinv(Xp.T @ Xp) @ Xp.T).dot(y)

	XTW = Xp.T @ W
	return (np.linalg.pinv(XTW @ Xp) @ XTW).dot(y)

# Weighted Linear regression
#Inputs X, y, x_test(Points where hypothesis needs to be ewaluated), tau
def weighted_linear_reg(X, y, X_test, tau):
	hypothesis = []
	for x_t in X_test:
		weights = np.exp(-np.sum((x_t - X) * (x_t - X), axis=1) / (2 * tau * tau))
		W = np.diag(weights)
		theta = analytical_sol(X, y, W)
		hypothesis.append(theta[0] + theta[1:].dot(x_t))

	return np.array(hypothesis)
