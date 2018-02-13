import numpy as np

# Retuens hypothesis for m examples given training examples and theta
# X assumed to have x0 column
def pred(X, theta):
	return 1 / (1 + np.exp(-X.dot(theta)))

# Likelihood function L(X, y, theta)
# X is assumed to have x0 column
def likelihood(X, y, theta):
	hypo = pred(X, theta)
	return np.sum(y * np.log(hypo) + (1 - y) * np.log(1 - hypo))

# Gradient for likelihood function L(X, y, theta)
# X assumed to have extra column for x0
def gradient(X, y, theta):
	return np.dot(X.T, y - pred(X, theta))

# Hessian of likelihood function L(X, y, theta)
# X assumed to have extra column for x0
def hessian(X, y, theta):
	hypo = pred(X, theta)
	return -(X.T * hypo) @ ((X.T * (1 - hypo)).T)

# Newton's Method
def newton(X, y, init_theta, epsilon):
	m, n = X.shape
	Xp = np.ones((m, n + 1))
	Xp[:, 1:] = X
	theta = np.array(init_theta)
	prev_val = 0.0
	iterations = 0
	while True:
		theta = theta - (np.linalg.pinv(hessian(Xp, y, theta)).dot(gradient(Xp, y, theta)))
		val = likelihood(Xp, y, theta)
		iterations += 1
		if abs(val - prev_val) < epsilon:
			break
		prev_val = val

	return theta, iterations