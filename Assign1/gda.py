import numpy as np

def gda(X, y, linear=True):
	m, n = X.shape

	phi = np.sum(y) / m
	meu = [0, 0]
	meu[0] = np.sum((1 - y) * X.T, axis=1) / (m - np.sum(y))
	meu[1] = np.sum(y * X.T, axis=1) / np.sum(y)

	if linear:
		sigma = np.zeros((n, n))
		for i in range(len(y)):
			sigma += np.outer(X[i] - meu[y[i]], X[i] - meu[y[i]])

		return phi, meu, sigma / m

	sigma = [np.zeros((n,n)), np.zeros((n,n))]
	for i in range(len(y)):
		sigma[y[i]] += np.outer(X[i] - meu[y[i]], X[i] - meu[y[i]])

	return phi, meu, [sigma[0] / (m - np.sum(y)), sigma[1] / np.sum(y)]

# Returns logistic regression equivalent theta parameter for linear boundary
# Input phi, meu = [meu0, meu1], sigma (sigma0 = sigma1 = sigma)
def linear_params(phi, meu, sigma):
	n = len(meu[0])
	theta = np.zeros(n + 1)
	sig_inv = np.linalg.pinv(sigma)

	theta[0] = -(np.dot(meu[1], sig_inv.dot(meu[1])) - np.dot(meu[0], sig_inv.dot(meu[0]))) / 2 - np.log((1 - phi) / phi)
	theta[1:] = sig_inv.dot(meu[1] - meu[0])

	return theta