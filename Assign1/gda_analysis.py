import gda
import dataprocessing as dp
import numpy as np

# Loading training examples
X = dp.load_data('q4x.dat', delim='  ')
y_text = dp.load_data('q4y.dat', dtype='str')

m, n = X.shape
y = np.zeros(m, dtype='int')

# Alaska - 0; Canada - 1
for i in range(len(y)):
	if y_text[i] == 'Canada':
		y[i] = 1

# # Normalizing the training examples
# X, meu, sigma = dp.normalize(X)

# Runnig GDA with sigma0 = sigma1
# meu = [mue0, meu1]
phi, meu, sigma = gda.gda(X, y)

print('meu0 = ', meu[0], ' meu1 = ', meu[1])
print('sigma :\n', sigma)

# Plotting the data and hypothesis
dp.plot_classification_data(X, y, ['Alaska', 'Canada'], 'Salmons from Alaska and Canada', 'Fresh Water', 'Marine Water')

input('Press Enter to draw Hypothesis:')

#Computing logistic regression equivalent theta parameter for linear boundary
theta = gda.linear_params(phi, meu, sigma)

#Plotting linear decision boundary
x_test = np.linspace(np.amin(X[:, 0]) - 1, np.amax(X[:, 0]) + 1, num=500)

dp.plot_linear_decision_boundary(x_test, theta)

input('Press Enter to calculate GDA params in general setting:')

# Runnig GDA with in general setting
# meu = [meu0, meu1]
# sigma = [sigma0, sigma1]
phi, meu, sigma = gda.gda(X, y, linear=False)

print('meu0 = ', meu[0], ' meu1 = ', meu[1])
print('sigma0 :\n', sigma[0])
print('sigma1 :\n', sigma[1])

input('Press Enter to draw hypothesis in general setting:')

# Calculating P(y = 1|X) on the mesh nfor contour
sig_inv = [np.linalg.pinv(sigma[0]), np.linalg.pinv(sigma[1])]
det = np.array([np.linalg.det(sigma[0]), np.linalg.det(sigma[1])])
det_log = np.log(det)
log_phi = np.log(phi / (1 - phi))

x1_range = (np.amin(X[:, 0]) - 1, np.amax(X[:, 0]) + 1)
x2_range = (np.amin(X[:, 1]) - 1, np.amax(X[:, 1]) + 1)

x1 = np.linspace(x1_range[0], x1_range[1], 100)
x2 = np.linspace(x2_range[0], x2_range[1], 100)
X1, X2 = np.meshgrid(x1, x2)

Z = np.zeros(X1.shape)
for i in range(len(x2)):
	for j in range(len(x1)):
		x_t = np.array([X1[i][j], X2[i][j]])
		p0 = np.dot(sig_inv[0].dot(x_t - meu[0]), x_t - meu[0])
		p1 = np.dot(sig_inv[1].dot(x_t - meu[1]), x_t - meu[1])
		Z[i][j] = p1 - p0 - 2 * log_phi + det_log[1] - det_log[0]

# Plotting quadratic boundary
dp.plot_contour(X1, X2, Z, val=[0])

input('Press Enter to close:')
dp.plot_close()