import linear_regression as rg
import dataprocessing as dp
import numpy as np

# Loading training examples
X = dp.load_data('linearX.csv')
y = dp.load_data('linearY.csv')

X = X.reshape((-1, 1))

#Normalizing the training examples
X, meu, sigma = dp.normalize(X)

#Initialising parameters for gradient descent
m, n = X.shape
init_theta = np.zeros(n + 1)
eta = 0.017
epsilon = 1e-10

# Executing gradient descent
theta, iterations, theta_history, cost_history = rg.linear_reg(X, y, init_theta, eta, epsilon)

# Plotting the data and hypothesis
dp.plot_training_data(X, y, 'Wine density vs acidity', 'Wine acidity', 'Wine density')

print('No. of iterations = ', iterations)
print('Theta = ', theta)

input('Press Enter to draw Hypothesis:')

#Plotting hypothesis
x_test = np.linspace(np.amin(X) - 1, np.amax(X) + 1, num=500)
y_test = rg.pred(x_test.reshape((-1, 1)), theta)
dp.plot_hypothesis(x_test, y_test)

input('Press Enter to draw 3D Mesh:')
dp.plot_close()

# Creating J(theta) values for mesh
theta0 = np.arange(-1, 3, 0.05)
theta1 = np.arange(-1, 3, 0.05)
Theta0, Theta1 = np.meshgrid(theta0, theta1)

Xp = np.ones((m, n + 1))
Xp[:, 1:] = X

J = np.zeros(Theta0.shape)
for i in range(len(theta1)):
	for j in range(len(theta0)):
		J[i][j] = rg.cost(Xp, y, np.array([Theta0[i][j], Theta1[i][j]]))

# Draw Animation for gradient descent
dp.animate_contour_and_mesh(Theta0, Theta1, J, theta_history, cost_history)

input('Press Enter to close:')
dp.plot_close()