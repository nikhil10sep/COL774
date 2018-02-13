import linear_regression as rg
import dataprocessing as dp
import numpy as np
import matplotlib.pyplot as plt

# Loading training examples
X = dp.load_data('linearX.csv')
y = dp.load_data('linearY.csv')

X = X.reshape((-1, 1))

#Normalizing the training examples
X, meu, sigma = dp.normalize(X)

#Initialising parameters for gradient descent
m, n = X.shape
init_theta = np.zeros(n + 1)
epsilon = 1e-10
eta = [0.001, 0.005, 0.009, 0.013, 0.017]
color = ['red', 'blue', 'green', 'yellow', 'magenta']

plt.ion()
for i in range(len(eta)): 
# Executing gradient descent
	theta, iterations, theta_history, cost_history = rg.linear_reg(X, y, init_theta, eta[i], epsilon)
	plt.plot(list(range(0, iterations + 1)), cost_history, color=color[i], label='eta = ' + str(eta[i]))

plt.xlabel('iterations')
plt.ylabel('Cost')
plt.title('Cost vs iterations')
plt.legend(loc=1)
plt.show()

input('Enter to close:')