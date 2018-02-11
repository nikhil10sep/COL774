import logistic_regression as log_rg
import dataprocessing as dp
import numpy as np

# Loading training examples
X_org = dp.load_data('logisticX.csv')
y = dp.load_data('logisticY.csv')

#Normalizing the training examples
X, meu, sigma = dp.normalize(X_org)

m, n = X.shape
init_theta = np.zeros(n + 1)
epsilon = 1e-10

theta, iterations = log_rg.newton(X, y, init_theta, epsilon)

#Compensating for normalization i.e changing to orginal variables
theta[0] = theta[0] - np.sum(theta[1:] * meu / sigma);
theta[1:] = theta[1:] / sigma

# Plotting the data and hypothesis
dp.plot_classification_data(X_org, y)

print('No. of iterations = ', iterations)
print('Theta = ', theta)

input('Press Enter to draw hypothesis')

#Plotting decision boundary
x_test = np.linspace(np.amin(X_org[:, 0]) - 1, np.amax(X_org[:, 0]) + 1, num=500)

dp.plot_linear_decision_boundary(x_test, theta)

input('Press Enter to close')
dp.plot_close()