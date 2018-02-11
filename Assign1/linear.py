import linear_regression as rg
import dataprocessing as dp
import numpy as np
import matplotlib.pyplot as plt


# Loading training examples
X_org = dp.load_data('linearX.csv')
y = dp.load_data('linearY.csv')

X_org = X_org.reshape((-1, 1))

#Normalizing the training examples
X, meu, sigma = rg.normalize(X_org)

m, n = X.shape
init_theta = np.zeros(n + 1)
eta = 0.003
epsilon = 1e-10

theta, iterations = rg.linear_reg(X, y, init_theta, eta, epsilon)

#Compensating for normalization i.e changing to orginal variables

theta[0] = theta[0] - np.sum(theta[1:] * meu / sigma);
theta[1:] = theta[1:] / sigma

# Plotting the data and hypothesis
dp.plot_training_data(X_org, y, 'Wine density vs acidity', 'Wine acidity', 'Wine density')

print('No. of iterations = ', iterations)
print('Theta = ', theta)

input('Press Enter to draw hypothesis')

#Plotting hypothesis
x_test = np.linspace(np.amin(X_org) - 1, np.amax(X_org) + 1, num=500)
y_test = rg.pred(x_test.reshape((-1, 1)), theta)
dp.plot_hypothesis(x_test, y_test)

input('Press Enter to close')
dp.plot_close()