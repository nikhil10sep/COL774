import linear_regression as rg
import dataprocessing as dp
import numpy as np


# Loading training examples
X = dp.load_data('weightedX.csv')
y = dp.load_data('weightedY.csv')

X = X.reshape((-1, 1))

#Analytical solution (without weights)
theta = rg.analytical_sol(X, y)

# Plotting the data
dp.plot_training_data(X, y, 'Wine density vs acidity', 'Wine acidity', 'Wine density')

print('Theta = ', theta)

input('Press Enter to draw hypothesis')

#Plotting hypothesis
x_test = np.linspace(np.amin(X) - 1, np.amax(X) + 1, num=500)
y_test = rg.pred(x_test.reshape((-1, 1)), theta)
dp.plot_hypothesis(x_test, y_test)

input('Press Enter to close')
dp.plot_close()

#Plotting training data
dp.plot_training_data(X, y, 'Wine density vs acidity', 'Wine acidity', 'Wine density')

#Analytical solution (with weights)
tau = 1

y_test = rg.weighted_linear_reg(X, y, x_test.reshape(-1,1), tau)

input('Press Enter to draw hypothesis')

#Plotting weighted linear regression hypothesis
dp.plot_hypothesis(x_test, y_test)


input('Press Enter to close')
dp.plot_close()