import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.lines as mlines

# Returns the data in numpy matrix form
def load_data (filename, delim=',', dtype=None):
	return np.genfromtxt('Assignment_1_datasets/' + filename, delimiter=delim, dtype=dtype)

# Normalize the training examples
# Returns normalized examples, mean vector and variance vector
def normalize(X):
	mean = X.mean(axis=0, keepdims=1)
	sigma = X.std(axis=0, keepdims=1)
	return ((X - mean) / sigma, mean[0], sigma[0])

def plot_set_labels(title, xlabel, ylabel):
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

# Scatter plot for training data
def plot_training_data (x, y, title='', xlabel='', ylabel=''):
	plt.ion()
	plt.plot(x, y, 'rx')
	plot_set_labels(title, xlabel, ylabel)
	plt.autoscale(enable=False)

# Learnt line of hypothesis
def plot_hypothesis(x_test, y_test):
	plt.plot(x_test, y_test, color='blue')

# Method for animation of gradient descent
def animate_contour_and_mesh(Theta0, Theta1, Z, theta_history, cost_history):
	plt.ion()

	fig_surface = plt.figure(1)
	ax_surface = fig_surface.add_subplot(111, projection = '3d')
	ax_surface.plot_surface(Theta0, Theta1, Z, alpha=0.75)
	ax_surface.set_autoscale_on(False)
	ax_surface.scatter(theta_history[0][0], theta_history[0][1], cost_history[0], c='red')
	
	ax_surface.set_xlabel('Theta0')
	ax_surface.set_ylabel('Theta1')
	ax_surface.set_zlabel('Cost')
	ax_surface.set_title('Surface plot (Cost)')

	fig_contour = plt.figure(2)
	ax_contour = fig_contour.add_subplot(111)
	ax_contour.contour(Theta0, Theta1, Z, 40)
	ax_contour.set_autoscale_on(False)
	ax_contour.scatter(theta_history[0][0], theta_history[0][1], c='red')
	
	ax_contour.set_xlabel('Theta0')
	ax_contour.set_ylabel('Theta1')
	ax_contour.set_title('Contour plot (Cost)')

	input("Press Enter to start Animation:")

	for i in range(1, len(theta_history)):
		ax_surface.plot([theta_history[i - 1][0], theta_history[i][0]], [theta_history[i - 1][1], theta_history[i][1]], [cost_history[i - 1], cost_history[i]], c='yellow')
		ax_surface.scatter(theta_history[i][0], theta_history[i][1], cost_history[i], c='red')
		
		ax_contour.plot([theta_history[i - 1][0], theta_history[i][0]], [theta_history[i - 1][1], theta_history[i][1]], c='yellow')
		ax_contour.scatter(theta_history[i][0], theta_history[i][1], c='red')
		
		ax_contour.set_title('Iteration number: ' + str(i))
		plt.pause(0.2)

# Scatter plot for training data with labelling for different classes
def plot_classification_data(X, y, label, title='', xlabel='', ylabel=''):
	classes = [[[], []], [[], []]]
	for i in range(len(y)):
		classes[y[i]][0].append(X[i][0])
		classes[y[i]][1].append(X[i][1])

	plt.ion()
	plt.scatter(classes[0][0], classes[0][1], facecolors='none', edgecolors='y', marker='o', label=label[0])
	plt.scatter(classes[1][0], classes[1][1], c='r', marker='+', label=label[1])
	plot_set_labels(title, xlabel, ylabel)
	plt.legend(loc=2)
	plt.autoscale(enable=False)

# Plots linear decision boundary given theta (belongs to R^3)
# x_test represents x1 column of training data
def plot_linear_decision_boundary(x_test, theta):
	plt.plot(x_test, (-theta[0] - theta[1] * x_test) / theta[2], color='blue', label='Linear Boundary')
	plt.legend(loc=2)

# Plots a contour
def plot_contour(X, Y, Z, val=None):
	plt.contour(X, Y, Z, val)

def plot_close():
	plt.close('all')


