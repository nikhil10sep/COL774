import numpy as np
import matplotlib.pyplot as plt


# Returns the data in numpy matrix form
def load_data (filename):
	return np.genfromtxt('Assignment_1_datasets/' + filename, delimiter=',')

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

def plot_training_data (x, y, title='', xlabel='', ylabel=''):
	plt.ion()
	plt.plot(x, y, 'rx')
	plot_set_labels(title, xlabel, ylabel)
	plt.draw()

def plot_hypothesis(x_test, y_test):
	ax = plt.axis()
	plt.plot(x_test, y_test, color='blue')
	plt.axis(ax)
	plt.draw()

def plot_classification_data(X, y, title='', xlabel='', ylabel=''):
	plt.ion()
	for i in range(len(y)):
		if y[i] == 0:
			plt.scatter(X[i][0], X[i][1], facecolors='none', edgecolors='y', marker='o')
		else:
			plt.scatter(X[i][0], X[i][1], c='r', marker='+')
	plot_set_labels(title, xlabel, ylabel)
	plt.draw()

# Plots linear decision boundary given theta (belongs to R^3)
# x_test represents x1 column of training data
def plot_linear_decision_boundary(x_test, theta):
	ax = plt.axis()
	plt.plot(x_test, (-theta[0] - theta[1] * x_test) / theta[2], color='blue')
	plt.axis(ax)
	plt.draw()

def plot_close():
	plt.close()


