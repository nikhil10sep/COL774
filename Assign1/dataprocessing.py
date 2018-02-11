import numpy as np
import matplotlib.pyplot as plt
import tkinter
import _tkinter

# Returns the data in numpy matrix form
def load_data (filename):
	return np.genfromtxt('Assignment_1_datasets/' + filename, delimiter=',')

def plot_training_data (x, y, title='', xlabel='', ylabel='', show=False):
	plt.ion()
	plt.plot(x, y, 'rx')
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.draw()

def plot_hypothesis(x_test, y_test):
	ax = plt.axis()
	plt.plot(x_test, y_test, color='blue')
	plt.axis(ax)
	plt.draw()

def plot_close():
	plt.close()


