import numpy as np
import matplotlib.pyplot as plt
import threading

# Returns the data in numpy matrix form
def load_data (filename):
	return np.genfromtxt('Assignment_1_datasets/' + filename, delimiter=',')

def plot_data (x, y, theta, title='', xlabel='', ylabel=''):
	thread_draw = threading.Thread(target=draw_plot, args=(x, y, theta, title, xlabel, ylabel))
	thread_draw.start()

#Ploting thread
def draw_plot (x, y, theta, title='', xlabel='', ylabel=''):
	plt.plot(x, y, 'rx')
	x = np.append(x, [[np.amin(x) - 1], [np.amax(x) + 1]], axis=0)
	plt.plot(x, theta[0] + theta[1] * x, color='blue')
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()