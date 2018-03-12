import numpy as np
import csv
import _pickle as pickle
from svmutil import *

def get_data(filepath):
	data = []
	with open(filepath, newline='') as csvfile:
		datareader = csv.reader(csvfile, delimiter=',')
		for row in datareader:
			data.append(list(map(int, row)))

	data = np.array(data)
	return data[:, :784], data[:, 784]

if __name__ == '__main__':
	X, y = get_data('mnist/train.csv')
	prob  = svm_problem(y.tolist(), (X / 255).tolist())

	file = open('cross_validation_results/cross_validation_accuracy.txt', 'w')
	file.write('C\t Accuracy\n')
	file.close()
	i = 1
	for c in [1e-5, 1e-3, 1, 5, 10]:
		m = svm_train(prob, '-s 0 -t 2 -c ' + str(c) + ' -g 0.05 -v 10')
		file = open('cross_validation_results/cross_validation_accuracy.txt', 'a')
		file.write(str(c) + '\t ' + str(m) + '\n')
		file.close()
		model = svm_train(prob, '-s 0 -t 2 -c ' + str(c) + ' -g 0.05')
		svm_save_model('cross_validation_results/smo_model_c_' + str(i) + '.model', model)
		i += 1
	file.close()