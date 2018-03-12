import numpy as np
import csv
import _pickle as pickle
import argparse
import random
import sys
import matplotlib.pyplot as plt
from svmutil import *

def get_data(filepath):
	data = []
	with open(filepath, newline='') as csvfile:
		datareader = csv.reader(csvfile, delimiter=',')
		for row in datareader:
			data.append(list(map(int, row)))

	data = np.array(data)
	return data[:, :784] / 255, data[:, 784]

def get_X(filepath):
	data = []
	with open(filepath, newline='') as csvfile:
		datareader = csv.reader(csvfile, delimiter=',')
		for row in datareader:
			data.append(list(map(int, row)))

	data = np.array(data)
	return data / 255

def get_class(X, y, label):
	ind = np.nonzero(y == label)[0]
	return X[ind, :]
	

def pegasos(X, y, epsilon=1e-3, batch_size=100, C=1, max_iters=2000):
	m, n = X.shape
	w = np.zeros(n)
	b = 0

	iters = 1
	while True:
		ind = random.sample(range(m), batch_size)
		A, y_t = X[ind, :], y[ind]
		t = ((A.dot(w) + b) * y_t)
		temp_ind = np.nonzero(t < 1)[0]
		delta = np.zeros(t.shape)
		delta[temp_ind] = 1
		eta =  1 / iters
		prev_w = w
		prev_b = b

		w = (1 - eta) * w + eta * C * ((delta * y_t) @ A)
		b = b + eta * C * (delta.dot(y_t))
		
		if iters == max_iters:
			break
		
		iters += 1
	return w, b

def train_pegasos(X, y):
	classes = list(map(int, set(y)))
	classes.sort()
	X_class = []

	for i in range(len(classes)):
		X_class.append(get_class(X, y, classes[i]))

	train = {}
	for i in range(len(classes)):
		for j in range(i + 1, len(classes)):
			y_i, y_j = -1 * np.ones(X_class[i].shape[0]), np.ones(X_class[j].shape[0])
			X_train = np.append(X_class[i], X_class[j], axis=0)
			y_train = np.append(y_i, y_j, axis=0)
 
			w, b = pegasos(X_train, y_train)
			train[(i, j)] = (w, b)

	return train, classes

def train_smo(X, y, params=None):
	model = svm_train(y, X, params)
	return model

def predict_smo(X, y, model):
	p_labs, p_acc, p_vals = svm_predict(y, X, model)
	return np.array(p_labs, dtype=int), p_acc[0]

def predict_pegasos(X, plane):
	dist = X.dot(plane[0]) + plane[1]
	pred = np.ones(X.shape[0])
	ind = np.nonzero(dist < 0)[0]
	pred[ind] = -1

	return pred

def pred_one_vs_one(X_test, train, classes):
	pred_count = np.zeros((X_test.shape[0], len(classes)))
	for key in train:
		pred_classes = predict_pegasos(X_test, train[key])
		ind_pos =  np.nonzero(pred_classes == 1)[0]
		ind_neg = np.nonzero(pred_classes == -1)[0]
		pred_count[ind_pos, key[1]] += 1
		pred_count[ind_neg, key[0]] += 1

	pred = np.zeros(X_test.shape[0])
	max_pred_count = np.max(pred_count, axis=1)
	for i in range(len(pred)):
		for j in range(len(classes)):
			if pred_count[i][j] == max_pred_count[i]:
				pred[i] = classes[j]

	return pred.astype(int)

def accuracy(pred, y_test):
	return len(np.nonzero(pred - y_test == 0)[0]) / len(pred)

def draw_confusion_matrix(truth_labels, pred, labels):
	truth_labels = truth_labels.astype(int)
	pred = pred.astype(int)
	conf_arr = np.zeros((len(labels), len(labels)), dtype=int)

	for i in range (0, len(truth_labels)):
		conf_arr[labels[truth_labels[i]]][labels[pred[i]]] += 1

	plt.ion()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(conf_arr / np.linalg.norm(conf_arr)), cmap=plt.cm.Greens, interpolation='nearest')
	width, height = conf_arr.shape

	for x in range(width):
		for y in range(height):
			ax.annotate(str(conf_arr[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

	plt.xticks(range(width), labels)
	plt.yticks(range(height), labels)
	plt.title('Confusion Matrix')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('training_file', type=str, help='Training file')
	parser.add_argument('method', type=int, help='Method (0:Pegasos, 1:Linear SMO, 2:RBF SMO')
	parser.add_argument('output_model', type=str, help='Output model file')
	parser.add_argument('-tt', '--test_file', type=str, help='Testing file')
	parser.add_argument('-p', '--pred_file', type=str, help='Prediction output file [Default: pred.out]')
	parser.add_argument('-c', '--C', type=float, help='C paramter [Default: 1]')
	parser.add_argument('-g', '--gamma', type=float, help='Gamma parameter in RBF Kernel [Default: 0.05]')
	parser.add_argument('-cf', '--confusion_matrix', action='store_true')
	args = parser.parse_args()

	X, y = get_data(args.training_file)
	y_test, pred,classes = None, None, None
	if args.method == 0:
		train, classes = train_pegasos(X, y)
		with open(args.output_model, 'wb') as model_out:
			pickle.dump(train, model_out)
			pickle.dump(classes, model_out)

		if args.test_file:
			X_test, y_test = get_data(args.test_file)
			pred = pred_one_vs_one(X_test, train, classes)
			print('Test Accuracy:', accuracy(pred, y_test))
		
	elif args.method == 1 or args.method == 2:
		X, y = get_data(args.training_file)
		classes = list(map(int, set(y)))
		classes.sort()
		C = 1
		if args.C:
			C = args.C

		model = None
		if args.method == 1:
			model = train_smo(X.tolist(), y.tolist(), '-q -s 0 -t 0 -c ' + str(C))
		else:
			g = 0.05
			if args.gamma:
				g = args.gamma
			model = train_smo(X.tolist(), y.tolist(), '-q -s 0 -t 2 -c ' + str(C) + ' -g ' + str(g))

		with open(args.output_model, 'wb') as model_out:
			svm_save_model(args.output_model + '.model', model)
			pickle.dump(args.output_model + '.model', model_out)
			pickle.dump(classes, model_out)

		X_test, y_test = get_data(args.test_file)
		pred, acc = predict_smo(X_test.tolist(), y_test.tolist(), model)
		print('Test Accuracy:', acc)

	else:
		print('Wrong method option: Method (0:Pegasos, 1:Linear SMO, 2:RBF SMO)')
		sys.exit(1)

	if args.test_file:
		pred_file = 'pred.out'
		if args.pred_file:
			pred_file = args.pred_file
		with open(pred_file, 'w') as pred_out:
			for i in range(len(pred)):
				pred_out.write(str(pred[i]) + '\n')

		if args.confusion_matrix:
			draw_confusion_matrix(y_test, np.array(pred), classes)
			input('Press Enter to exit:')