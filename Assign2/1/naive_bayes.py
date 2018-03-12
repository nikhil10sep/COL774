import string
import numpy as np
from scipy import sparse as sps
import matplotlib.pyplot as plt
import argparse
import _pickle as pickle

def get_processed_data(file_path):
	file = open(file_path, 'r')

	punc = string.punctuation
	table = str.maketrans(punc, ' ' * len(punc))
	
	printable = set(string.printable)
	data = []
	for line in file:
		review =  ''.join(s for s in line if s in printable)
		words = line.translate(table).lower().split()
		if len(words) != 0:
			data.append(words)

	file.close()
	return data

def get_dictionary(data):
	dictionary, index = {}, 0
	for line in data:
		for word in line:
			if not word in dictionary:
				dictionary[word] = index
				index += 1
	return dictionary

def get_labels(file_path):
	return np.genfromtxt(file_path, dtype=int)

def get_class_labels(file_path):
	labels = np.genfromtxt(file_path, dtype=int)

	classes = sorted(set(labels))
	label_map, inv_label_map = {}, {}
	for i in range(0, len(classes)):
		label_map[classes[i]] = i
		inv_label_map[i] = classes[i]

	row, column = [], []
	i = 0
	for label in labels:
		row.append(label_map[label])
		column.append(i)
		i += 1

	class_labels = sps.coo_matrix(((np.ones(len(row))), (row, column)), shape=(len(classes), len(labels))).tocsr()
	return label_map, inv_label_map, class_labels

def get_features(data, dictionary, l=False):
	m, V = len(data), len(dictionary)

	i = 0
	row, column = [], []
	for line in data:
		for word in line:
			if word in dictionary:
				row.append(i)
				column.append(dictionary[word])
				if l and i % 40 == 0:
					row.append(i)
					column.append(V)
		i += 1

	if l:
		return sps.coo_matrix(((np.ones(len(row))), (row, column)), shape=(m, V + 1)).tocsr()

	return sps.coo_matrix(((np.ones(len(row))), (row, column)), shape=(m, V)).tocsr()

def naive_bayes(features, class_labels, c = 1):
	class_features = (class_labels @ features).toarray()
	m, V = features.shape

	label_sum = (class_labels.sum(axis=1).A).flatten()
	
	phi_j = (class_features + c) / (class_features.sum(axis=1)[:,None] + V * c)
	return (np.log(phi_j), np.log(label_sum / m))

def predict(features, model, inv_label_map):
	f_t = features.transpose()
	log_phi_j, log_phi_y = model[0], model[1]

	prob = log_phi_j @ f_t + log_phi_y.reshape((log_phi_y.shape[0], 1))

	pred = []
	p = prob.argmax(axis=0)
	for i in range(features.shape[0]):
		pred.append(inv_label_map[p[i]])

	return np.array(pred)

def get_accuracy(truth_labels, pred):
	correct, count = 0, 0
	for i in range(0, len(truth_labels)):
		if truth_labels[i] == pred[i]:
			correct += 1
		count += 1

	return correct / count

def draw_confusion_matrix(truth_labels, pred, label_map):
	conf_arr = np.zeros((len(label_map), len(label_map)), dtype=int)

	for i in range (0, len(truth_labels)):
		conf_arr[label_map[truth_labels[i]]][label_map[pred[i]]] += 1

	plt.ion()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(conf_arr / np.linalg.norm(conf_arr)), cmap=plt.cm.Greens, interpolation='nearest')
	width, height = conf_arr.shape

	for x in range(width):
		for y in range(height):
			ax.annotate(str(conf_arr[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

	plt.xticks(range(width), list(label_map.keys()))
	plt.yticks(range(height), list(label_map.keys()))
	plt.title('Confusion Matrix')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('train_text', type=str, help='Training text file')
	parser.add_argument('train_labels', type=str, help='Training text labels file')
	parser.add_argument('output_model', type=str, help='Output file for trained model')
	parser.add_argument('-l', '--length', action='store_true', help='Take document length as feature')
	parser.add_argument('-p', '--pred_file', type=str, help='Prediction output file [Default: pred.out]')
	parser.add_argument('-tt', '--test_text', type=str, help='Testing text file')
	parser.add_argument('-tl', '--test_labels', type=str, help='Testing labels file')
	parser.add_argument('-c', '--confusion_matrix', action='store_true')
	args = parser.parse_args()

	data = get_processed_data(args.train_text)
	dictionary = get_dictionary(data)

	features = None
	if args.length:
		features = get_features(data, dictionary, l=True)
	else:
		features = get_features(data, dictionary)

	label_map, inv_label_map, class_labels = get_class_labels(args.train_labels)

	model = naive_bayes(features, class_labels)

	with open(args.output_model, 'wb') as model_out:
		pickle.dump(dictionary, model_out)
		pickle.dump(label_map, model_out)
		pickle.dump(inv_label_map, model_out)
		pickle.dump(model, model_out)

	if args.test_text:
		test_features = None
		if args.length:
			test_features = get_features(get_processed_data(args.test_text), dictionary, l=True)
		else:
			test_features = get_features(get_processed_data(args.test_text), dictionary)
		
		pred_file = 'pred.out'
		if args.pred_file:
			pred_file = args.pred_file

		pred = predict(test_features, model, inv_label_map)
		with open(pred_file, 'w') as pred_out:
			for i in range(len(pred)):
				pred_out.write(str(pred[i]) + '\n') 

		if args.test_labels:
			test_labels = get_labels(args.test_labels)
			acc = get_accuracy(test_labels, pred)
			print('Test accuracy:', acc)

		if args.confusion_matrix:
			draw_confusion_matrix(test_labels, pred, label_map)
			input('Press Enter to exit:')