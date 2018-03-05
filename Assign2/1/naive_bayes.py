import string
import numpy as np
from scipy import sparse as sps
import matplotlib.pyplot as plt
import time

def get_processed_data(file_path):
	file = open(file_path, 'r')

	punc = string.punctuation
	table = str.maketrans(punc, ' ' * len(punc))
	
	printable = set(string.printable)
	data = []
	for line in file:
		review =  ''.join(s for s in line if s in printable).replace('<br />', ' ')
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

	class_labels = sps.lil_matrix((len(classes), len(labels)))
	label_count, i = [0] * len(classes), 0
	for label in labels:
		class_labels[label_map[label], i] = 1
		label_count[label_map[label]] += 1
		i += 1

	return label_map, inv_label_map, class_labels, label_count

def get_features(data, dictionary):
	m, V = len(data), len(dictionary)
	features = sps.lil_matrix((m, V))
	
	i = 0
	for line in data:
		for word in line:
			if word in dictionary:
				features[i, dictionary[word]] = 1
		i += 1

	return features

def naive_bayes(features, class_labels, label_count, c = 1):
	label_sum = np.array(label_count)
	m = sum(label_sum)
	
	phi_j = ((class_labels @ features).toarray() + c) / (label_sum[:, None] + 2 * c)
	return phi_j, label_sum / m

def predict(features, phi_j, phi_y):
	f_t = features.transpose()
	log_phi_j = np.log(phi_j)
	inv_log_phi_j = np.log(1 - phi_j)
	C, m = phi_j.shape[0], f_t.shape[1]
	
	temp = np.empty((C, m))
	temp[:,:] = np.sum(inv_log_phi_j, axis=1).reshape((phi_j.shape[0], 1))

	prob = (log_phi_j - inv_log_phi_j) @ f_t + temp + np.log(phi_y).reshape((phi_y.shape[0], 1))

	pred = prob.argmax(axis=0)
	return pred

def get_accuracy(truth_labels, pred, label_map):
	correct, count = 0, 0
	for i in range(0, len(truth_labels)):
		if label_map[truth_labels[i]] == pred[i]:
			correct += 1
		count += 1

	return correct / count

def draw_confusion_matrix(truth_labels, pred, label_map):
	conf_arr = np.zeros((len(label_map), len(label_map)))

	for i in range (0, len(truth_labels)):
		conf_arr[label_map[truth_labels[i]]][pred[i]] += 1

	plt.ion()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(conf_arr / np.linalg.norm(conf_arr)), cmap=plt.cm.RdPu, interpolation='nearest')
	width, height = conf_arr.shape

	for x in range(width):
		for y in range(height):
			ax.annotate(str(conf_arr[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

	print(np.trace(conf_arr) / np.sum(conf_arr))

	plt.xticks(range(width), list(label_map.keys()))
	plt.yticks(range(height), list(label_map.keys()))
	plt.title('Confusion Matrix')

if __name__ == '__main__':
	start = time.time()
	data = get_processed_data('imdb/train_text')
	dictionary = get_dictionary(data)

	features = get_features(data, dictionary)

	label_map, inv_label_map, class_labels, label_count = get_class_labels('imdb/imdb_train_labels.txt')

	phi_j, phi_y = naive_bayes(features, class_labels, label_count)

	test_features = get_features(get_processed_data('imdb/test_text'), dictionary)
	test_labels = get_labels('imdb/imdb_test_labels.txt')

	pred = predict(test_features, phi_j, phi_y)

	acc = get_accuracy(test_labels, pred, label_map)

	print(acc)

	end = time.time()
	print(end - start)

	draw_confusion_matrix(test_labels, pred, label_map)
	input('Exit: ')