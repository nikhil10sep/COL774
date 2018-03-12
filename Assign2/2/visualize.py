from svm_train import *

def save_images(features_file, pred_file_name, truth_labels_file_name, pl, tl):
	pred_file = open(pred_file_name, 'r')
	pred = []
	for p in pred_file.readlines():
		pred.append(int(p))

	truth_labels_file = open(truth_labels_file_name, 'r')
	truth_labels = []
	for t in truth_labels_file.readlines():
		truth_labels.append(int(t))

	i = 0
	with open(features_file, 'r') as csv_file:
		for pixels in csv.reader(csv_file):
			if pred[i] == pl and truth_labels[i] == tl:
				pixels = np.array(pixels, dtype='uint8')
				pixels = pixels.reshape((28, 28))
				plt.imshow(pixels, cmap='gray')
				plt.savefig('images/' + str(i) + '_' + str(pl) + '_' + str(tl), format='png')
				plt.close()
			i += 1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('test_text', type=str, help='Testing text file')
	parser.add_argument('pred_file', type=str, help='Predictions')
	parser.add_argument('truth_labels', type=str, help='Truth labels for test data')
	parser.add_argument('pl', type=int, help='Prediction Label')
	parser.add_argument('tl', type=int, help='Truth Label')
	args = parser.parse_args()

	save_images(args.test_text, args.pred_file, args.truth_labels, args.pl, args.tl)


