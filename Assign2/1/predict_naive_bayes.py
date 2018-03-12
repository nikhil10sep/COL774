from naive_bayes import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('test_text', type=str, help='Testing text file')
	parser.add_argument('model', type=str, help='Naive Bayes model')
	parser.add_argument('predictions', type=str, help='Output file for predictions')
	parser.add_argument('-tl', '--test_labels', type=str, help='Truth labels for test data')
	parser.add_argument('-c', '--confusion_matrix', action='store_true')
	args = parser.parse_args()

	with open(args.model, 'rb') as model_in:
		dictionary = pickle.load(model_in)
		label_map = pickle.load(model_in)
		inv_label_map = pickle.load(model_in)
		m = pickle.load(model_in)

		test_features = None
		if m[0].shape[1] == len(dictionary):
			test_features = get_features(get_processed_data(args.test_text), dictionary)
		elif m[0].shape[1] == len(dictionary) + 1:
			test_features = get_features(get_processed_data(args.test_text), dictionary, l=True)
		
		pred = predict(test_features, m, inv_label_map)

		with open(args.predictions, 'w') as pred_out:
			for i in range(len(pred)):
				pred_out.write(str(pred[i]) + '\n') 

		if args.test_labels:
			test_labels = get_labels(args.test_labels)
			print('Test Accuracy:', get_accuracy(test_labels, pred))
			if args.confusion_matrix:
				draw_confusion_matrix(test_labels, pred, label_map)
				input('Press Enter to exit:')