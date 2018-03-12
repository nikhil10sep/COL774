from svm_train import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('test_text', type=str, help='Testing text file')
	parser.add_argument('method', type=int, help='Method (0:Pegasos, 1:Linear SMO, 2:RBF SMO')
	parser.add_argument('model', type=str, help='SVM model')
	parser.add_argument('pred_file', type=str, help='Output file for predictions')
	parser.add_argument('-tl', '--test_labels', type=str, help='Truth labels for test data')
	parser.add_argument('-cf', '--confusion_matrix', action='store_true')
	args = parser.parse_args()

	X_test = get_X(args.test_text)
	y_test, pred, classes = None, None, None
	if args.method == 0:
		with open(args.model, 'rb') as model_in:
			train = pickle.load(model_in)
			classes = pickle.load(model_in)

			pred = pred_one_vs_one(X_test, train, classes)

		if args.test_labels:
			y_test = np.genfromtxt(args.test_labels, dtype=int)
			print('Test Accuracy:', accuracy(pred, y_test))
		
	elif args.method == 1 or args.method == 2:
		with open(args.model, 'rb') as model_in:
			model_file = pickle.load(model_in)
			classes = pickle.load(model_in)

			model = svm_load_model(args.model + '.model')

		if args.test_labels:
			y_test = np.genfromtxt(args.test_labels, dtype=int)
			pred, acc = predict_smo(X_test.tolist(), y_test.tolist(), model)
			print('Test Accuracy:', acc)

		else:
			pred, acc = predict_smo(X_test.tolist(), [0] * X_test.shape[0], model)

	else:
		print('Wrong method option: Method (0:Pegasos, 1:Linear SMO, 2:RBF SMO)')
		sys.exit(1)

	with open(args.pred_file, 'w') as pred_out:
		for i in range(len(pred)):
			pred_out.write(str(pred[i]) + '\n')
	
	if args.test_labels:
		if args.confusion_matrix:
			draw_confusion_matrix(y_test, pred, classes)
			input('Press Enter to exit:')