from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sys

#initializing stemmer
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

# function that takes an input file and performs stemming to generate the output file
def getStemmedDocument(inputFileName, outputFileName, is_stem):
	out = open(outputFileName, 'w')
	f = open(inputFileName)
	
	for doc in f:
		raw = doc.lower()
		raw = raw.replace("<br /><br />", " ")
		tokens = tokenizer.tokenize(raw)
		stopped_tokens = [token for token in tokens if token not in en_stop]
		if is_stem == 1:
			stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]
			documentWords = ' '.join(stemmed_tokens)
		else:
			documentWords = ' '.join(stopped_tokens)
		
		out.write(documentWords + '\n')
	out.close()

# creates the new stemmed documents with the suffix 'new' for both train and test files
old_file=sys.argv[1]
#with open(old_file, encoding='utf-8', errors='ignore') as f:
#	x = f.readlines()
new_file=sys.argv[2]
getStemmedDocument(old_file, new_file, int(sys.argv[3]))
