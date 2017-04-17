import numpy as np
from ClassTree import ClassificationTree
from ClassTreeBagging import TreeBagger
from ClassForest import RandomForest
import sys
import string
import copy
from collections import Counter
from operator import itemgetter

# Create the classifier objects
tree = ClassificationTree()
bag = TreeBagger(n_trees=50)
forest = RandomForest(n_trees=50)

# Get datasets from scikit-learn
from sklearn.datasets import load_iris # iris classification


def process_str(s):
    rem_punc = str.maketrans('', '', string.punctuation)
    return s.translate(rem_punc).lower().split()

def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
            dataset.append( (int(class_label), words) )

    return dataset

def get_most_commons(dataset, skip=100, total=100):
    counter = Counter()
    for item in dataset:
        counter = counter + Counter(set(item[1]))

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words

def generate_vectors(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i

    vectors = []
    labels = []
    for item in dataset:
        vector = [0] * len(common_words)
        # Intercept term.
        vector.append(1)

        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append(vector)
        labels.append(item[0])

    return np.array(vectors), np.array(labels)

if len(sys.argv) == 4:
        train_data_file = sys.argv[1]
        test_data_file = sys.argv[2]
        model_idx = int(sys.argv[3])

        train_data = read_dataset(train_data_file)
        test_data = read_dataset(test_data_file)

        common_words = get_most_commons(train_data, skip=100, total=1000)

        X_train, y_train = generate_vectors(train_data, common_words)
        X_test, y_test = generate_vectors(test_data, common_words)
        tree.train(X_train, y_train);
        print("Accuracy of the simple tree on iris dataset is %f" % tree.evaluate(X_test, y_test))
		# Cross validation of a tree
		
else:
        print('usage: python demo.py train.csv test.csv modelIdx')
		
		