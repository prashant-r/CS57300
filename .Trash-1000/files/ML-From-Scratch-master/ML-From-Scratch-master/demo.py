from __future__ import print_function
import sys, os
from sklearn import datasets
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/utils")
from data_manipulation import train_test_split, normalize
from data_operation import accuracy_score
from kernels import *
# Import ML models
sys.path.insert(0, dir_path + "/supervised_learning")
from adaboost import Adaboost
from decision_tree import ClassificationTree
from random_forest import RandomForest
import sys

print ("+-------------------------------------------+")
print ("|                                           |")
print ("|       Machine Learning From Scratch       |")
print ("|                                           |")
print ("+-------------------------------------------+")

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
# ...........
#  LOAD DATA
# ...........


if len(sys.argv) != 4:
	print('usage: python hw4.py train.csv test.csv modelIdx')

else:
	train_data_file = sys.argv[1]
	test_data_file = sys.argv[2]
	model_idx = int(sys.argv[3])
	train_data = read_dataset(train_data_file)
	test_data = read_dataset(test_data_file)
	common_words = get_most_commons(train_data, skip=100, total=1000)
	X_train, y_train = generate_vectors(train_data, common_words)
	X_test, y_test = generate_vectors(test_data, common_words)
	# ..........................
	#  TRAIN / TEST SPLIT
	# ..........................
	# Rescaled labels {-1, 1}
	rescaled_y_train = 2*y_train - np.ones(np.shape(y_train))
	rescaled_y_test = 2*y_test - np.ones(np.shape(y_test))
	# .......
	#  SETUP
	# .......
	adaboost = Adaboost(n_clf = 8)
	decision_tree = ClassificationTree()
	random_forest = RandomForest(n_estimators=50)
	# ........
	#  TRAIN
	# ........
	print ("Training:")
	print ("\tAdaboost")
	adaboost.fit(X_train, rescaled_y_train)
	print ("\tDecision Tree")
	decision_tree.fit(X_train, y_train)
	print ("\tRandom Forest")
	random_forest.fit(X_train, y_train)
	# .........
	#  PREDICT
	# .........
	y_pred = {}
	y_pred["Adaboost"] = adaboost.predict(X_test)
	y_pred["Decision Tree"] = decision_tree.predict(X_test)
	y_pred["Random Forest"] = random_forest.predict(X_test)

	# ..........
	#  ACCURACY
	# ..........

	print ("Accuracy:")
	for clf in y_pred:

		if clf == "Adaboost":
			print ("\t%-23s: %.5f" %(clf, accuracy_score(rescaled_y_test, y_pred[clf])))
		else:
			print ("\t%-23s: %.5f" %(clf, accuracy_score(y_test, y_pred[clf])))


