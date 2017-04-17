import sys
import string
import copy
from collections import Counter
from operator import itemgetter

def process_str(s):
    return s.translate(string.punctuation).lower().split()

# dataset format:
# list of (class_label, set of words)
def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
            dataset.append( (int(class_label), set(words)) )
    return dataset

def get_most_commons(dataset, skip=100, total=100):
    my_list = []
    for item in dataset:
        my_list += list(item[1])

    counter = Counter(my_list)

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words

# the length of the common words will be the
# length of the vectors
def generate_vectors(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i

    vectors = []
    for item in dataset:
        vector = [0] * len(common_words)
        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append( (item[0], vector) )

    return vectors

def naive_bayes_learn(train_vectors):
    likelihoods = []
    priors = [0.0, 0.0]
    train_vector_length = len(train_vectors[0][1])

    for vector in train_vectors:
        priors[ vector[0] ] += 1

    summed = sum(priors)
    priors[0] = priors[0] / summed
    priors[1] = priors[1] / summed

    for i in range(train_vector_length):
        likelihood = [[1.0, 1.0], [1.0, 1.0]] # class, value
        for vector in train_vectors:
            likelihood[vector[0]][vector[1][i]] += 1

        summed1 = sum(likelihood[0])
        summed2 = sum(likelihood[1])

        likelihood[0][0] = likelihood[0][0] / summed1
        likelihood[0][1] = likelihood[0][1] / summed1
        likelihood[1][0] = likelihood[1][0] / summed2
        likelihood[1][1] = likelihood[1][1] / summed2

        likelihoods.append(likelihood)

    return priors, likelihoods

def naive_bayes_classify(priors, likelihoods, vector):
    posterior = copy.deepcopy(priors)
    for index in range(len(vector)):
        for i in range(2): #class 0 or 1
            posterior[i] = posterior[i] * likelihoods[index][i][vector[index]]
    return posterior.index(max(posterior))

def naive_bayes_train_test(train_vectors, test_vectors):
    mistake = 0.0

    priors, likelihoods = naive_bayes_learn(train_vectors)

    for vector in test_vectors:
        classified = naive_bayes_classify(priors, likelihoods, vector[1])
        if classified != vector[0]:
            mistake += 1

    return mistake / len(test_vectors)

def main():
    if len(sys.argv) == 3:
        train_data_file = sys.argv[1]
        test_data_file = sys.argv[2]

        train_data = read_dataset(train_data_file)
        test_data = read_dataset(test_data_file)

        top_ten = get_most_commons(train_data, skip=100, total=10)
        for i in range(len(top_ten)):
             print('WORD' + str(i+1) +' '+ top_ten[i])

        common_words = get_most_commons(train_data, skip=100, total=500)

        train_vectors = generate_vectors(train_data, common_words)
        test_vectors = generate_vectors(test_data, common_words)

        zero_one_loss = naive_bayes_train_test(train_vectors, test_vectors)

        print('ZERO-ONE-LOSS ' + str(zero_one_loss))
    else:
        print('usage: python nbc.py train.csv test.csv')
        print('exiting...')

main()
