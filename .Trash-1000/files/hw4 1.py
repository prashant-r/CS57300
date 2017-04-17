#!/usr/bin/env python
# Python 3.

from collections import Counter
from collections import defaultdict

import csv
import sys
import string
import math
import random
from scipy import stats
from pylab import *

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

def read_from_csv(csv_file_name):
    list_of_dict = []
    dataset = []
    with open(csv_file_name, 'rU') as f:
        reader = csv.DictReader(f, dialect=csv.excel_tab, fieldnames=['reviewID', 'classLabel', 'reviewText'])
        for row in reader:
            if(row['reviewText'] is not None and row['classLabel'] is not None):
                dataset.append( (int(row['classLabel']), process_str(str(row['reviewText']))) )
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

def generate_vector(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i

    v = []
    for item in dataset:
        vector = [0] * len(common_words)
          # Intercept term.
        vector.append(1)
        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        v.append((vector, item[0]))

    return np.array(v)

def get_random_subsets(X, y, n_subsets, replacements=True):
    n_samples = np.shape(X)[0]
    # Concatenate x and y and do a random shuffle
    X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(X_y)
    subsets = []

    # Uses 50% of training samples without replacements
    subsample_size = n_samples // 2
    if replacements:
        subsample_size = n_samples      # 100% with replacements

    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_samples),
            size=np.shape(
                range(subsample_size)),
            replace=replacements)
        X = X_y[idx][:, :-1]
        y = X_y[idx][:, -1]
        subsets.append([X, y])
    return subsets

class LogisticRegression:
    def __init__(self):
        self.w = None

    def reset(self):
        self.__init__()

    def logistic(self, vector):
        return 1.0 / (1 + np.exp(-vector))

    def fit(self, features, labels):
        n, d = features.shape
        w = np.zeros(d).T
        w_prev = np.ones(d).T
        i = 0

        # Parameters for logistic regression.
        # max_iter - Maximum iterations.
        # tol - Tolerance value.
        # alpha - Step size.
        # l - L2 regularization penalty.
        max_iter = 100; tol = 1e-6; alpha = 0.01; l = 0.01

        while True:
            if (np.linalg.norm(w - w_prev) < tol) or (i >= max_iter):
                break

            h = self.logistic(features @ w)
            loss_grad = (features.T @ (h - labels)) + (l * w)
            w_prev = w
            w = w - (alpha * loss_grad)
            i += 1
        self.w = w;
        return w

    def logistic_pred(self, features):
        threshold = 0.5
        pred = np.where(self.logistic(features @ self.w) >= threshold, 1, 0)
        return pred

    def score(self, features, labels):
        test_pred = self.logistic_pred(features);
        return self.calc_error(test_pred, labels);

    def calc_error(self, pred, labels):
        error = sum(np.where(pred != labels, 1, 0))
        return (error / labels.size)

class SVM:

    def __init__(self):
        self.w = None

    def reset(self):
        self.__init__()


    def fit(self, features, labels):
        # test sub-gradient SVM
        total = features.shape[1]
        lam = 1.; D = total
        x = features; y = (labels-0.5)*2
        w = np.zeros(D); wpr = np.ones(D)
        eta = 0.5; lam = 0.01; i = 0; MAXI = 100; tol = 1e-6
        while True:
            if np.linalg.norm(w-wpr) < tol or i > MAXI:
                break
            f = w @ x.T
            pL = np.where(np.multiply(y,f) < 1, -x.T @ np.diag(y), 0)
            pL = np.mean(pL,axis=1) + lam*w
            wpr = w
            w = w - eta*pL
            i += 1
        self.w = w;
        return w

    def score(self, features, labels):
        test_pred = self.svm_pred(features);
        return self.calc_error(test_pred, labels);

    def svm_pred(self, features):
        return np.where((features @ self.w) >= 0, 1, 0)

    def calc_error(self, pred, labels):
        error = sum(np.where(pred != labels, 1, 0))
        return (error / labels.size)


class DTree():
    class DecisionNode():
        def __init__(self, feature_i=0, threshold=None,
                 value=None, true_branch=None, false_branch=None):
            self.feature_i = feature_i          # Index for the feature that is tested
            self.threshold = threshold          # Threshold value for feature
            self.value = value                  # Value if the node is a leaf in the tree
            self.true_branch = true_branch      # 'Left' subtree
            self.false_branch = false_branch    # 'Right' subtree

    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=20, max_features = None, loss=None):
        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        # If y is nominal
        self.one_dim = None
        # If Gradient Boost
        self.loss = loss
        # If rf_tree
        self.max_features = max_features
        # Build tree

    def fit(self, X, y):
        # Build tree
        self.one_dim = len(np.shape(y)) == 1
        self.loss=None
        self.root = self._build_tree(X, y, 1)

    def dt_pred(self, X):
        preds = list([ self.predict(X[:,i]) for i in range(X.shape[1]) ])
        return preds;

    thresh = 0;
    def predict(self, x, tree=None):
        if tree is None:
            tree = self.root
        
        # Choose the feature that we will test
        feature_value = x[tree.feature_i]
        global thresh;
        thresh = tree.threshold;
        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if feature_value > 0:
            branch = tree.true_branch

        if branch is None:
            return thresh;
        # Test subtree
        return self.predict(x, branch)

    def _build_tree(self, X, y, depth):
        num_features, num_samples = X.shape
        # Recursion-termination condition
        best_criteria = None 
        best_sets = None

        if np.size(np.unique(y)) == 1:
            return DecisionTree.DecisionNode(threshold=y[0])
        if depth <= self.max_depth and np.size(y) >= self.min_samples_split:
            # Recurse   
            if self.max_features is not None:
                perm = np.random.choice(range(num_features), size=self.max_features, replace=True)
                best_feature_i = perm[self._impurity_calculation_index(X[perm], y)]
            else:
                best_feature_i = self._impurity_calculation_index(X, y);
            largest_impurity = impurity = self._impurity_calculation(X,y,best_feature_i)
            i_neg = X[best_feature_i] == 0
            i_pos = X[best_feature_i] > 0
            best_criteria = {
                    "feature_i": best_feature_i}
            best_sets = {
                    "leftX": X[:, i_neg],
                    "lefty": y[i_neg],
                    "rightX": X[:, i_pos],
                    "righty": y[i_pos]
            } 
            false_branch = self._build_tree(best_sets["leftX"],best_sets["lefty"] , depth+1)
            true_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], depth+1)
            return DecisionTree.DecisionNode(feature_i=best_criteria["feature_i"], threshold= self._majority_vote(y), true_branch=true_branch, false_branch=false_branch)
        return None
    
    def score(self, features, labels):
        test_pred = self.dt_pred(features);
        return self.calc_error(test_pred, labels);

    def calc_error(self, pred, labels):
        error = sum(np.where(pred != labels, 1, 0))
        return (error / labels.size)

class DecisionTree(DTree):

    def set_max_depth(self, depth):
        self.max_depth = depth

    def set_number_of_trees(self, numTrees):
        None

    def _calculate_gini_gain(self, X, y, k):
        X_num_pos = np.sum(X[k]) + 1e-7
        X_num_neg = X.shape[1] - X_num_pos
        X_pos_y_pos = np.dot(X[k], y)
        X_pos_y_neg = X_num_pos - X_pos_y_pos
        X_neg_y_pos = np.dot( abs(X[k]-1), y )
        X_neg_y_neg = X_num_neg - X_neg_y_pos
        gini_pos = 1 - (1.*X_pos_y_pos / X_num_pos)**2 - (1.*X_pos_y_neg / X_num_pos)**2
        gini_neg = 1 - (1.*X_neg_y_pos / X_num_neg)**2 - (1.*X_neg_y_neg / X_num_neg)**2
        return 0 - ( gini_pos*X_num_pos + gini_neg*X_num_neg ) / X.shape[1]

    def _calculate_gini_gain_index(self, X, y):
        X_num_pos = np.sum(X, axis=1) + 1e-7
        X_num_neg = X.shape[1] - X_num_pos
        X_pos_y_pos = np.dot(X, y)
        X_pos_y_neg = X_num_pos - X_pos_y_pos
        X_neg_y_pos = np.dot( abs(X-1), y )
        X_neg_y_neg = X_num_neg - X_neg_y_pos
        gini_pos = 1 - (1.*X_pos_y_pos / X_num_pos)**2 - (1.*X_pos_y_neg / X_num_pos)**2
        gini_neg = 1 - (1.*X_neg_y_pos / X_num_neg)**2 - (1.*X_neg_y_neg / X_num_neg)**2
        return np.argmax( 0 - ( gini_pos*X_num_pos + gini_neg*X_num_neg ) / X.shape[1] )

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_gini_gain
        self._impurity_calculation_index = self._calculate_gini_gain_index
        self._leaf_value_calculation = self._majority_vote
        super(DecisionTree, self).fit(X, y)

    def reset(self):
        self.__init__()


class BaggedDecisionTree:

    def __init__(self, n_trees=50, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth if max_depth in set({int, float, np.int64, np.float64}) else np.inf
        self.fraction = 1
        self.trees = [0]*n_trees
        self.trained = False

    def reset(self):
        self.trained = False
        self.trees = [0]*self.n_trees

    def set_number_of_trees(self, numTrees):
        self.n_trees = numTrees;

    def set_max_depth(self, depth):
        self.max_depth = depth
    # train() trains the Bagged Forest with input numpy arrays X and y
    def fit(self, X, y):
        #check dimensions
        if not len(X.T) == len(y):
            raise IndexError("The number of samples in X and y do not match")

        if type(y) is not np.ndarray:
            y = self.__numpify(y)
            if not y:
                raise TypeError("input label vector y is not a valid numeric array")

        #check if trained
        if self.trained:
            self.reset()

        indices = np.arange(len(y))
        #determine the size of the bootstrap sample
        strapsize = np.int(len(y)*self.fraction)
        for t in range(self.n_trees):
            #creat a new classification tree
            tree = DecisionTree(max_depth=self.max_depth)
            #bootstrap a sample
            bootstrap = np.random.choice(indices, strapsize)
            Xstrap = X[:,bootstrap]
            ystrap = y[bootstrap]
            #train the t-th tree with the strapped sample
            tree.fit(Xstrap,ystrap)
            self.trees[t] = tree
        self.trained = True

    # predict() uses a trained Bagged Forest to predict labels for a supplied numpy array X
    # returns a one-dimensional vector of predictions, which is selected by a plurality
    # vote from all the bagged trees
    def score(self, features, labels):
        if not self.trained:
            raise RuntimeError("The bagged forest classifier hasn't been trained yet")
        #get predictions from each tree
        #combine predictions into one matrix
        #get the mode of predictions for each sample
        prediction_matrix = np.zeros((len(features.T), self.n_trees))
        for t in range(self.n_trees):
            pred = self.trees[t].dt_pred(features)
            prediction_matrix[:,t] = pred
        final_vote = stats.mode(prediction_matrix, axis=1)[0]
        pred = final_vote.flatten()
        error = sum(np.where(pred != labels, 1, 0))
        return (error / labels.size);
            
class RandomForest():

    def __init__(self, n_estimators=50, max_features=None, min_samples_split=10,
                 min_gain=1e-7, max_depth=10, debug=False):
        self.n_estimators = n_estimators    # Number of trees
        self.max_features = max_features    # Maxmimum number of features per tree
        self.feature_indices = []           # The indices of the features used for each tree
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain            # Minimum information gain req. to continue
        self.max_depth = max_depth          # Maximum depth for tree
        self.debug = debug

        # Initialize decision trees
        self.trees = [0]*n_estimators

    def fit(self, X, y):
        num_features = np.shape(X)[0]
        num_samples  = np.shape(X)[1]
        # If max_features have not been defined => select it as
        # sqrt(n_features)
        if not self.max_features:
            self.max_features = int(math.sqrt(num_features))
        for i in range(self.n_estimators):
            dt = DecisionTree(max_depth=10, max_features = self.max_features)
            # Sample with replacement
            idx = np.random.randint(0, num_samples, num_samples)
            dt.fit(X[:,idx], y[idx])
            self.trees[i] = dt;
        self.trained = True

    def set_number_of_trees(self, numTrees):
        self.n_estimators = numTrees;

    def set_max_depth(self, depth):
        self.max_depth = depth

    def reset(self):
        self.__init__()

    def score(self, features, labels):
        if not self.trained:
            raise RuntimeError("The bagged forest classifier hasn't been trained yet")
        #get predictions from each tree
        #combine predictions into one matrix
        #get the mode of predictions for each sample
        prediction_matrix = np.zeros((len(features.T), self.n_estimators))
        for t in range(self.n_estimators):
            pred = self.trees[t].dt_pred(features)
            prediction_matrix[:,t] = pred
        final_vote = stats.mode(prediction_matrix, axis=1)[0]
        pred = final_vote.flatten()
        error = sum(np.where(pred != labels, 1, 0))
        return (error / labels.size);



def kfoldCV(classifier, features, k, Assignment4 = False, seed = None):
    partitions = partition(features, k, seed)
    errors = list()
        
    # Run the algorithm k times, record error each time
    for i in range(k):
        trainingSet = list()
        for j in range(k):
            if j != i:
                trainingSet.append(partitions[j])

        # flatten training set
        trainingSet = [item for entry in trainingSet for item in entry]
        testSet = partitions[i]
        
        # Train and classify model
        trainedClassifier = train(classifier, trainingSet, Assignment4)
        error = classify(classifier, testSet, Assignment4)
        errors.append(error)
        
    # Compute statistics
    mean = sum(errors)/k
    variance = sum([(error - mean)**2 for error in errors])/(k)
    standardDeviation = variance**.5
    confidenceInterval = (mean - 1.96*standardDeviation, mean + 1.96*standardDeviation)
 
    #print("\t\tMean = {0:.2f} \n\t\tVariance = {1:.4f} \n\t\tStandard Devation = {2:.3f} \n\t\t95% Confidence interval: [{3:.2f}, {4:.2f}]"\
    #        .format(mean, variance, standardDeviation, confidenceInterval[0], confidenceInterval[1]))
    l = [];
    l.append(errors);
    l.append(mean);
    l.append(standardDeviation);
    l.append(confidenceInterval);
    l.append(k)
    return l;

# Divides data set into k partitions
def partition(dataSet, k, seed=None):
    size = math.ceil(len(dataSet)/float(k))
    partitions = [[] for i in range(k)]
    j = 0
    a= 0;
    for entry in dataSet:
        a = a+1;
        x = assign(partitions, k, size, seed) 
        partitions[x].append(entry)

    return partitions


# Assigns each entry to a non-full partition
def assign(partitions, k, size, seed=None):
    if seed is not None:
        np.random.Random(seed)
    x = np.random.randint(0,k-1)
    while(len(partitions[x%k]) >= size):
        x = np.random.randint(0,100)
    return x%k

def data_random_split(data, percentage):
    np.random.shuffle(data)
    train = data[:int(len(data)*percentage)]
    test = data[len(train):]
    return (train, test);

def train(classifier, trainingSet, Assignment4 = False):
    X = [entry[0] for entry in trainingSet]
    y = [entry[1] for entry in trainingSet]
    if Assignment4 :
        return classifier.fit(np.array(X).T,np.array(y))
    else:
        return classifier.fit(np.array(X),np.array(y))

def plot(x, y, yerr, y2, y2err, y3, y3err, y4, y4err, xlabel, ylabel, figDesc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.suptitle(figDesc, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([x[0], 5000])
    ax.plot(x, y, 'r-', label='Decision Tree')
    ax.plot(x, y2, 'g-', label='Bagged Decision Tree')
    ax.plot(x, y3, 'y-', label='Random Forest')
    ax.plot(x, y4, 'b-', label='SVM')
    ax.errorbar(x, y, yerr=yerr, fmt='ro')
    ax.errorbar(x, y2, yerr=y2err, fmt='go')
    ax.errorbar(x, y3, yerr=y3err, fmt='yo')
    ax.errorbar(x, y4, yerr=y4err, fmt='bo')
    ax.autoscale_view(True,True,True)
    ax.legend(loc='upper right')
    plt.savefig(figDesc)

# Runs a classifier on the input, outputs the success rate
def classify(classifier, dataSet, Assignment4 = False):
    X = [entry[0] for entry in dataSet]
    y = [entry[1] for entry in dataSet]
    if Assignment4 :
        return classifier.score(np.array(X).T,np.array(y))
    else:
        return classifier.score(np.array(X),np.array(y))
if __name__ == '__main__':
    analyseOne = 0;
    analyseAll = 1;
    if len(sys.argv) == 4:
        train_data_file = sys.argv[1]
        test_data_file = sys.argv[2]
        model_idx = int(sys.argv[3])
        train_data = read_dataset(train_data_file)
        test_data = read_dataset(test_data_file)
        common_words = get_most_commons(train_data, skip=100, total=1000)
        train_f, train_l = generate_vectors(train_data, common_words)
        test_f, test_l = generate_vectors(test_data, common_words)

        split_percentages = [0.025, 0.05, 0.125, 0.25]
        D = read_from_csv('yelp_data.csv');
        classifier = None;

        if model_idx == 1:
            dt = DecisionTree();
            classifier = dt;
            train_f = train_f.T;
            test_f = test_f.T;
            w = dt.fit(train_f, train_l)
            print('ZERO-ONE-LOSS-DT', dt.score(test_f, test_l))

        elif model_idx == 2:
            bt = BaggedDecisionTree();
            classifier = bt;
            train_f = train_f.T;
            test_f = test_f.T;
            w = bt.fit(train_f, train_l)
            print('ZERO-ONE-LOSS-BT', bt.score(test_f, test_l))
        
        elif model_idx == 3:
            rf = RandomForest();
            classifier = rf;
            train_f = train_f.T;
            test_f = test_f.T;
            w = rf.fit(train_f, train_l)
            print('ZERO-ONE-LOSS-RF', rf.score(test_f, test_l))

        elif model_idx == 4:
            lr = LogisticRegression();
            classifier = lr;
            w = lr.fit(train_f, train_l)
            test_pred = lr.logistic_pred(test_f)
            print('ZERO-ONE-LOSS-LR', lr.calc_error(test_pred, test_l))

        elif model_idx == 5:
            svm = SVM();
            classifier = svm;
            w = svm.fit(train_f, train_l)
            test_pred = svm.svm_pred(test_f)
            print('ZERO-ONE-LOSS-SVM', svm.calc_error(test_pred, test_l))

        else:
            print('Illegal modelIdx')
            sys.exit(-1)

        if(analyseOne == 1):
            classifier.reset()
            for p in split_percentages:
                D_per = data_random_split(D, p)[0];
                common_words = get_most_commons(D_per, skip=100, total=1000)
                features = generate_vector(D_per, common_words)
                if model_idx <= 3 :
                    kfoldCV(classifier, np.array(features), 10, True);
                else:
                    kfoldCV(classifier, np.array(features), 10, False);
        if(analyseAll):
            analysisList = [];
            analysisList.append("Analysis1");
            analysisList.append("Analysis2");
            analysisList.append("Analysis3");
            analysisList.append("Analysis4");
            print('Running Analysis 1')
            #Analysis 1
            analysisLabel = analysisList[0];
            plotsx = [];
            plotsy = [];
            sdevsy = [];
            for i in range(4):
                plotsy.append([])
                sdevsy.append([])
            model = 0;
            kfold = [];
            for modelIdx in range(1,5):
                model = model + 1;
                split_percentages = [0.025, 0.05, 0.125, 0.25]
                D = read_from_csv('yelp_data.csv');
                classifier = None;

                if modelIdx == 1:
                    dt = DecisionTree();
                    classifier = dt;
                
                elif modelIdx == 2:
                    bt = BaggedDecisionTree();
                    classifier = bt;
            
                elif modelIdx == 3:
                    rf = RandomForest();
                    classifier = rf;

                elif modelIdx == 4:
                    svm = SVM();
                    classifier = svm;ctor(D_per, common_words)
            
                    if modelIdx <= 3 :
                        classifier.reset()
                        kfold.append(kfoldCV(classifier, np.array(features), 10, True))
                    else:
                        kfold.append(kfoldCV(classifier, np.array(features), 10, False));
            counter = 0;
            for i in range(4):
                for s in split_percentages:
                    plotsy[i].append(kfold[counter][1])
                    sdevsy[i].append(kfold[counter][2]);
                    counter = counter + 1;
            for s in split_percentages:
                plotsx.append(s);
            plot(plotsx, plotsy[0], sdevsy[0], plotsy[1], sdevsy[1], plotsy[2], sdevsy[2], plotsy[3], sdevsy[3], "Sample Size' Fraction", "Mean Zero-One Loss", analysisLabel)
            
            print('Running Analysis 2')
            ## Analysis 2
            analysisLabel = analysisList[1];
            plotsx = [];
            plotsy = [];
            sdevsy = [];
            features_sizes = [200, 500, 1000, 1500];
            for i in range(4):
                plotsy.append([])
                sdevsy.append([])
            model = 0;
            kfold = [];
            for modelIdx in range(1,5):
                model = model + 1;
                split_percentages = 0.25
                classifier = None;

                if modelIdx == 1:
                    dt = DecisionTree();
                    classifier = dt;
                
                elif modelIdx == 2:
                    bt = BaggedDecisionTree();
                    classifier = bt;
            
                elif modelIdx == 3:
                    rf = RandomForest();
                    classifier = rf;

                elif modelIdx == 4:
                    svm = SVM();
                    classifier = svm;

                for p in features_sizes:
                    D_per = data_random_split(D, split_percentages)[0];
                    common_words = get_most_commons(D_per, skip=100, total=p)
                    features = generate_vector(D_per, common_words)
            
                    if modelIdx <= 3 :
                        classifier.reset()
                        kfold.append(kfoldCV(classifier, np.array(features), 10, True))
                    else:
                        kfold.append(kfoldCV(classifier, np.array(features), 10, False));
            counter = 0;
            for i in range(4):
                for p in features_sizes:
                    plotsy[i].append(kfold[counter][1])
                    sdevsy[i].append(kfold[counter][2]);
                    counter = counter + 1;
            for p in features_sizes:
                plotsx.append(p);
            plot(plotsx, plotsy[0], sdevsy[0], plotsy[1], sdevsy[1], plotsy[2], sdevsy[2], plotsy[3], sdevsy[3], "Feature Size", "Mean Zero-One Loss", analysisLabel)
            
            print('Running Analysis 3')
            #Analysis 3
            analysisLabel = analysisList[2];
            depth_limits =  [5, 10, 15, 20]
            plotsx = [];
            plotsy = [];
            sdevsy = [];
            for i in range(4):
                plotsy.append([])
                sdevsy.append([])
            model = 0;
            kfold = [];
            for modelIdx in range(1,5):

                model = model + 1;
                split_percentages = 0.25
                classifier = None;

                if modelIdx == 1:
                    dt = DecisionTree();
                    classifier = dt;
                
                elif modelIdx == 2:
                    bt = BaggedDecisionTree();
                    classifier = bt;
            
                elif modelIdx == 3:
                    rf = RandomForest();
                    classifier = rf;

                elif modelIdx == 4:
                    svm = SVM();
                    classifier = svm;
                
                    D_per = data_random_split(D, split_percentages)[0];
                    common_words = get_most_commons(D_per, skip=100, total=1000)
                    features = generate_vector(D_per, common_words)
                    for p in depth_limits:
                        if modelIdx <= 3 :
                            classifier.reset()
                            classifier.set_max_depth(p);
                            kfold.append(kfoldCV(classifier, np.array(features), 10, True, p))
                    else:
                        kfold.append(kfoldCV(classifier, np.array(features), 10, False));
            counter = 0;
            for i in range(4):
                for p in depth_limits:
                    plotsy[i].append(kfold[counter][1])
                    sdevsy[i].append(kfold[counter][2]);
                    counter = counter + 1;
            for p in depth_limits:
                plotsx.append(p);
            plot(plotsx, plotsy[0], sdevsy[0], plotsy[1], sdevsy[1], plotsy[2], sdevsy[2], plotsy[3], sdevsy[3], "Depth Limits", "Mean Zero-One Loss", analysisLabel)
            

            print('Running Analysis 4')
            #Analysis 4
            analysisLabel = analysisList[3];
            number_of_trees =  [10, 25, 50, 100];
            plotsx = [];
            plotsy = [];
            sdevsy = [];
            for i in range(4):
                plotsy.append([])
                sdevsy.append([])
            model = 0;
            kfold = [];
            for modelIdx in range(1,5):

                model = model + 1;
                split_percentages = 0.25
                classifier = None;

                if modelIdx == 1:
                    dt = DecisionTree();
                    classifier = dt;
                
                elif modelIdx == 2:
                    bt = BaggedDecisionTree();
                    classifier = bt;
            
                elif modelIdx == 3:
                    rf = RandomForest();
                    classifier = rf;

                elif modelIdx == 4:
                    svm = SVM();
                    classifier = svm;
                
                    D_per = data_random_split(D, split_percentages)[0];
                    common_words = get_most_commons(D_per, skip=100, total=1000)
                    features = generate_vector(D_per, common_words)
                    for p in number_of_trees:
                        if modelIdx <= 3 :
                            classifier.reset()
                            classifier.set_max_depth(10);
                            classifier.set_number_of_trees(p);
                            kfold.append(kfoldCV(classifier, np.array(features), 10, True))
                    else:
                        kfold.append(kfoldCV(classifier, np.array(features), 10, False));
            counter = 0;
            for i in range(4):
                for p in number_of_trees:
                    plotsy[i].append(kfold[counter][1])
                    sdevsy[i].append(kfold[counter][2]);
                    counter = counter + 1;
            for p in number_of_trees:
                plotsx.append(p);
            plot(plotsx, plotsy[0], sdevsy[0], plotsy[1], sdevsy[1], plotsy[2], sdevsy[2], plotsy[3], sdevsy[3], "Number of Trees ", "Mean Zero-One Loss", analysisLabel)
            
    else:
        print('usage: python hw4.py train.csv test.csv modelIdx')
        sys.exit(-1)
