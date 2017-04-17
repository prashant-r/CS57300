import numpy as np
import warnings
from TreeNode import Node
import json


class ClassificationTree:

    def __init__(self, depth_limit=10, impurity="gini"):
        #depth limit of the tree
        self.depth_limit = depth_limit if type(depth_limit) in set({int, float, np.int64, np.float64}) else np.inf
        #an array that holds all the nodes created during training
        #each level is a separate list
        self.nodes = [[]]
        #whether the model has been trained
        self.trained = False
        #set the desired impurity measure
        self.impurity = impurity
        #dimensions of the supplied training data
        self.dimensions = 0

    #helper functions

    #__classshares() calculates the proportions of each class in the supplied label vector
    def __classshares(self, labels):
        classes, counts = np.unique(labels, return_counts=True)
        shares = ((counts*1.) / len(labels)).reshape((len(classes),1))
        return classes, shares

    #__bestguess() finds the most probable class in the supplied label vector
    def __bestguess(self, labels):
        classes, shares = self.__classshares(labels)
        max_index = np.argmax(shares)
        return classes[max_index]

    #__entropy() calculates the entropy of the input dataset
    #labels are the data labels
    def __entropy(self, labels):
        if len(labels) == 0:
            return 0.
        classes, props = self.__classshares(labels)
        entropy = -np.dot( props.T, np.log2(props+0.00001) )
        return entropy[0][0]

    #__gini() is an alternative impurity calculation to entropy
    def __gini(self, labels):
        if len(labels) == 0:
            return 0.
        classes, props = self.__classshares(labels)
        gini = 1 - np.dot(props.T, props)
        return gini[0][0]

    #__impurity() is a wrapper for impurity calculations (entropy/gini)
    def __impurity(self, labels):
        if self.impurity == "gini":
            return self.__gini(labels)
        elif self.impurity == "entropy":
            return self.__entropy(labels)
        else:
            return self.__gini(labels)

    #__bestsplit() finds the split that results into lowest entropy
    def __bestsplit(self, feature, labels):
        values = np.unique(feature)
        bestentropy = np.inf
        bestsplit = 0
        #check if the number of values isn't too large
        if len(values) > 10:
            minv = values.min()
            maxv = values.max()
            values = np.arange(minv, maxv, (maxv-minv)/10.)
        for v in values:
            leftmask = feature <= v
            rightmask = feature > v
            leftentropy = self.__impurity(labels[leftmask])
            rightentropy = self.__impurity(labels[rightmask])
            #weighted mean of impurity
            mean_entropy = (np.sum(leftmask)*leftentropy + np.sum(rightmask)*rightentropy) / len(labels)
            if mean_entropy < bestentropy:
                bestentropy = mean_entropy
                bestsplit = v
        return bestsplit, bestentropy

    #__algorithm() is the main function for training the decision tree classifier
    #it proceeds as follows:
    # 1. calculate entropy at root node
    # 2. if entropy at root node is zero, there is only one class, so create a terminal node
    #    and end the algorithm
    # 3. if entropy is positive, start searching through possible splits
    # 4. for each feature, determine the smallest entropy if the set is split along this feature
    # 5. pick the feature with smallest entropy, split the tree
    # 6. if the optimal split results into putting all samples down one branch, make the node terminal
    # 7. move down the two branches and repeat from 1.

    def __algorithm(self, S, labels, level=0, par_node=None, left=False, terminal_flag=False):
        #calculate initial entropy
        null_entropy = self.__impurity(labels)
        #check if everyone is in the same class
        if null_entropy <= 0. or level >= self.depth_limit or terminal_flag:
            #terminate the algorithm, everyone's been classified or maximum depth has been reached
            final_node = Node(parent=par_node,level=level,entropy=null_entropy)
            final_node.outcome[0] = self.__bestguess(labels)
            self.nodes[level].extend( [final_node] )
            return final_node
        else:
            #go over all the features in this dataset
            features = list(range(S.shape[1]))
            min_entropy = np.inf
            best_split = [0,0] #this will hold feature number and threshold value for the best split
            for f in features:
                #try all possible splits along this feature
                #return the best (lowest) entropy
                #if this entropy is smaller then current minimum, update
                Sfeat = S[:,f]
                split, entropy = self.__bestsplit(Sfeat, labels)
                if entropy < min_entropy:
                    min_entropy = entropy
                    best_split = [f, split]

            new_node = Node(feature=best_split[0], threshold=best_split[1], parent=par_node, level=level, entropy=min_entropy)
            self.nodes[level].extend( [new_node] )
            #split dataset
            #check if S is a vector
            if len(S.shape) == 1:
                #S is a one-feature vector
                S = S.reshape((len(S),1))

            leftMask = S[:,best_split[0]] <= best_split[1]
            rightMask = S[:,best_split[0]] > best_split[1]
            features.remove(best_split[0])

            leftLabels = labels[leftMask]
            rightLabels = labels[rightMask]

            # check if you shouldn't terminate here
            # when the split puts all samples into left or right branch

            if leftMask.all():
                new_node.make_terminal(self.__bestguess(leftLabels))
                return new_node
            if rightMask.all():
                new_node.make_terminal(self.__bestguess(rightLabels))
                return new_node

            if len(features) == 0:
                leftS = S[leftMask,:]
                rightS = S[rightMask,:]
                terminal_flag = True
            else:
                leftS = S[leftMask,:][:,features]
                rightS = S[rightMask,:][:,features]

            #check if you shouldn't terminate here

            if len(leftS) == 0 or leftS.shape[1] == 0:
                new_node.make_terminal(self.__bestguess(rightLabels))
                return new_node
            if len(rightS) == 0 or rightS.shape[1] == 0:
                new_node.make_terminal(self.__bestguess(leftLabels))
                return new_node

            #check if a level below you already exists
            try:
                self.nodes[level+1]
            except IndexError:
                self.nodes.append([])

            #recursively call self again on the two children nodes
            new_node.outcome[0] = self.__algorithm(leftS,leftLabels,level=level+1,par_node=new_node,terminal_flag=terminal_flag)
            new_node.outcome[1] = self.__algorithm(rightS,rightLabels,level=level+1,par_node=new_node,terminal_flag=terminal_flag)
            return new_node
        #print("Tree grown")

    #__classify() takes one sample x and classifies it into a label
    def __classify(self, x):
        node = self.nodes[0][0]
        while isinstance(node.outcome[0], Node):
            val = x[node.feature]
            x = np.delete(x, node.feature)
            node = node.decide(val)
        return node.outcome[0]

    #__untrain() removes old learned nodes when a new train() is called on a trained tree
    def __untrain(self):
        self.trained = False
        self.nodes = [[]]

    #__numpify() takes a regular python list and turns it into a numpy array
    def __numpify(self, array):
        numpied = np.array(array)
        if numpied.dtype in ['int64', 'float64']:
            return numpied
        else:
            return False

    #__node_count() returns the total number of nodes
    def __node_count(self):
        if not self.trained:
            return 0
        else:
            n = 0
            for level in self.nodes:
                n += len(level)
            return n

    # train() is the function the user calls to train the tree. It's mainly a wrapper for the __algorithm() function
    def train(self, X, y):
        #check dimensions
        if not len(X) == len(y):
            raise IndexError("The number of samples in X and y do not match")
        #check if X and y are numpy arrays
        if type(X) is not np.ndarray:
            X = self.__numpify(X)
            if not X:
                raise TypeError("input dataset X is not a valid numeric array")
        if type(y) is not np.ndarray:
            y = self.__numpify(y)
            if not y:
                raise TypeError("input label vector y is not a valid numeric array")
        if self.trained:
            self.__untrain()

        self.__algorithm(X, y)
        self.trained = True
        self.dimensions = X.shape[1]

    # once the tree has been trained, you can call the predict() function to generate predicted labels for the supplied dataset
    def predict(self, X):
        if not self.trained:
            raise RuntimeError("The decision tree classifier hasn't been trained yet")
        if not X.shape[1] == self.dimensions:
            raise IndexError("The supplied dataset has %d features, which do not match %d features from training" % (X.shape[1], self.dimensions))

        yhat = np.zeros(len(X))
        for i,x in enumerate(X):
            yhat[i] = self.__classify(x)

        return yhat

    # one the tree has been trained, the evaluate() function scores the prediction compared to supplied ground truth.
    # there are three scoring methods implemented:
    # F1 score is the default:
    #  its formula is (2*precision*recall)/(precision+recall)
    #  its preferable to simple accuracy when classes are not balanced
    # Accuracy is a simple accuracy measure (percentage of samples correctly classified)
    # Matthews correlation coefficient is an alternative to the F1 score for evaluating an algorithm
    #  when classes are not balanced

    def __f1(self, y, yhat):
        # check if this is a multi-class problem
        classes, _ = self.__classshares(y)
        if len(classes) <= 2:
            # binary F1
            accurate = y == yhat
            positive = np.sum(y == 1)
            hatpositive = np.sum(yhat == 1)
            tp = np.sum(yhat[accurate] == 1)
            recall = 1.*tp/positive if positive > 0 else 0.
            precision = 1.*tp/hatpositive if hatpositive > 0 else 0.
            f1 = (2.*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.
            return f1
        else:
            # multi-class problem
            # iterate over classes, weighing by true occurence
            f1 = 0
            for label in classes:
                # create binary vectors to recursively calculate f1
                y_binary = np.copy(y)
                y_binary[y == label] = 1
                y_binary[y != label] = 0

                yhat_binary = np.copy(yhat)
                yhat_binary[yhat == label] = 1
                yhat_binary[yhat != label] = 0

                f1 += np.sum(y == label) * self.__f1(y_binary, yhat_binary)
            return f1 / len(yhat)

    def evaluate(self, X, y, method='f1'):
        yhat = self.predict(X)
        accurate = y == yhat

        #F1 score
        if method == 'f1':
            return self.__f1(y, yhat)
        #simple accuracy measure
        elif method == 'acc':
            return (1.*np.sum(accurate))/len(yhat)
        #matthews correlation coefficient
        elif method == 'matthews':
            tp = np.sum(yhat[accurate] == 1)
            tn = np.sum(yhat[accurate] == 0)
            fp = np.sum(yhat[np.invert(accurate)] == 1)
            fn = np.sum(yhat[np.invert(accurate)] == 0)
            denominator = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn*fn))
            mat = 1.*((tp*tn)-(fp*fn)) / denominator if denominator > 0 else 0.
            return mat
        else:
            warnings.warn("Wrong evaluation method specified, defaulting to F1 score", RuntimeWarning)
            return self.evaluate(X,y)

    # the cross_val() function implements cross_validation in training
    # it takes the input data X,y and based on the supplied paramters (train-test split and the number of folds)
    # it trains the tree fold-times using a (1-split) fraction of the original dataset, using the split share for scoring
    def cross_val(self, X, y, split=0.3, method='f1', folds=1):
        indices = np.arange(len(X))
        set_ind = set(indices)
        size = np.int(len(X)*(1-split))
        scores = np.zeros(folds)
        for f in range(folds):
            train = np.random.choice(indices, size, replace=True)
            set_train = set(train)
            set_test = list(set_ind.difference(set_train))
            Xtrain = X[train,:]
            ytrain = y[train]
            Xtest = X[set_test,:]
            ytest = y[set_test]
            self.train(Xtrain,ytrain)
            scores[f] = self.evaluate(Xtest,ytest,method)
            print(scores[f])
        return scores

    # the describe() function returns a human-readable description of the fitted model
    def describe(self):
        if not self.trained:
            if self.depth_limit == np.inf:
                print("I am an untrained decision tree with unlimited depth")
            else:
                print("I am an untrained decision tree with depth limited to %d levels" % self.depth_limit)
        else:
            if self.depth_limit == np.inf:
                print("I am a trained decision tree with unlimited depth")
            else:
                print("I am a trained decision tree with depth limited to %d levels\n" % self.depth_limit)
            nnodes = self.__node_count()
            print("I have %d decision and terminal nodes, arranged in %d levels." % (nnodes, len(self.nodes)))

    #__find_node() finds the position of a node in the given level of the tree
    def __find_node(self, node, level):
        pos = [i for i,n in enumerate(self.nodes[level]) if n == node]
        return pos[0]