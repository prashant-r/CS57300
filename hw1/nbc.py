import csv, sys, argparse, string, operator, random,pandas as pd, numpy as np
from collections import Counter
from string import punctuation
import pylab as pl


class NaiveBayesClassifier:
	def __init__(self, sample):
		self.Sample = sample
		self.FeatureVector = []
		self.BagOfWords = []
		self.pOfPositive = 0
		self.pOfNegative = 0
		self.pOfWordGivenPositive = []
		self.pOfWordGivenNegative = []

	def learn(self, discard, retain, topwords):
		words = list()
		PositiveCount = 0
		NegativeCount = 0
		for line in self.Sample:
			for w in line[1]:
				if(w.isdigit() is False):
					words.append(w)
			if(line[0] == 1):
				PositiveCount+=1
			else:
				NegativeCount+=1
		topwords = Counter(words)
		self.FeatureVector = topwords
		for a in range(len(self.Sample)):
			bitVector = np.zeros(len(self.FeatureVector), dtype=np.int)
			for i in range(len(self.FeatureVector)):
				if self.FeatureVector[i] in self.Sample[a][1]:
					bitVector[i] = 1
				else:
					bitVector[i] = 0

			self.BagOfWords.append(bitVector)
		if( len(self.FeatureVector) > 0):
			self.pOfPositive = np.divide((PositiveCount + 1.0),(PositiveCount + NegativeCount + 2.0))
			self.pOfNegative = np.divide((NegativeCount + 1.0),(NegativeCount + PositiveCount + 2.0))

			positiveWordFreqArray = np.zeros((len(self.FeatureVector),), dtype = np.int)
			negativeWordFreqArray = np.zeros((len(self.FeatureVector),), dtype = np.int)

			for i in range(len(self.Sample)):
				if self.Sample[i][0] == 1:
					positiveWordFreqArray += self.BagOfWords[i];
				else:
					negativeWordFreqArray += self.BagOfWords[i];

			for i in range(len(self.FeatureVector)):
				self.pOfWordGivenPositive.append(np.divide(positiveWordFreqArray[i] + 1.0 ,(PositiveCount + 2.0)))
				self.pOfWordGivenNegative.append(np.divide(negativeWordFreqArray[i] + 1.0 ,(NegativeCount + 2.0)))


	def classify(self, wVector):
		# b + w ⊤ x
		prior = np.log(np.divide(self.pOfPositive, self.pOfNegative))
		prediction = prior
		numerator = 0
		denominator =0
		for i in range(len(wVector)):
			if(wVector[i] == 1):
				prediction += np.log(np.divide(self.pOfWordGivenPositive[i], self.pOfWordGivenNegative[i]))
			else:
				prediction += np.log(np.divide(1- self.pOfWordGivenPositive[i], 1 - self.pOfWordGivenNegative[i]))

		if prediction > 0 :
			return 1
		else:
			return 0

	def test(self, testData):
		words = list()
		mistakes = 0.0
		numCorrect = 0
		numWrong = 0
		priorm = 0.0
		prior = np.log(np.divide(self.pOfPositive, self.pOfNegative))
		if prior < 0 :
			guess = 0
		else:
			guess = 1
		for line in testData:
			for word in line[1]:
				words.append(word)
					# print top 3 with 'he' prefix

		for a in range(len(testData)):
			bitVector = np.zeros(len(self.FeatureVector), dtype=np.int)
			for i in range(len(self.FeatureVector)):
				if self.FeatureVector[i] in testData[a][1]:
					bitVector[i] = 1
				else:
					bitVector[i] = 0
			pred = self.classify(bitVector)
			if pred == 1 and testData[a][0] == 0:
				mistakes += 1
			elif pred == 0 and testData[a][0] == 1:
				mistakes += 1
			if guess != testData[a][1]:
				priorm +=1
		return (mistakes / len(testData)), (priorm/len(testData))

def plot(metrics):
	# use pylab to plot x and y
	x1 = []
	y1 = []
	x2 = []
	y2 = []
	for metric in metrics:
		x1.append(metric[0])
		y1.append(metric[1])
		x2.append(metric[0])
		y2.append(metric[2])
	pl.plot(x1, y1, label='zero-one loss')
	pl.plot(x2, y2, label= 'guess loss')
	pl.legend(loc='upper left')
	# show the plot on the screen
	pl.show()

def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
            dataset.append( (int(class_label), set(words)) )
    return dataset

def process_str(s):
    return s.translate(string.punctuation).lower().split()


def get_most_commons(dataset, skip=100, total=100):
    my_list = []
    for item in dataset:
        my_list += list(item[1])

    counter = Counter(my_list)

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words


def create_train_test(percentage):
	data = read_dataset('yelp_data.csv')
	random.shuffle(data)
	train_data = data[:int((len(data)+1)*(percentage))] #Remaining percentage% to training set
	test_data = data[int((len(data) + 1)*(percentage)):] #Splits percetage% data to test set
	return train_data, test_data

def compute_metrics(zolosses, guesslosses):
	mzoloss = np.mean(zolosses)
	mguessloss = np.mean(guesslosses)
	stdzoloss = np.std(zolosses)
	stdguessloss = np.std(guesslosses)
	return (mzoloss, mguessloss, stdzoloss, stdguessloss)

def main(args):
	small_sample_mode = False;
	metrics = []
	if small_sample_mode:
		PartC = False
		if PartC :
			percentages = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.9])
			for takeSample in percentages:
				zoLosses = []
				guessLosses = []
				for i in range(10):
					ftrain, ftest = create_train_test(takeSample)
					naiveBayes = NaiveBayesClassifier(ftrain);
					naiveBayes.learn(100,500);
					zoloss = naiveBayes.test(ftest);
					zoLosses.append(zoloss[0])
					guessLosses.append(zoloss[1])
				metric = compute_metrics(zoLosses, guessLosses)
				metrics.append((len(ftrain), metric[0], metric[1]))
			plot(metrics)
		else:
			takeSample = 0.5
			wArr = np.array([10, 50, 250, 500, 1000, 4000])
			for w in wArr:
				zoLosses = []
				guessLosses = []
				for i in range(10):
					ftrain, ftest = create_train_test(takeSample)
					naiveBayes = NaiveBayesClassifier(ftrain);
					naiveBayes.learn(100,w);
					zoloss = naiveBayes.test(ftest);
					zoLosses.append(zoloss[0])
					guessLosses.append(zoloss[1])
				metric = compute_metrics(zoLosses, guessLosses)
				metrics.append((w, metric[0], metric[1]))
			plot(metrics)
	else:
		parser = argparse.ArgumentParser()
		parser.add_argument('trainingDataFilename')
		parser.add_argument('testDataFilename')
		vals = parser.parse_args(args)
		ftrain = read_dataset(vals.trainingDataFilename)
		ftest = read_dataset(vals.testDataFilename)
		top_ten = get_most_commons(ftrain, skip=100, total=10)
		for i in range(len(top_ten)):
			print('WORD' + str(i+1) +' '+ top_ten[i])
		common_words = get_most_commons(ftrain, skip=100, total=500)
	naiveBayes = NaiveBayesClassifier(ftrain);
	naiveBayes.learn(100,500, common_words);
	zoloss = naiveBayes.test(ftest);
	print('ZERO-ONE LOSS' , zoloss[0]);
if __name__ == "__main__":
	main(sys.argv[1:])
