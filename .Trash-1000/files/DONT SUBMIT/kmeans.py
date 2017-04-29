from pylab import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, mutual_info_score

def wc_ssd(X, C, ind):
	'''
		Within Cluster Sum of Squared Distance.
	'''
	return sum((X - C[ind])**2)

def sc(X, C, ind):
	'''
		Silhouette Coefficient.
	'''
	def _sc(x):
		'''
			Compute SC for a single sample.
		'''
		# x[-1] is encoded as the ind of x
		dist = norm(X - x[:-1], axis=1)
		nc_ind = argsort(norm(C - x[:-1], axis=1))[1]
		# Avoid singleton cluster
		if sum(ind == x[-1]) == 1: 
			return 1. 
		a = sum(dist[ind == x[-1]]) / (sum(ind == x[-1]) - 1)
		b = mean(dist[ind == nc_ind])
		return (b - a) / max(a, b)
	# Encode ind along with X
	X_enc = hstack((X, ind.reshape(-1, 1)))
	return mean(apply_along_axis(_sc, 1, X_enc))

def nmi(y, ind):
	'''
		Normalized Mutual Information Gain.
	'''
	y = array(y, dtype=int)
	pc = unique(y, return_counts=True)[1]
	pc = pc * 1.0 / sum(pc)
	logpc = log(pc)
	n_classes = size(pc)
	pg = unique(ind, return_counts=True)[1]
	pg = pg * 1.0 / sum(pg)
	logpg = log(pg)
	n_clusters = size(pg)
	# Compute the contingency table
	# pcg is n_clusters x n_classes
	pcg = histogram(n_classes * ind + y,
					bins=arange(n_clusters * n_classes + 1))[0] \
					.reshape(n_clusters, n_classes)
	pcg = pcg * 1.0 / sum(pcg) 
	# Setting zero entries to 1 to avoid log0
	# Zero entries should have no contribution to NMI
	tmp = pcg / pc / pg[:, None]
	tmp[tmp == 0] = 1.
	numer = sum(pcg * log(tmp))
	denom = - dot(pc, logpc) - dot(pg, logpg)
	# print "NMI numer %.3f" % numer
	return numer / denom

class KMeans(object):
	"""KMeans clustering"""
	def __init__(self, n_clusters=10, max_iter=50, debug=False):
		super(KMeans, self).__init__()
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.debug = debug
		self.WC_SSD = -1.
		self.SC = -1.
		self.NMI = -1.

	def fit(self, X, y=None):
		'''
			Compute k-means clustering.
			X is n_samples x n_features.
			y is n_samples.
		'''
		n_samples, n_features = X.shape
		# Randomly select K samples as initial centers
		perm = permutation(n_samples)[:self.n_clusters]
		C = X[perm]

		# Nearest neighbor function
		def nn(x):
			return argmin(norm(C - x, axis=1))
		# Iterate 50 times
		for i in range(50):
			# Assign points to nearest cluster centers
			NN = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(C)
			dist, ind = NN.kneighbors(X)
			ind = ind.flatten()
			# ind = apply_along_axis(nn, 1, X)
			# Update the cluster centers
			for k in range(self.n_clusters):
				C[k] = mean(X[ind == k], axis=0)
		if self.debug:
			if isinstance(y, ndarray):
				print("Class membership counts:")
				print(unique(y, return_counts=True)[1])
			print("Cluster membership counts:")
			print(unique(ind, return_counts=True)[1])
		# Compute the WC_SSD

		self.WC_SSD = wc_ssd(X, C, ind)

		# Compute the SC
		self.SC = sc(X, C, ind)
		# Compute the NMI
		if isinstance(y, ndarray):
			self.NMI = nmi(y, ind)
		print("WC-SSD %.3f" % self.WC_SSD)
		print("SC %.3f" % self.SC)
		print("NMI %.3f" % self.NMI)
		if self.debug: 
			print("sklearn SC %.3f" % silhouette_score(X, ind))
			if isinstance(y, ndarray): 
				print("sklearn NMI %.3f" % mutual_info_score(y, ind))
		return ind

	def get_evals(self):
		'''
			Return WC_SSD, SC, NMI.
		'''
		return self.WC_SSD, self.SC, self.NMI

def main():
	n_clusters = 10
	n = 1000
	X = randn(n, 2)
	y = randint(n_clusters, size=n)
	kmeans = KMeans(n_clusters, debug=True)
	ind = kmeans.fit(X, y)
	colors = rand(n_clusters, 3)[ind, :]
	scatter(X[:, 0], X[:, 1], c=colors, alpha=0.9, s=30)
	show()     

if __name__ == '__main__':
	main()