import sys
import os
import math
from sklearn.neighbors import NearestNeighbors
import random
from scipy.spatial.distance import sqeuclidean, euclidean
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from pylab import *

# Calculate the covariance matrix for the dataset X
def calculate_covariance_matrix(X, Y=np.empty((0,0))):
    if not Y.any():
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    return np.array(covariance_matrix, dtype=float)

# Normalize the dataset X
def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def avg(x):
    """Return the average (mean) of a given list"""
    return (float(sum(x)) / len(x)) if x else 0


# Calculate the distance between two vectors
def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)

    return math.sqrt(distance)


class PCA():
    """A method for doing dimensionality reduction by transforming the feature
    space to a lower dimensionality, removing correlation between features and 
    maximizing the variance along each feature axis. This class is also used throughout
    the project to plot data.
    """
    def __init__(self): pass

    # Fit the dataset to the number of principal components
    # specified in the constructor and return the transformed dataset
    def transform(self, X, n_components):
        covariance = calculate_covariance_matrix(X)

        # Get the eigenvalues and eigenvectors.
        # (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed

    def get_color_map(self, N):
        color_norm  = colors.Normalize(vmin=0, vmax=N-1)
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
        def map_index_to_rgb_color(index):
            return scalar_map.to_rgba(index)
        return map_index_to_rgb_color

    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_2d(self, X, y=None, title=None, accuracy=None, legend_labels=None):
        X_transformed = self.transform(X, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        class_distr = []

        y = np.array(y).astype(int)

        # Color map
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

        # Plot the different class distributions
        for i, l in enumerate(np.unique(y)):
            _x1 = x1[y == l]
            _x2 = x2[y == l]
            _y = y[y == l]
            class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

        # Plot legend
        if not legend_labels is None: 
            plt.legend(class_distr, legend_labels, loc=1)

        # Plot title
        if title:
            if accuracy:
                percent = 100 * accuracy
                plt.suptitle(title)
                plt.title("Accuracy: %.1f%%" % percent, fontsize=10)
            else:
                plt.title(title)

        # Axis labels
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig(str(title) + str(".png"))


class KMeans():
    def __init__(self, k=2, max_iterations=50):
        self.k = k
        self.max_iterations = max_iterations
        self.precomputed = None

    # Initialize the centroids as random samples
    def _init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # Return the index of the closest centroid to the sample
    def _closest_centroid(self, samples, centroids):
        NN = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(centroids)
        dist, closest_ind = NN.kneighbors(samples)
        closest_ind = closest_ind.flatten()
        return closest_ind    

    # Assign the samples to the closest centroids to create clusters
    def _create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        closest_indices = self._closest_centroid(X, centroids)
        for i in range(len(closest_indices)):
            clusters[closest_indices[i]].append(i)
        return clusters

    # Calculate new centroids as the means of the samples
    # in each cluster
    def _calculate_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # Classify samples as the index of their clusters
    def _get_cluster_labels(self, clusters, X):
        # One prediction for each sample
        y_pred = np.zeros(np.shape(X)[0], dtype= int)
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = int(cluster_i)
        return y_pred

    def _get_wc_ssd_score(self, X, C, labels):
        return sum(sum((X - C[labels])**2))

    def _get_sili_sc_score(self, X, C, labels, index = None):
        if index == None:
            return avg([self._get_sili_sc_score(X, C, labels, i) for i in range(np.shape(X)[0])])
        cind = labels[index]
        a = euclidean(X[index], C[cind])
        b = min([euclidean(X[index], c) for i,c in enumerate(C) if i != cind])
        return float(b - a) / max(a, b) if max(a, b) > 0 else 0.0

    def _get_nmi_score(self, y, ind):

        y = array(y, dtype=int)
        pc = unique(y, return_counts=True)[1]
        pc = pc * 1.0 / sum(pc)
        logpc = log(pc)
        n_classes = size(pc)
        pg = unique(ind, return_counts=True)[1]
        pg = pg * 1.0 / sum(pg)
        logpg = log(pg)
        n_clusters = size(pg)
        pcg = histogram(n_classes * ind + y,
                        bins=arange(n_clusters * n_classes + 1))[0] \
                        .reshape(n_clusters, n_classes)

        pcg = pcg * 1.0 / sum(pcg) 
        tmp = pcg / pc / pg[:, None]

        tmp[tmp == 0] = 1.
        numer = sum(pcg * log(tmp))
        denom = - dot(pc, logpc) - dot(pg, logpg)

        return numer / denom

    def predict(self, X, y = None):
        centroids = self._init_random_centroids(X)
        result = [];
        for i in range(self.max_iterations):
            clusters = self._create_clusters(centroids, X)
            prev_centroids = centroids
            centroids = self._calculate_centroids(clusters, X)
            diff = centroids - prev_centroids
            if not diff.any():
                break
        labels = self._get_cluster_labels(clusters, X)
        result.append(labels)
        result.append(self._get_wc_ssd_score(X, centroids, labels))
        result.append(self._get_sili_sc_score(X, centroids, labels))
        if y is not None:
            result.append(self._get_nmi_score(y, labels));
        return result;

def create_alternate_files():
    textdata = np.genfromtxt('digits-embedding.csv', delimiter=',')
    y = textdata[:, 1]
    np.savetxt('digits-embedding-2or4or6or7', textdata[np.logical_or(y == 2, np.logical_or(y == 4, np.logical_or(y == 6, y == 7)))], delimiter=',')
    np.savetxt('digits-embedding-6or7', textdata[np.logical_or(y == 6, y == 7)], delimiter=',')

def get_random_subsets(data, n_subsets, n_digits):
    n_samples = np.shape(data)[0]
    # Concatenate x and y and do a random shuffle
    np.random.shuffle(data)
    subsets = []
    subsample_size = n_subsets 
    for num in range(n_digits):
        ind = data[:, 1] == num
        idx = np.random.choice(
            ind,
            size=np.shape(
                range(subsample_size)),
            replace=False)
        subsets.append([data[ind, 2:][idx], data[ind, 1][idx]])
    return subsets


def main():
    if len(sys.argv) == 3:
        train_data_file = sys.argv[1]
        K =[]
        K.append(int(sys.argv[2]))
        filenames = []
        filenames.append(train_data_file)
        y_pred = []
        wc_ssd = np.zeros((len(filenames), len(K)))
        sili_sc= np.zeros((len(filenames), len(K)))
        nmi_sc= np.zeros((len(filenames), len(K)))
        for f, filename in enumerate(filenames):  
            X = np.genfromtxt(filename, delimiter=',')[:, 2:]
            y = np.genfromtxt(filename, delimiter=',')[:, 1]
            for indk, k in enumerate(K):
                while True:
                        try:
                            kmeans = KMeans(k=k,max_iterations= 50)
                            prediction = kmeans.predict(X, y)
                            break;
                        except:
                            pass
                wc_ssd[f, indk] =  prediction[1]
                sili_sc[f, indk] = prediction[2]
                nmi_sc[f, indk] = prediction[3]
        for f, filename in enumerate(filenames):
            print("WC-SSD %.3f" % wc_ssd[f,0])
            print("SC %.3f" % sili_sc[f,0])
            print("NMI %.3f" % nmi_sc[f,0])

    elif len(sys.argv) == 2:
        create_alternate_files();
        analysis = sys.argv[1]
        if analysis == "A1" :
            print("Performing Analysis A1")
            images = np.genfromtxt('digits-raw.csv', delimiter=',')
            getrandom = np.random.randint(images.shape[0])
            image = images[getrandom][2:]
            reshaped_image = image.reshape(28, 28)
            fig, ax = plt.subplots()
            ax.imshow(reshaped_image, cmap='gray')
            plt.title("Analysis1");
            plt.savefig("Analysis1.png");
        elif analysis == "A2" :
            print("Performing Analysis A2")
            embeddings = np.genfromtxt('digits-embedding.csv', delimiter=',')
            permutations = np.random.permutation(embeddings.shape[0])[:1000]
            pca = PCA()
            pca.plot_in_2d(embeddings[permutations, 2:], embeddings[permutations, 1], title="Analysis2")
        elif analysis == "B1" :
            K = [2, 4, 8, 16, 32]
            filenames = ['digits-embedding.csv', 'digits-embedding-2or4or6or7', 'digits-embedding-6or7']
            y_pred = []
            wc_ssd = np.zeros((len(filenames), len(K)))
            sili_sc= np.zeros((len(filenames), len(K)))
            for f, filename in enumerate(filenames):  
                X = np.genfromtxt(filename, delimiter=',')[:, 2:]
                y = np.genfromtxt(filename, delimiter=',')[:, 1]
                for indk, k in enumerate(K):
                    kmeans = KMeans(k=k,max_iterations= 50)
                     # Cluster the data using K-Means
                    prediction = kmeans.predict(X, y)
                    wc_ssd[f, indk] =  prediction[1]
                    sili_sc[f, indk] = prediction[2]

            figDesc = "WC_SSD vs K"
            import matplotlib.pyplot as plt
            plt.ioff()
            fig = plt.figure()
            fig.suptitle(figDesc, fontsize=14, fontweight='bold')
            ax = fig.add_subplot(111)
            ax.set_xlabel("K")
            ax.set_ylabel("WC_SSD")
            for f, filename in enumerate(filenames):
                ax.plot(K, wc_ssd[f], label = filename)
            ax.autoscale_view(True,True,True)
            ax.legend(loc='upper right')
            plt.savefig(figDesc)
            
            figDesc = "SC vs K"
            plt.ioff()
            fig = plt.figure()
            fig.suptitle(figDesc, fontsize=14, fontweight='bold')
            ax = fig.add_subplot(111)
            ax.set_xlabel("K")
            ax.set_ylabel("SC")
            for f, filename in enumerate(filenames):
                ax.plot(K, sili_sc[f], label = filename)
            ax.autoscale_view(True,True,True)
            ax.legend(loc='upper right')
            plt.savefig(figDesc)
        elif analysis == "B1" :
            K = [2, 4, 8, 16, 32]
            filenames = ['digits-embedding.csv', 'digits-embedding-2or4or6or7', 'digits-embedding-6or7']
            y_pred = []
            wc_ssd = np.zeros((len(filenames), len(K)))
            sili_sc= np.zeros((len(filenames), len(K)))
            for f, filename in enumerate(filenames):  
                X = np.genfromtxt(filename, delimiter=',')[:, 2:]
                y = np.genfromtxt(filename, delimiter=',')[:, 1]
                for indk, k in enumerate(K):
                    while True:
                        try:
                            kmeans = KMeans(k=k,max_iterations= 50)
                            prediction = kmeans.predict(X)
                            break;
                        except:
                            pass
                    wc_ssd[f, indk] =  prediction[1]
                    sili_sc[f, indk] = prediction[2]

            figDesc = "B1. WC_SSD vs K"
            import matplotlib.pyplot as plt
            plt.ioff()
            fig = plt.figure()
            fig.suptitle(figDesc, fontsize=14, fontweight='bold')
            ax = fig.add_subplot(111)
            ax.set_xlabel("K")
            ax.set_ylabel("WC_SSD")
            for f, filename in enumerate(filenames):
                ax.plot(K, wc_ssd[f], label = filename)
            ax.autoscale_view(True,True,True)
            ax.legend(loc='upper right')
            plt.savefig(figDesc)
            
            figDesc = "B1. SC vs K"
            plt.ioff()
            fig = plt.figure()
            fig.suptitle(figDesc, fontsize=14, fontweight='bold')
            ax = fig.add_subplot(111)
            ax.set_xlabel("K")
            ax.set_ylabel("SC")
            for f, filename in enumerate(filenames):
                ax.plot(K, sili_sc[f], label = filename)
            ax.autoscale_view(True,True,True)
            ax.legend(loc='upper right')
            plt.savefig(figDesc)

        elif analysis == "B3" :
            K = [2, 4, 8, 16, 32]
            filenames = ['digits-embedding.csv', 'digits-embedding-2or4or6or7', 'digits-embedding-6or7']
            y_pred = []
            wc_ssd = np.zeros((len(filenames), len(K), 10))
            sili_sc= np.zeros((len(filenames), len(K), 10))
            for f, filename in enumerate(filenames):  
                X = np.genfromtxt(filename, delimiter=',')[:, 2:]
                y = np.genfromtxt(filename, delimiter=',')[:, 1]
                for indk, k in enumerate(K):
                    print(k)
                    for j in range(10):
                        # Cluster the data using K-Means
                        while True:
                            try:
                                kmeans = KMeans(k=k,max_iterations= 50)
                                prediction = kmeans.predict(X)
                                break;
                            except:
                                pass

                        wc_ssd[f, indk,j] =  prediction[1]
                        sili_sc[f, indk,j] = prediction[2]
            wc_ssd_means = mean(wc_ssd, axis=2)
            sili_sc_means = mean(sili_sc, axis=2)
            wc_ssd_std = std(wc_ssd, axis=2)
            sili_sc_std = std(sili_sc, axis=2)
            figDesc = "B3. WC_SSD vs K"
            import matplotlib.pyplot as plt
            plt.ioff()
            fig = plt.figure()
            fig.suptitle(figDesc, fontsize=14, fontweight='bold')
            ax = fig.add_subplot(111)
            ax.set_xlabel("K")
            ax.set_ylabel("WC_SSD")
            for f, filename in enumerate(filenames):
                ax.errorbar(K, wc_ssd_means[f], yerr=wc_ssd_std[f], label =filename)
            ax.autoscale_view(True,True,True)
            ax.legend(loc='upper right')
            plt.savefig(figDesc)
            
            figDesc = "B3. SC vs K"
            plt.ioff()
            fig = plt.figure()
            fig.suptitle(figDesc, fontsize=14, fontweight='bold')
            ax = fig.add_subplot(111)
            ax.set_xlabel("K")
            ax.set_ylabel("SC")
            for f, filename in enumerate(filenames):
                ax.errorbar(K, sili_sc_means[f], yerr=sili_sc_std[f], label =filename)
            ax.autoscale_view(True,True,True)
            ax.legend(loc='upper right')
            plt.savefig(figDesc)

        elif analysis == "B4":
            K = [[8], [4], [2]]
            filenames = ['digits-embedding.csv', 'digits-embedding-2or4or6or7', 'digits-embedding-6or7']
            y_pred = []
            wc_ssd = np.zeros((len(filenames), len(K)))
            sili_sc= np.zeros((len(filenames), len(K)))
            nmi_sc= np.zeros((len(filenames), len(K)))
            for f, filename in enumerate(filenames):  
                X = np.genfromtxt(filename, delimiter=',')[:, 2:]
                y = np.genfromtxt(filename, delimiter=',')[:, 1]
                for indk, k in enumerate(K[f]):
                    while True:
                        try:
                            kmeans = KMeans(k=k,max_iterations= 50)
                            prediction = kmeans.predict(X, y)
                            break;
                        except:
                            pass
                    pca = PCA()
                    pca.plot_in_2d(X, prediction[0], title="%s %s" % ("B.4 K =" ,k, "-Means Clustering-", filename ))
                    wc_ssd[f, indk] =  prediction[1]
                    sili_sc[f, indk] = prediction[2]
                    nmi_sc[f, indk] = prediction[3]
            for f, filename in enumerate(filenames):
                print("NMI %.3f" % nmi_sc[f,0], " filename -> ", filename, " K- > ", K[f])

        elif analysis == "C1":
            dataset= np.genfromtxt('digits-embedding.csv', delimiter=',')
            subsets = get_random_subsets(dataset, 10, 10);
            single_linkage = linkage(X, method='single')
            dendrogram(single_linkage, leaf_font_size=16)
            figDesc = "WC_SSD vs K"
            import matplotlib.pyplot as plt
            plt.ioff()
            fig = plt.figure()
            fig.suptitle(figDesc, fontsize=14, fontweight='bold')
            ax = fig.add_subplot(111)
            ax.set_xlabel("K")
            ax.set_ylabel("WC_SSD")
            for f, filename in enumerate(filenames):
                ax.errorbar(K, wc_ssd_means[f], yerr=wc_ssd_std[f], label =filename)
            ax.autoscale_view(True,True,True)
            ax.legend(loc='upper right')
            plt.savefig(figDesc)

        elif analysis == "C2":
            None
    
        elif analysis == "C3":
            None

        elif analysis == "C5":
            None

        elif analysis == "Bon1":
            None

        elif analysis == "Bon2":
            None

        elif analysis == "Bon3":
            None

        elif analysis == "Bon4":
            None

        elif analysis == "Bon5":
            None

        else:
            print('usage: python hw5.py InsertAnalysisId')

    else:
        print('usage: python hw5.py dataFileName K')
        print('-->-SEE->--  Alternate usage: python hw5.py InsertWhichAnalysisId')
        sys.exit(-1)

if __name__ == "__main__":
    np.seterr(all='raise')
    main()