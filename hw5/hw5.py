import sys
import os
import math
import random
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

from sklearn.metrics import silhouette_score
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

    # Initialize the centroids as random samples
    def _init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # Return the index of the closest centroid to the sample
    def _closest_centroid(self, sample, centroids):
        closest_i = None
        closest_distance = float("inf")
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(sample, centroid)
            if distance < closest_distance:
                closest_i = i
                closest_distance = distance
        return closest_i

    # Assign the samples to the closest centroids to create clusters
    def _create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
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
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # Do K-Means clustering and return cluster indices
    def predict(self, X):
        # Initialize centroids
        centroids = self._init_random_centroids(X)
        result = [];
        # Iterate until convergence or for max iterations
        for _ in range(self.max_iterations):
            # Assign samples to closest centroids (create clusters)
            clusters = self._create_clusters(centroids, X)
            prev_centroids = centroids
            # Calculate new centroids from the clusters
            centroids = self._calculate_centroids(clusters, X)

            # If no centroids have changed => convergence
            diff = centroids - prev_centroids
            if not diff.any():
                break

        result.append(self._get_cluster_labels(clusters, X))


        return result;


def main():
    if len(sys.argv) >= 3:
        train_data_file = sys.argv[1]
        K = sys.argv[2]
        analysis = sys.argv[3]
        if analysis == "A1" :
            print("Performing Analysis A1")
            images = np.genfromtxt('digits-raw.csv', delimiter=',')
            getrandom = np.random.randint(images.shape[0])
            image = images[getrandom][2:]
            reshaped_image = image.reshape(28, 28)
            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray')
            plt.title("Analysis1");
            plt.show()
        elif analysis == "A2" :
            print("Performing Analysis A2")
            emb = np.genfromtxt('digits-embedding.csv', delimiter=',')
            perm = np.random.permutation(emb.shape[0])[:1000]
            labels = np.array(emb[perm, 1], dtype=int)
            colors = np.random.rand(10, 3)[labels, :]
            fig, ax = plt.subplots()
            plt.scatter(emb[perm, 2], emb[perm, 3], c=colors, alpha=0.9, s=10)
            plt.title("Analysis2");
            plt.savefig(str(plt.title))
        elif analysis == "B1" :
            K = [2, 4, 6, 8, 16, 32]
            fnames = ['digits-embedding.csv']
            #wc_ssd = np.zeros((len(fnames), len(K)))
            #sili_sc = np.zeros((len(fnames), len(K)))
            for i, fname in enumerate(fnames):  
                X = np.genfromtxt(fname, delimiter=',')[:, 2:]
                y = np.genfromtxt(fname, delimiter=',')[:, 1]
                for j, k in enumerate(K):
                    kmeans = KMeans(k=k,max_iterations= 50)
                     # Cluster the data using K-Means
                    y_pred = kmeans.predict(X)
                    print(y_pred)
                    pca = PCA()
                    pca.plot_in_2d(X, y_pred[0], title= str("K=") + str(k) + str("-Means Clustering"))
                    print("sklearn SC =", silhouette_score(X, y_pred[0]))

            pca.plot_in_2d(X, y, title="Actual Clustering")
            #plot_wc_ssd(wc_ssd);
            #plot_silhouette_score(sili_sc);
    else:
        print('usage: python hw5.py dataFileName K')
        sys.exit(-1)

if __name__ == "__main__":
    main()