import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random


def getdata():
    X_train, true_labels = make_blobs(
        n_samples=100, centers=centers, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)
    return X_train, true_labels


def calculate_distances(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))

class KMeans:
    def __init__(self, cluster_count=3, max_iter=300):
        self.cluster_count = cluster_count
        self.max_iter = max_iter

    def fit(self, X_train):
        self.centroids = [random.choice(X_train)]
        for _ in range(self.cluster_count-1):
            dists = np.sum([calculate_distances(centroid, X_train)
                           for centroid in self.centroids], axis=0)
            dists /= np.sum(dists)
            new_centroid_idx, = np.random.choice(
                range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]

        iteration = 0
        prev_centroids = None
        should_iterate = True
        while should_iterate and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.cluster_count)]
            for x in X_train:
                dists = calculate_distances(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0)
                              for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
            should_iterate = np.not_equal(self.centroids, prev_centroids).any()

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = calculate_distances(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs


centers = 3
X_train, true_labels = getdata()
kmeans = KMeans(cluster_count=centers)
kmeans.fit(X_train)
class_centers, classification = kmeans.evaluate(X_train)
sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=true_labels,
                style=classification,
                palette="deep",
                legend=None
                )
plt.plot([x for x, _ in kmeans.centroids],
         [y for _, y in kmeans.centroids],
         'k+',
         markersize=10,
         )
plt.show()
