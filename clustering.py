import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def parse_file(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            try:
                row = [float(x) for x in line.split()]
                data.append(row)
            except ValueError:
                print(f"Skipping invalid row: {line}")
    return np.array(data)

def kmeans_clustering(data, n_clusters=3, init='k-means++', n_init=10):
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init)
    kmeans.fit(data)
    return kmeans

def plot_clusters(data, kmeans):
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3,
                color='r', zorder=10)
    plt.show()

file_names = ['s1.txt', 's2.txt', 's3.txt', 's4.txt', 'spiral.txt']
for file_name in file_names:
    data = parse_file(file_name)
    kmeans = kmeans_clustering(data)
    plot_clusters(data, kmeans)