import numpy as np
from MethodWrapper import MethodWrapper
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from random import randint
import matplotlib.pyplot as plt
import pylab

class AgglomerativeClusteringWrapper(MethodWrapper, name = "AgglomerativeClusteringWrapper"):
    """AgglomerativeClustering wrapper"""
    
    def __init__(self):
        MethodWrapper.__init__(self)
        self.n_clusters = 2
        self.affinity = "euclidean"
        self.compute_full_tree = "auto"
        self.linkage = "ward"
    ##    self.total_points = 200

    
    def set_n_clusters(self, value:str):
        self.n_clusters = int(value)

    def set_affinity(self, value:str):
        self.affinity = str(value)
    
    ##def set_algorithm(self, value:str):
    ##    if value in  ["auto", "ball_tree", "kd_tree", "brute"]:
    ##        self.algorithm = value

    def set_compute_full_tree(self, value:str):
        self.compute_full_tree = str(value)
    
    def set_linkage(self, value:str):
        self.linkage = str(value)

    ##def set_total_points(self, value:str):
    ##    self.total_points = int(value)

    def execute(self, dataset):
        #centers = [[1, 1], [-1, -1], [1, -1]]
        #dataset, labels_true = make_blobs(n_samples=self.total_points, centers=centers, cluster_std=0.4,random_state=0)
        #dataset = StandardScaler().fit_transform(dataset)
        dataset, labels_true = dataset
        dataset = StandardScaler().fit_transform(dataset)
        digits = datasets.load_digits(n_class=10)
        X = dataset
        #y = digits.target
        #n_samples, n_features = X.shape
        #np.random.seed(0)

        #shift = lambda x: ndimage.shift(x.reshape((8, 8)),.3 * np.random.normal(size=2),mode='constant',).ravel()
        #X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
        #Y = np.concatenate([y, y], axis=0)
        #X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
        X_red = StandardScaler().fit_transform(X)
        #for linkage in ('ward', 'average', 'complete'):
        #    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
        #    t0 = time()
        #    clustering.fit(X_red)
        #    print("%s : %.2fs" % (linkage, time() - t0))

        #    plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)

        clustering = AgglomerativeClustering(n_clusters=self.n_clusters, affinity=self.affinity, compute_full_tree=self.compute_full_tree,linkage=self.linkage)
        clustering.fit(X_red)
        labels = clustering.labels_
        #plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)
        x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
        X_red = (X_red - x_min) / (x_max - x_min)

        plt.figure(figsize=(6, 4))
        for i in range(X_red.shape[0]):
            plt.text(X_red[i, 0], X_red[i, 1], str(randint(0, 9)),
                     color=plt.cm.spectral(clustering.labels_[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})

        plt.xticks([])
        plt.yticks([])
        #if title is not None:
        #    plt.title(title, size=17)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        stats = self.get_stats(labels_true, labels)
        print(stats.get_formatted())