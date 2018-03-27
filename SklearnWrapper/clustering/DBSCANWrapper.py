import numpy as np
from MethodWrapper import MethodWrapper
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

class DBSCANWrapper(MethodWrapper, name = "DBSCAN"):
    """DBSCAN wrapper"""
    
    def __init__(self):
        self.eps = 0.1
        self.min_samples = 3
        self.metric = "euclidean"
        self.algorithm = "auto"
        self.leaf_size = 30
        self.animation_delay = 0.01
        self.p = 2


    def execute(self):

        #temp code
        #should be replaced with custom source
        centers = [[1, 1], [-1, -1], [1, -1]]
        X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.4,
                                    random_state=0)

        X = StandardScaler().fit_transform(X)

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        import matplotlib.pyplot as plt

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        black = [0, 0, 0, 1]
        plt.plot(X[:, 0], X[:, 1],  'o', markerfacecolor= black,
                     markeredgecolor = "k", markersize=5)
        
        plt.show()
        if self.animation_delay != 0:            
            plt.ion()
            plt.pause(1)

        #simulate animation since no way to interfere in real processing
        from sklearn.neighbors import NearestNeighbors
        neighborFinder = NearestNeighbors(radius=self.eps, algorithm=self.algorithm,
                                          leaf_size=self.leaf_size, metric=self.metric, p=self.p)
        neighborFinder.fit(X) 
        neighborhood = neighborFinder.radius_neighbors(X, radius = self.eps, return_distance = False)
        for k, col in zip(unique_labels, colors):
            if k != - 1:
                class_member_mask = (labels == k)
                core = X[class_member_mask & core_samples_mask]
                point_indices = [np.where(X == core[0])[0][0]]                
                visited = set()               
                while point_indices:
                    point_index = point_indices.pop()
                    visited.add(point_index)
                    point = X[point_index]
                    if point_index in db.core_sample_indices_:
                        plt.plot(point[0], point[1], 'o', markerfacecolor= col,
                            markeredgecolor = "k", markersize=10)
                    else:
                        plt.plot(point[0], point[1], 'o', markerfacecolor= col,
                            markeredgecolor = "k", markersize=5)
                    if self.animation_delay != 0:
                        plt.pause(self.animation_delay)
                    visited.add(point_index)
                    for n in neighborhood[point_index]:
                        if ((n not in visited) & (db.labels_[n] == k)):                                
                            point_indices.append(n)


        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.ioff()
