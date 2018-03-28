import numpy as np
from MethodWrapper import MethodWrapper
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pylab

class DBSCANWrapper(MethodWrapper, name = "DBSCAN"):
    """DBSCAN wrapper"""
    
    def __init__(self):
        self.eps = 0.1
        self.min_samples = 3
        self.metric = "euclidean"
        self.algorithm = "auto"
        self.leaf_size = 30
        self.animation_delay = -1
        self.p = 2
        self.total_points = 300

    
    def set_eps(self, value:str):
        self.eps = float(value)

    def set_min_samples(self, value:str):
        self.min_samples = int(value)

    def set_metric(self, value:str):
        if value in ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]:
            self.metric = value
        else:
            self.metric = eval(value)
    
    def set_algorithm(self, value:str):
        if value in  ["auto", "ball_tree", "kd_tree", "brute"]:
            self.algorithm = value

    def set_leaf_size(self, value:str):
        self.leaf_size = int(value)

    def set_animation_delay(self, value:str):
        self.animation_delay = float(value)
    
    def set_p(self, value:str):
        self.p = int(value)

    def set_total_points(self, value:str):
        self.total_points = int(value)

    def animate(self, colors, core_samples_mask, db, labels, unique_labels, X):
         
        black = [0, 0, 0, 1]
        plt.plot(X[:, 0], X[:, 1],  'o', markerfacecolor= black,
                     markeredgecolor = "k", markersize=5)
        plt.show()
        plt.ion()
        plt.pause(1)

        #simulate animation since no way to interfere in real processing
        from sklearn.neighbors import NearestNeighbors
        neighborFinder = NearestNeighbors(radius=self.eps, algorithm=self.algorithm,
                                          leaf_size=self.leaf_size, metric=self.metric, p=self.p)
        neighborFinder.fit(X) 
        neighborhood = neighborFinder.radius_neighbors(X, radius = self.eps, return_distance = False)
        visited = set()  
        for k, col in zip(unique_labels, colors):
            if k != - 1:
                class_member_mask = (labels == k)
                core = X[class_member_mask & core_samples_mask]
                point_indices = [np.where(X == core[0])[0][0]]                             
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
                    for n in neighborhood[point_index]:
                        if ((n not in visited) & (db.labels_[n] == k)):
                            visited.add(n)                                
                            point_indices.append(n)

    def display(self, colors, core_samples_mask, labels, unique_labels, X):
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
        
            class_member_mask = (labels == k)
        
            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=10)
        
            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=5)
        plt.show()
    def execute(self):

        #temp code
        #should be replaced with custom source
        centers = [[1, 1], [-1, -1], [1, -1]]
        X, labels_true = make_blobs(n_samples=self.total_points, centers=centers, cluster_std=0.4,
                                    random_state=0)

        X = StandardScaler().fit_transform(X)

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)        

        unique_labels = set(labels)
        colors = [plt.get_cmap("gist_ncar")(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
                
        if self.animation_delay <= 0:   
            self.display(colors, core_samples_mask, labels, unique_labels, X)
        else:
            self.animate(colors, core_samples_mask, db, labels, unique_labels, X)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.ioff()
