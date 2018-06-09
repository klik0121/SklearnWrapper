import numpy as np
from MethodWrapper import MethodWrapper
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pylab

class DBSCANWrapper(MethodWrapper, name = "DBSCAN"):
    """DBSCAN wrapper"""

    def __init__(self):
        MethodWrapper.__init__(self)
        self.eps = 0.1
        self.min_samples = 3
        self.metric = "euclidean"
        self.algorithm = "auto"
        self.leaf_size = 30
        self.p = 2

    def compute_animate(self, colors, core_samples_mask, db, labels, unique_labels, dataset, animate, outfile):

        black = [0, 0, 0, 1]
        plt.plot(dataset[:, 0], dataset[:, 1],  'o', markerfacecolor= black,
                     markeredgecolor = "k", markersize=5)
        plt.ion()
        if animate:
            plt.show()
            plt.pause(1)

        #simulate animation since there is no way to interfere in real processing
        from sklearn.neighbors import NearestNeighbors
        neighborFinder = NearestNeighbors(radius=self.eps, algorithm=self.algorithm,
                                          leaf_size=self.leaf_size, metric=self.metric, p=self.p)
        neighborFinder.fit(dataset)
        neighborhood = neighborFinder.radius_neighbors(dataset, radius = self.eps, return_distance = False)
        visited = set()

        with open(outfile, "a") as result_file:
            for k, col in zip(unique_labels, colors):
                if k != - 1:
                    class_member_mask = (labels == k)
                    core = dataset[class_member_mask & core_samples_mask]
                    point_indices = [np.where(dataset == core[0])[0][0]]
                    while point_indices:
                        point_index = point_indices.pop()
                        visited.add(point_index)
                        point = dataset[point_index]
                        if point_index in db.core_sample_indices_:
                            result_file.write("Cluster: " + str(k) + ", Type: Core, " + str(list(point)) + "\n")
                            if animate:
                                plt.plot(point[0], point[1], 'o', markerfacecolor= col,
                                    markeredgecolor = "k", markersize=10)
                        else:
                            result_file.write("Cluster: " + str(k) + ", Type: Common, " + str(list(point)) + "\n")
                            if animate:
                                plt.plot(point[0], point[1], 'o', markerfacecolor= col,
                                    markeredgecolor = "k", markersize=5)
                        if self.animation_delay > 0 & animate:
                            plt.pause(self.animation_delay)
                        for n in neighborhood[point_index]:
                            if ((n not in visited) & (db.labels_[n] == k)):
                                visited.add(n)
                                point_indices.append(n)
                result_file.write("\n")

    def draw(self, colors, core_samples_mask, labels, unique_labels, dataset, outfile):
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = dataset[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=10)

            xy = dataset[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=5)
        plt.show()

    def execute(self, dataset):

        #temp code
        #should be replaced with custom source
        dataset, labels_true = dataset
        dataset = StandardScaler().fit_transform(dataset)

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(dataset)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        #temp code
        #should be replaced with create file dialog
        outfile = "result.txt"
        open(outfile, 'w').close()

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        unique_labels = set(labels)
        colors = [plt.get_cmap("gist_ncar")(each)
                  for each in np.linspace(0, 1, len(unique_labels))]


        self.compute_animate(colors, core_samples_mask, db, labels, unique_labels, dataset,
                             (dataset.shape[1] < 3) & (self.animation_delay > 0), outfile)
        #if animation is disabled and dataset is drawable
        if (dataset.shape[1] < 3) & (self.animation_delay <= 0):
            self.draw(colors, core_samples_mask, labels, unique_labels, dataset, outfile)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.ioff()
