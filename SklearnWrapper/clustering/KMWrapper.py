from MethodWrapper import MethodWrapper
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

class KMWrapper(MethodWrapper, name = "K-Means"):
    """KMeans wrapper"""

    def __init__(self):
        MethodWrapper.__init__(self)
        self.n_clusters = 8
        self.init = 'k-means++'
        self.n_init = 10
        self.max_iter = 300
        self.tol = 1e-4
        self.precompute_distances = 'auto'
        self.verbose = 0
        self.random_state = None
        self.copy_x = True
        #self.n_jobs = 1
        self.algorithm = 'auto'

    def execute(self, dataset):
        X = dataset[0]
        X = StandardScaler().fit_transform(X)
        clf = KMeans( n_clusters            = self.n_clusters,
                      init                  = self.init,
                      n_init                = self.n_init,
                      max_iter              = self.max_iter,
                      tol                   = self.tol, 
                      precompute_distances  = self.precompute_distances,
                      verbose               = self.verbose,
                      random_state          = self.random_state,
                      copy_x                = self.copy_x,
                      #n_jobs                = self.n_jobs,
                      algorithm             = self.algorithm )
        clf.fit(X)
        y = clf.predict(X)

        labels = set(y)
        colors = ListedColormap([plt.get_cmap(name = "gist_ncar")(each)
            for each in np.linspace(0, 1, len(labels))])

        X0, X1 = X[:,0], X[:,1]
        plt.clf()
        plt.scatter(X0, X1, c=y, cmap=colors, s=20, edgecolors='k')
        plt.xlim(X0.min() - 0.5, X0.max() + 0.5)
        plt.ylim(X1.min() - 0.5, X1.max() + 0.5)
        plt.title('K-Means Clustering')
        plt.show()
