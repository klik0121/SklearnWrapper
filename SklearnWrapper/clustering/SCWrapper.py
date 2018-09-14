from MethodWrapper import MethodWrapper
from sklearn.cluster.spectral import SpectralClustering
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from output_clustering import output

class SCWrapper(MethodWrapper, name = 'Spectral Clustering'):
    """SpectralClustering wrapper"""

    def __init__(self):
        MethodWrapper.__init__(self)
        self.n_clusters = 8
        self.eigen_solver = None
        self.random_state = None
        self.n_init = 10
        self.gamma = 1.0
        self.affinity = 'rbf'
        self.n_neighbors = 10
        self.eigen_tol = 0.0
        self.assign_labels = 'kmeans'
        self.degree = 3
        self.coef0 = 1
        self.n_jobs = 1

    def execute(self, dataset):
        X = dataset[0]
        y_true = dataset[1]
        X = StandardScaler().fit_transform(X)

        clf = SpectralClustering(n_clusters=self.n_clusters,
                                 eigen_solver=self.eigen_solver,
                                 random_state=self.random_state,
                                 n_init=self.n_init,
                                 gamma=self.gamma,
                                 affinity=self.affinity,
                                 n_neighbors=self.n_neighbors,
                                 eigen_tol=self.eigen_tol,
                                 assign_labels=self.assign_labels,
                                 degree=self.degree,
                                 coef0=self.coef0,
                                 n_jobs=self.n_jobs)
        y = clf.fit_predict(X)
        output(type(self).__name__, y, y_true, self.n_clusters)
        labels = set(y)
        colors = ListedColormap([plt.get_cmap(name = "gist_ncar")(each)
            for each in np.linspace(0, 1, len(labels))])

        X0, X1 = X[:,0], X[:,1]
        plt.clf()
        plt.scatter(X[:,0], X[:,1], c=y, cmap=colors, s=20, edgecolors='k')
        plt.xlim(X0.min() - 0.5, X0.max() + 0.5)
        plt.ylim(X1.min() - 0.5, X1.max() + 0.5)
        plt.title('Spectral Clustering')
        plt.show()
