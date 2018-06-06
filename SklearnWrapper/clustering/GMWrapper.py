from MethodWrapper import MethodWrapper
from sklearn.mixture import GaussianMixture
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

class GMWrapper(MethodWrapper, name = "Gaussian Mixture"):
    """GaussianMixture wrapper"""

    def __init__(self):
        MethodWrapper.__init__(self)
        self.total_points = 300
        self.covariance_type = 'full'
        self.tol = 1e-3
        self.reg_covar = 1e-6
        self.max_iter = 100
        self.n_init = 1
        self.init_params = 'kmeans'
        self.warm_start = False

    def set_total_points(self, value:str):
        self.total_points = int(value)
    def set_covariance_type(self, value:str):
        self.covariance_type = value
    def set_tol(self, value:str):
        self.tol = float(value)
    def set_reg_covar(self, value:str):
        self.reg_covar = float(value)
    def set_max_iter(self, value:str):
        self.max_iter = int(value)
    def set_n_init(self, value:str):
        self.n_init = int(value)
    def set_init_params(self, value:str):
        self.init_params = value
    def set_warm_start(self, value:str):
        self.warm_start = bool(value)

    def execute(self):
        centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
        X = make_blobs(n_samples=self.total_points, centers=centers, cluster_std=0.5,
                                    random_state=0)[0]
        X = StandardScaler().fit_transform(X)

        clf = GaussianMixture(n_components=4)
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
        plt.title('Gaussian Mixture Clustering')
        plt.show()


