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
        self.n_components = 1
        self.covariance_type = 'full'
        self.tol = 1e-3
        self.reg_covar = 1e-6
        self.max_iter = 100
        self.n_init = 1
        self.init_params = 'kmeans'
        self.warm_start = False

    def execute(self, dataset):
        X = dataset[0]
        X = StandardScaler().fit_transform(X)

        clf = GaussianMixture(n_components = self.n_components,
                              covariance_type = self.covariance_type,
                              tol = self.tol,
                              reg_covar = self.reg_covar,
                              max_iter = self.max_iter,
                              n_init = self.n_init,
                              init_params = self.init_params,
                              warm_start = self.warm_start)
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
