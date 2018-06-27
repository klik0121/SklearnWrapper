import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.kernel_ridge import KernelRidge
from MethodWrapper import MethodWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class KRRWrapper(MethodWrapper, name = 'Kernel Ridge Regression'):
    """Kernel Ridge Regression Wrapper class"""
    def __init__(self):
        MethodWrapper.__init__(self)
        self.validation_fraction = 0.1
        self.alpha = 0.8
        self.kernel = 'linear'
        self.gamma = None
        self.degree = 3
        self.coef0 = 1

    def make_meshgrid(self, X, Y, h=0.2):
        x_min, x_max = X.min() - 0.5, X.max() + 0.5
        y_min, y_max = Y.min() - 0.5, Y.max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def execute(self, dataset):
        # X - набор свойств, y - результат, зависящий от X
        X, y = dataset

        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.validation_fraction)
        X0,X1 = X[:, 0], X[:, 1]
        xx, yy = self.make_meshgrid(X0, X1)

        labels = set(y)
        colors = ListedColormap([plt.get_cmap(name = "rainbow")(each)
            for each in np.linspace(0, 1, len(labels))])

        classifier = KernelRidge(alpha = self.alpha,
                                 kernel = self.kernel,
                                 degree = self.degree,
                                 coef0 = self.coef0,
                                 gamma = self.gamma)

        classifier.fit(X_train, y_train)
        xxr,yyr = xx.ravel(), yy.ravel()
        cxy = np.c_[xx.ravel(), yy.ravel()]
        Z = classifier.predict(cxy)
        Z = Z.reshape(xx.shape)

        plt.clf()
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=colors, s=20, edgecolors='k')
        plt.scatter(X_test[:,0], X_test[:,1], alpha=0.5, c=y_test, cmap=colors, s=20, edgecolors='k')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        score = classifier.score(X_test, y_test)
        plt.title('Kernel Ridge Reduction Classification\n score: ' + str(round(score, 5)))
        plt.show()
