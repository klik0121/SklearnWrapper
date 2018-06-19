import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from MethodWrapper import MethodWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import matplotlib.pyplot as plt
import sys

class KNeighWrapper(MethodWrapper, name = 'The k-neighbors classification'):
    """The k-neighbors classification class"""

    def __init__(self):
        MethodWrapper.__init__(self)
        self.n_neighbors = 15
        self.validation_fraction = 0.1
        self.weights = 'uniform'

    def make_meshgrid(self, X, Y, h=0.2):
        x_min, x_max = X.min() - 0.5, X.max() + 0.5
        y_min, y_max = Y.min() - 0.5, Y.max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def execute(self, dataset):
        h = .02
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        X, y = dataset
        X = StandardScaler().fit_transform(X)
        sys.stdout = open(self.file_name, 'a')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.validation_fraction)
        X0,X1 = X[:, 0], X[:, 1]
        xx, yy = self.make_meshgrid(X0, X1)

        clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        open(self.file_name, 'w').close() #clear file
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        score = clf.score(X_test, y_test)
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (self.n_neighbors, self.weights) + ' score:' + str(round(score, 5)))
        plt.show()
        sys.stdout = sys.__stdout__
