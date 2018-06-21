import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from MethodWrapper import MethodWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import sys

class LRWrapper(MethodWrapper, name = "Linear Regression"):
    """LinearRegression wrapper"""

    def __init__(self):
        MethodWrapper.__init__(self)
        self.validation_fraction = 0.1

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

        classifier = LinearRegression()

        # Обучение классификатора
        classifier.fit(X_train, y_train)
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.clf()
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=colors, s=20, edgecolors='k')
        plt.scatter(X_test[:,0], X_test[:,1], alpha=0.5, c=y_test, cmap=colors, s=20, edgecolors='k')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        # Тестирование классификатора на новых данных
        score = classifier.score(X_test, y_test)
        plt.title('Ordinary Least Squares Classification\n score: ' + str(round(score, 5)))
        plt.show()
