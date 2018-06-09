import numpy as np
from matplotlib.colors import ListedColormap
from MethodWrapper import MethodWrapper
from sklearn.neural_network import MLPClassifier
from sklearn.datasets.samples_generator import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.contour import QuadContourSet
import matplotlib.pyplot as plt
import pylab
import sys

class MLPWrapper(MethodWrapper, name = "MLPClassifier"):
    """MLPClassifier wrapper"""

    def __init__(self):
        MethodWrapper.__init__(self)
        self.hidden_layer_sizes = (100,)
        self.activation = "relu"
        self.solver = "adam"
        self.alpha = 0.0001
        self.batch_size = "auto"
        self.learning_rate = "constant"
        self.learning_rate_init = 0.001
        self.power_t = 0.5
        self.max_iter = 200
        self.tol = 1e-4
        self.validation_fraction = 0.1

    def execute(self, dataset):
        file_name = "output.txt"

        X, y = dataset
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.validation_fraction)

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 0].max() + 0.5

        h = 0.2 #mesh step size
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        labels = set(y)
        colors = ListedColormap([plt.get_cmap(name = "rainbow")(each)
            for each in np.linspace(0, 1, len(labels))])

        prev_loss = 100
        prev_score = 0

        if((X.shape[1] < 3) & (self.animation_delay > 0)):
            warm_start = True
            max_iter = 1
            iter_count = 200
        else:
            warm_start = False
            max_iter = self.max_iter
            iter_count = 1

        sys.stdout = open(file_name, 'a')

        classifier = MLPClassifier(hidden_layer_sizes = self.hidden_layer_sizes,
                                   activation = self.activation,
                                   solver = self.solver,
                                   alpha = self.alpha,
                                   batch_size = self.batch_size,
                                   learning_rate = self.learning_rate,
                                   learning_rate_init = self.learning_rate_init,
                                   power_t = self.power_t,
                                   max_iter = max_iter,
                                   shuffle = self.shuffle,
                                   tol = self.tol,
                                   early_stopping = False,
                                   warm_start = warm_start,
                                   verbose = True)

        plt.show()
        plt.ion()
        plt.pause(1)
        plt.scatter(X_train[:,0], X_train[:, 1], c = y_train, cmap = colors,
                    edgecolors = 'k')
        plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test, cmap = colors, alpha = 0.6,
                    edgecolors = 'k')
        open(file_name, 'w').close() #clear file
        contour = None
        for it in range(0, iter_count):
            if not(contour is None):
                for coll in contour.collections:
                    coll.remove()
            classifier.fit(X_train, y_train)
            score = classifier.score(X_test, y_test)
            loss = classifier.loss_

            Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)
            contour = plt.contourf(xx, yy, Z, cmap = colors, edgecolors = 'k', alpha = 0.6)
            if(self.animation_delay > 0):
                plt.pause(self.animation_delay)
            plt.title("Iteration: " + str(it + 1) + " , loss: " + str(round(loss, 5)) + ", score:" + str(round(score, 5)))
            if((abs(score - prev_score) < self.tol) | (abs(loss - prev_loss) < self.tol)):
                break

        plt.ioff()
        sys.stdout = sys.__stdout__
