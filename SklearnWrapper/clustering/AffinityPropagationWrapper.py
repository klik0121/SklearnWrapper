import numpy as np
from matplotlib.colors import ListedColormap
from MethodWrapper import MethodWrapper
from sklearn.cluster import AffinityPropagation
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.utils import as_float_array, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import euclidean_distances, pairwise_distances_argmin
import matplotlib.pyplot as plt
import pylab
import sys
from itertools import cycle


class AffinityPropagationWrapper(MethodWrapper, name = "Affinity propagation"):
    """Affinity propagation wrapper"""

    def __init__(self):
        MethodWrapper.__init__(self)
        self.damping = 0.5
        self.max_iter = 200
        self.convergence_iter = 15
        self.preference = -50

    def draw_result(self, A, it, n_samples, R, S, X):
        plt.clf()
        axes = plt.gca()
        axes.set_xlim([min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5])
        axes.set_ylim([min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5])

        cluster_centers_indices, labels = self.get_result(A, it, n_samples, R, S, X)
        if(not cluster_centers_indices is None):
            n_clusters = len(cluster_centers_indices)
            unique_labels = set(labels)
            colors = ListedColormap([plt.get_cmap(name = "rainbow")(each)
                for each in np.linspace(0, 1, len(unique_labels))])
            plt.scatter(X[:, 0], X[:, 1], c = labels, cmap=colors, edgecolors='k', s=40)
            plt.scatter(X[cluster_centers_indices][:, 0], X[cluster_centers_indices][:, 1],
                        c = labels[cluster_centers_indices], cmap=colors, edgecolors='k',
                        s=120)
            plt.title('Iteration ' + str(it) + ('. Estimated number of clusters: %d' % n_clusters))
            if(self.animation_delay > 0):
                plt.pause(self.animation_delay)

    def get_result(self, A, it, n_samples, R, S, X):
        I = np.where(np.diag(A + R) > 0)[0]
        K = I.size  # Identify exemplars

        if K > 0:
            c = np.argmax(S[:, I], axis=1)
            c[I] = np.arange(K)  # Identify clusters
            # Refine the final set of exemplars and clusters and return results
            for k in range(K):
                ii = np.where(c == k)[0]
                j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
                I[k] = ii[j]

            c = np.argmax(S[:, I], axis=1)
            c[I] = np.arange(K)
            labels = I[c]
            # Reduce labels to a sorted, gapless, list
            cluster_centers_indices = np.unique(labels)
            labels = np.searchsorted(cluster_centers_indices, labels)
        else:
            labels = np.empty((n_samples, 1))
            cluster_centers_indices = None
            labels.fill(np.nan)

        print("Iteration " + str(it) + "\n\r")
        print("Cluster centers:\n\r")
        if(cluster_centers_indices is None):
            print("None")
        else:
            print(str(X[cluster_centers_indices]))
        print("\n\r")
        print("Labels: ")
        if(cluster_centers_indices is None):
            print("None")
        else:
            print(str(labels))
        print("\n\r")
        return cluster_centers_indices, labels

    def execute(self, dataset):
        #temp code
        #should be replaced with custom source
        X, labels_true = dataset

        S = -euclidean_distances(X, squared=True)
        n_samples = S.shape[0]
        random_state = np.random.RandomState(0)
        S.flat[::(n_samples + 1)] = self.preference
        A = np.zeros((n_samples, n_samples))
        R = np.zeros((n_samples, n_samples))
        tmp = np.zeros((n_samples, n_samples))
        S += ((np.finfo(np.double).eps * S + np.finfo(np.double).tiny * 100) *
          random_state.randn(n_samples, n_samples))
        e = np.zeros((n_samples, self.convergence_iter))
        ind = np.arange(n_samples)
        plt.show()
        for it in range(self.max_iter):
            # tmp = A + S; compute responsibilities
            np.add(A, S, tmp)
            I = np.argmax(tmp, axis=1)
            Y = tmp[ind, I]  # np.max(A + S, axis=1)
            tmp[ind, I] = -np.inf
            Y2 = np.max(tmp, axis=1)

            # tmp = Rnew
            np.subtract(S, Y[:, None], tmp)
            tmp[ind, I] = S[ind, I] - Y2

            # Damping
            tmp *= 1 - self.damping
            R *= self.damping
            R += tmp

            # tmp = Rp; compute availabilities
            np.maximum(R, 0, tmp)
            tmp.flat[::n_samples + 1] = R.flat[::n_samples + 1]

            # tmp = -Anew
            tmp -= np.sum(tmp, axis=0)
            dA = np.diag(tmp).copy()
            tmp.clip(0, np.inf, tmp)
            tmp.flat[::n_samples + 1] = dA

            # Damping
            tmp *= 1 - self.damping
            A *= self.damping
            A -= tmp

            # Check for convergence
            E = (np.diag(A) + np.diag(R)) > 0
            e[:, it % self.convergence_iter] = E
            K = np.sum(E, axis=0)

            if(self.animation_delay > 0):
                self.draw_result(A, it, n_samples, R, S, X)
            else:
                self.get_result(A, it, n_samples, R, S, X)

            if it >= self.convergence_iter:
                se = np.sum(e, axis=1)
                unconverged = (np.sum((se == self.convergence_iter) + (se == 0))
                               != n_samples)
                if (not unconverged and (K > 0)) or (it == self.max_iter):
                    break

        self.draw_result(A, it, n_samples, R, S, X)
