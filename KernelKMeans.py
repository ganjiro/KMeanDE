"""Kernel K-means"""

# Author: Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

import numpy as np
import pandas as pd  # For data management
import pylab
from sklearn.preprocessing import StandardScaler  # To transform the dataset


from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state


class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means

    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """

    def __init__(self, n_clusters=2, max_iter=50, tol=1e-4, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0, init='random', scaling=True, dataset_name="None"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose
        self.init = init
        self.scaling = scaling
        self.dataset_name = dataset_name
        self._print_parameters()


    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        self.kernel = "rbf" # TODO REMOVE
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        self._prepareDataset(X) # TODO CHECK che X è il mio dataset

        n_samples = X.shape[0]

        K = self._get_kernel(X)

        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        if self.init == 'random':  # TODO CHECK
            rs = check_random_state(self.random_state)
            self.labels_ = rs.randint(self.n_clusters, size=n_samples)
        else:
            self.labels_ = self.init

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print("Converged at iteration", it + 1)
                break

        self.X_fit_ = X

        return self

    def _prepareDataset(self, data):

        data = pd.DataFrame(data)

        if self.verbose:
            print("\nHead of dataset")
            print(data.head())

        if self.scaling:
            print("\nScaling dataset...")
            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(data)
            data = pd.DataFrame(scaled_array, columns=data.columns)
            if self.verbose:
                print("Head of scaled dataset")
                print(data.head())

        if self.verbose:
            print("\nShape of dataset: ", data.shape)

        self.data = data.to_numpy()

        print("\n******* Initialization *******")
        return


    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        # print("Clusters:")
        # print(dist.min(axis=1))
        return dist.argmin(axis=1)

    def _print_parameters(self):
        print("******* Starting KernelKMeans *******")
        print("Datset:", self.dataset_name)
        if self.scaling:
            print("Dataset will be scaled")
        else:
            print("Dataset will not be scaled")
        print("Number of clusters:", self.n_clusters)
        print("Tolerance:", self.tol)
        print("Fit maximum iteration limit:", self.max_iter)
        print()

if __name__ == '__main__':
    '''
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=1000, centers=5, random_state=0)
    km = KernelKMeans(n_clusters=5, max_iter=100, random_state=0, verbose=1)
    print(km.fit_predict(X)[:10])
    '''
    dataset_name = "bupa.data"
    bupa_data = np.loadtxt(dataset_name, delimiter=',')
    bupa_data = np.delete(bupa_data, 6, 1)

    kkm = KernelKMeans(n_clusters=5, max_iter=5000, random_state=0, verbose=1, scaling=True, dataset_name=dataset_name)
    membership = kkm.fit_predict(bupa_data)
    print("\n\nMembership vector:")
    print(membership)


