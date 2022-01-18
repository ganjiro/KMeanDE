import math
import random

import numpy as np  # For data management
import pandas as pd  # For data management
from scipy.spatial import distance
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler  # To transform the dataset
from sklearn.utils._readonly_array_wrapper import ReadonlyArrayWrapper
from numpy import linalg as LA


class Particle:
    def __init__(self, cluster_centers, label, data):
        self.cluster_centers = cluster_centers
        self.label = label
        self.ss_distance = np.sum([distance.pdist([cluster_centers[i], data[label == i][j]], 'sqeuclidean') for i in
                                   range(len(cluster_centers)) for j in range(len(data[label == i]))])

        self.best_ss_distance = self.ss_distance
        self.best_position = self.cluster_centers
        self.velocity = [[np.random.uniform(low=0.0, high=0.1, size=1)[0] for _ in range(len(data[0]))] for _ in range(len(cluster_centers))]

    def update(self, cluster_centers, data, velocity):
        self.cluster_centers = cluster_centers
        self.velocity = velocity
        self.label = compute_labels(data, np.array([1.0 for _ in range(len(data))]),
                                    np.array([LA.norm(i)**2 for i in data]), self.cluster_centers) #todo check
        self.ss_distance = np.sum(
            [distance.pdist([cluster_centers[i], data[self.label == i][j]], 'sqeuclidean') for i in
             range(len(cluster_centers)) for j in range(len(data[self.label == i]))])

        if self.best_ss_distance > self.ss_distance:
            self.best_ss_distance = self.ss_distance
            self.best_position = self.cluster_centers


class PSClustering(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters=2, max_iter=5000, tol=1e-3,
                 verbose=0, population_lenght=100, kmeans_max_iter=5000, scaling=True,
                 dataset_name="None", cognitive=1.49, social=1.49, inertia=0.72):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.scaling = scaling
        self.dataset_name = dataset_name
        self.population_lenght = population_lenght
        self.verbose = verbose
        self.kmeans_max_iter = kmeans_max_iter
        self.population = None
        self.data = None
        self.cognitive = cognitive
        self.social = social
        self.inertia = inertia
        if self.verbose:
            self.print_parameters()

    def _prepareDataset(self, data):

        data = pd.DataFrame(data)

        if self.verbose:
            print("\nHead of dataset")
            print(data.head())

        if self.scaling:
            if self.verbose:
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

        if self.verbose:
            print("\n******* Initialization *******")
        return

    def _initializeS(self):
        self.population = []

        for i in range(self.population_lenght // 3):
            kmeans_model = KMeans(n_clusters=self.n_clusters, init='random', max_iter=100,
                                  n_init=1)

            kmeans_model.fit(self.data)
            centroids = kmeans_model.cluster_centers_
            labels = kmeans_model.labels_

            self.population.append(Particle(centroids, labels, self.data))

        while len(self.population) < self.population_lenght:
            centers = np.random.rand(self.n_clusters, len(self.data[0]))

            label = compute_labels(self.data, np.array([1.0 for _ in range(len(self.data))]),
                                   np.array([LA.norm(i) for i in self.data]), centers)

            self.population.append(Particle(centers, label, self.data))

        if self.verbose == 2:

            print("**** Initial population (initial centroids) ****\n")
            for i in range(len(self.population)):
                print("Cluster+" + str(i + 1) + "\n ", self.population[i].cluster_centers)

        return

    def _stopping_criterion(self, itr):

        if itr >= self.max_iter:
            if self.verbose:
                print("First stopping criterion, reach maximum iteration limit\n")
            return False

        adder = 0

        population_unique = self.unique()
        if len(population_unique) < len(self.population) and self.verbose:
            print("Number of redundant population elements: ", len(self.population) - len(population_unique))

        for k in range(len(population_unique)):
            adder += np.sum([np.abs(population_unique[k].ss_distance - population_unique[x].ss_distance) for x in
                             range(k + 1, len(population_unique))])

        if self.verbose:
            print("Objective tollerance:", self.tol, " Actual precision:", adder)

        if adder <= self.tol:
            if self.verbose:
                print("Second stopping criterion, population has converged\n")
            return False
        else:
            return True

    def _best_solution(self):

        best = self.population[0].ss_distance
        best_index = 0
        for i in range(self.population_lenght - 1):
            trial = self.population[0].ss_distance
            if trial < best:
                best = trial
                best_index = i
        if self.verbose:
            print("Best solution is the" + str(best_index + 1) + "-th element of the population,\ncentroids:\n",
                  self.population[best_index].cluster_centers)
        return self.population[best_index]

    def fit(self, data):
        self._prepareDataset(data)
        self._initializeS()

        count_iter = 0

        while self._stopping_criterion(count_iter):
            count_iter += 1
            if self.verbose:
                print("\n\n******* Iteration-" + str(count_iter) + " *******")

            global_best_position = self._get_global_best()
            for particle in self.population:
                new_velocitys = []
                new_centers = []
                mached_self_best, matched_global_best_position =self._matching(particle.cluster_centers, particle.best_position, global_best_position)
                for i in range(len(particle.cluster_centers)):
                    new_velocity = []
                    new_center = []

                    for j in range(len(particle.cluster_centers[i])):
                        r1 = np.random.uniform(low=0.0, high=1.0, size=1)[0]
                        r2 = np.random.uniform(low=0.0, high=1.0, size=1)[0]

                        new_velocity.append(self.inertia * particle.velocity[i][j] + self.cognitive * (
                                mached_self_best[i][j] - particle.cluster_centers[i][j]) * r1 + self.social * (
                                                    matched_global_best_position[i][j] - particle.cluster_centers[i][j]) * r2)

                        new_center.append(particle.cluster_centers[i][j] + new_velocity[j])

                    new_velocitys.append(new_velocity)
                    new_centers.append(new_center)

                particle.update(np.array(new_centers), self.data, new_velocitys)

        best_solution = self._best_solution()
        self.labels_ = best_solution.label
        self.cluster_centers_ = best_solution.cluster_centers
        return self

    def _matching(self, reference, parent_1, parent_2):
        # print("p1", parent_1)
        # print("p2", parent_2)
        distances1 = pairwise_distances(reference, parent_1)
        distances2 = pairwise_distances(reference, parent_2)
        parent_1 = self._reorder(distances1, parent_1)
        parent_2 = self._reorder(distances2, parent_2)
        # print("p1", parent_1)
        # print("p2", parent_2)
        return parent_1, parent_2

    @staticmethod
    def _reorder(distances, parent):
        temp = []
        init_set = set([i for i in range(len(parent))])
        for i in range(len(parent)):
            min_value = math.inf
            min_idx = -1
            for j in init_set:
                if distances[i, j] < min_value:
                    min_value = distances[i, j]
                    min_idx = j
            init_set.remove(min_idx)
            temp.append(parent[min_idx])
        return temp

    def print_parameters(self):

        print("******* Starting Particle Swarm Clustering *******")
        print("Datset:", self.dataset_name)
        if self.scaling:
            print("Dataset will be scaled")
        else:
            print("Dataset will not be scaled")
        print("Number of clusters:", self.n_clusters)
        print("Tolerance:", self.tol)
        print("Size of population:", self.population_lenght)
        print("Fit maximum iteration limit:", self.max_iter)
        print("KMeans subroutine maximum iteration limit:", self.kmeans_max_iter)
        print()

    def _get_global_best(self):
        best = self.population[0].best_ss_distance
        best_index = 0
        for i in range(self.population_lenght - 1):
            trial = self.population[0].ss_distance
            if trial < best:
                best = trial
                best_index = i

        return self.population[best_index].best_position

    def unique(self):
        set_list = [set(self.population[0].cluster_centers.flatten())]

        population_unique = [self.population[0]]

        for value in self.population[1:]:
            add = True
            for set_value in set_list:
                if set_value == set(value.cluster_centers.flatten()):
                    add = False
                    break

            if add:
                population_unique.append(value)
                set_list.append(set(value.cluster_centers.flatten()))

        return population_unique


def compute_labels(X, sample_weight, x_squared_norms, centers, n_threads=1):
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    labels = np.full(n_samples, -1, dtype=np.int32)
    weight_in_clusters = np.zeros(n_clusters, dtype=centers.dtype)
    center_shift = np.zeros_like(weight_in_clusters)

    _labels = lloyd_iter_chunked_dense

    X = ReadonlyArrayWrapper(X)

    _labels(
        X,
        sample_weight,
        x_squared_norms,
        centers,
        centers,
        weight_in_clusters,
        labels,
        center_shift,
        n_threads,
        update_centers=False,
    )
    return labels


if __name__ == '__main__':
    dataset_name = "bupa.data"
    bupa_data = np.loadtxt(dataset_name, delimiter=',')

    bupa_data = np.delete(bupa_data, 6, 1)
    bupa_data = np.delete(bupa_data, 5, 1)
    bupa_data = np.delete(bupa_data, 4, 1)
    bupa_data = np.delete(bupa_data, 3, 1)

    for _ in range(250):
        bupa_data = np.delete(bupa_data, -1, 0)

    bupa_data = np.delete(bupa_data, 2, 1)
    # data = np.delete(data, 1, 1)

    km = PSClustering(n_clusters=5, verbose=1, scaling=True, dataset_name=dataset_name)


    print(km.fit_predict(bupa_data))
    print(km.cluster_centers_)

# [2 3 3 2 3 3 3 3 3 3 3 3 3 3 3 2 3 3 2 2 3 3 3 3 2 3 3 3 2 3 3 3 3 2 3 0 3
#  3 2 3 2 3 2 2 3 3 3 3 3 3 3 3 0 3 3 3 2 3 1 3 3 3 3 3 3 3 3 3 1 3 3 3 3 3
#  3 3 2 1 3 1 2 3 2 3 0 3 3 3 3 2 3 3 1 3 3 3 3 2 3 1 3 3 3 2 2 3 2 2 3 2 2
#  2 3 2 0 3 3 3 2 3 1 2 2 3 2 3 1 1 3 3 1 1 0 0 3 3 3 1 1 1 1 1 1 1 1 1 1 0
#  1 1 0 1 1 3 1 1 0 1 2 1 1 1 1 1 3 1 0 0 0 1 2 1 1 1 0 1 1 1 0 1 0 1 0 1 1
#  0 0 1 0 0 3 3 2 2 3 2 3 3 3 2 3 3 2 2 2 2 3 2 2 3 2 3 2 2 3 2 3 2 3 2 2 3
#  3 3 3 3 2 2 2 3 3 3 0 3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 3 3 2 3 3 3 3
#  3 2 2 1 3 2 2 2 2 3 3 3 1 3 3 3 3 2 0 3 1 3 3 3 3 1 0 3 1 2 3 3 3 3 3 0 3
#  3 2 1 0 1 3 1 1 2 1 0 3 1 1 0 0 1 1 1 0 0 1 1 1 2 1 0 3 1 1 1 1 1 3 0 1 1
#  0 2 1 2 1 1 0 1 0 0 1 0]
# [[ 0.60910872  0.31838428  1.91561867  2.04705959  1.96473953  1.18116596]
#  [ 0.83581078 -0.16362428 -0.15769155 -0.09719618 -0.07364267  0.82510064]
#  [-0.46631036  1.26651864  0.00257058 -0.01427659 -0.02975714 -0.38803179]
#  [-0.34580962 -0.52556044 -0.36267129 -0.4147475  -0.40078857 -0.50416561]]



# 5 clust 2 dim
#pop 50

# [1 4 4 3 3 2 3 3 0 3 0 4 3 1 2 3 3 3 1 1 0 0 3 2 1 3 0 3 1 4 1 0 3 1 2 3 4
#  4 1 3 1 3 1 1 4 3 0 2 3 4 3 4 0 0 4 4 1 0 2 3 0 3 0 0 3 3 3 4 2 3 3 3 3 3
#  4 3 1 2 0 2 1 0 1 2 0 2 0 3 3 1 3 4 2 3 3]
# [[ 0.34385715 -1.01746757]
#  [-0.54190858  1.57003757]
#  [ 1.75904608 -0.12596902]
#  [-0.00390408  0.11166404]
#  [-1.25750258 -0.8421645 ]]

