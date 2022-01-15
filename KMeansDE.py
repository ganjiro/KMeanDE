import math
import random

import numpy as np  # For data management
import pandas as pd  # For data management
from scipy.spatial import distance
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans  # To instantiate, train and use model
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler  # To transform the dataset


class Solution:
    def __init__(self, cluster_centers, label, data):
        self.cluster_centers = cluster_centers
        self.label = label
        self.ss_distance = np.sum([distance.pdist([cluster_centers[i], data[label == i][j]], 'sqeuclidean') for i in
                                   range(len(cluster_centers)) for j in range(len(data[label == i]))])


class KMeansDE(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters=2, max_iter=5000, tol=1e-4,
                 verbose=0, population_lenght=150, kmeans_max_iter=5000, scaling=True, dataset_name="None"):
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
        self._print_parameters()

    def _compute_crossover(self, parent_1, parent_2, parent_3):  # S1 + F(S2-S3)
        f = random.uniform(0.5, 0.8)
        trial_centroids = []
        parent_1, parent_2 = self._matching(parent_3, parent_1, parent_2)
        for i in range(len(parent_1)):
            trial_centroids.append(parent_1[i] + f * (parent_2[i] - parent_3[i]))

        return trial_centroids

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

    def _initializeS(self):
        self.population = []
        for i in range(self.population_lenght):
            kmeans_model = KMeans(n_clusters=self.n_clusters, init='random', max_iter=1,
                                  n_init=1)

            kmeans_model.fit(self.data)
            centroids = kmeans_model.cluster_centers_
            labels = kmeans_model.labels_

            self.population.append(Solution(centroids, labels, self.data))

        if self.verbose == 2:
            print("**** Initial population (initial centroids) ****\n")
            for i in range(len(self.population)):
                print("Cluster+" + str(i + 1) + "\n ", self.population[i].cluster_centers)

        return

    def _stopping_criterion(self, consecutive_else, pick_success):
        if not pick_success:
            if self.verbose:
                print("Second stopping criterion, population has converged\n")
            return False

        if self.verbose:
            print("Consecutive iteration without improvement:", consecutive_else)

        if consecutive_else >= self.max_iter:
            if self.verbose:
                print("First stopping criterion, reach maximum iteration limit\n")
            return False

        adder = 0

        population_unique = self._unique()
        if len(population_unique) < len(self.population) and self.verbose:
            print("Number of redundant population elements: ", len(self.population) - len(population_unique))

        for k in range(len(population_unique)):
            adder += np.sum([np.abs(population_unique[k].ss_distance - population_unique[x].ss_distance) for x in
                             range(k + 1, len(population_unique))])

        if self.verbose:
            print("Objective tollerance:", self.tol, " Actual precision:", adder)

        if adder <= self.tol:
            if self.verbose:
                print("Second stopping criterion, population has converged with precision:", adder)
            return False
        else:
            return True

    def _pick_parents(self, initial_solution_id):
        count_itr = 0
        choices = random.sample(self.population, 3)
        success = True

        while not self._different_parents(self.population[initial_solution_id], choices) and count_itr < 1000:
            count_itr += 1
            choices = random.sample(self.population, 3)

        parent1 = choices[0].cluster_centers
        parent2 = choices[1].cluster_centers
        parent3 = choices[2].cluster_centers

        if not self._different_parents(self.population[initial_solution_id], choices):
            success = False

        return parent1, parent2, parent3, success

    def _different_parents(self, initial_solution, choices):

        initial_solution = set(initial_solution.cluster_centers.flatten())
        parent1 = set(choices[0].cluster_centers.flatten())
        parent2 = set(choices[1].cluster_centers.flatten())
        parent3 = set(choices[2].cluster_centers.flatten())

        return not (
                initial_solution == parent1 or initial_solution == parent2 or initial_solution == parent3 or parent1 == parent2 or parent2 == parent3)

    def _compute_kmeans(self, trial_centroids):
        trial_centroids = np.array(trial_centroids)

        kmeans_model = KMeans(n_clusters=self.n_clusters, init=trial_centroids, max_iter=self.kmeans_max_iter,
                              n_init=1, verbose=0)  # preparo l'algoritmo di kmeans

        kmeans_model.fit(self.data)
        centroids = kmeans_model.cluster_centers_
        labels = kmeans_model.labels_

        return Solution(centroids, labels, self.data)

    def _improvement(self, initial_solution, new_centroids):
        return new_centroids.ss_distance < initial_solution.ss_distance

    # TODO REMOVE INUTILE
    def get_membership(self):
        best = self.population[0].ss_distance
        best_index = 0
        for i in range(self.population_lenght - 1):
            trial = self.population[0].ss_distance
            if trial < best:
                best = trial
                best_index = i
        return self.population[best_index].label

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
                  self.population[best_index].cluster_centers, "\nmembership:\n",
                  self.population[best_index].label)
        return self.population[best_index]

    def fit(self, data):
        self._prepareDataset(data)
        self._initializeS()

        count_if = 0
        count_else = 0
        consecutive_else = 0
        count_iter = 0
        pick_success = True
        while self._stopping_criterion(consecutive_else, pick_success):
            count_iter += 1
            print("\n\n******* Iteration-" + str(count_iter) + " *******")

            for i in range(self.population_lenght):

                parent_1, parent_2, parent_3, pick_success = self._pick_parents(i)
                trial_centroids = self._compute_crossover(parent_1, parent_2, parent_3)

                new_centroids = self._compute_kmeans(trial_centroids)

                if self._improvement(self.population[i], new_centroids):
                    self.population[i] = new_centroids
                    count_if += 1
                    consecutive_else = 0
                else:
                    count_else += 1
                    consecutive_else += 1
        best_solution = self._best_solution()
        self.labels_ = best_solution.label
        self.cluster_centers_ = best_solution.cluster_centers
        return self

    def _matching(self, reference, parent_1, parent_2):

        distances1 = pairwise_distances(reference, parent_1)
        distances2 = pairwise_distances(reference, parent_2)
        parent_1 = self._reorder(distances1, parent_1)
        parent_2 = self._reorder(distances2, parent_2)

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

    def _unique(self):
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

    def _print_parameters(self):
        print("******* Starting KMeansDE *******")
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


if __name__ == '__main__':
    dataset_name = "bupa.data"
    bupa_data = np.loadtxt(dataset_name, delimiter=',')

    bupa_data = np.delete(bupa_data, 6, 1)
    # bupa_data = np.delete(bupa_data, 5, 1)
    # bupa_data = np.delete(bupa_data, 4, 1)
    # bupa_data = np.delete(bupa_data, 3, 1)

    # for _ in range(250):
    #     data = np.delete(data, -1, 0)

    # bupa_data = np.delete(bupa_data, 2, 1)
    # data = np.delete(data, 1, 1)

    km = KMeansDE(n_clusters=5, verbose=1, scaling=True, dataset_name=dataset_name)
    membership = km.fit_predict(bupa_data)
    print("\n\nMembership vector:")
    print(membership)

