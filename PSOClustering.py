import math
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
        self.velocity = [[np.random.uniform(low=0.0, high=0.1, size=1)[0] for _ in range(len(data[0]))] for _ in
                         range(len(cluster_centers))]

    def update(self, cluster_centers, data, velocity):
        self.cluster_centers = cluster_centers
        self.velocity = velocity
        self.label = compute_labels(data, np.array([1.0 for _ in range(len(data))]),
                                    np.array([LA.norm(i) ** 2 for i in data]), self.cluster_centers)  # todo check
        self.ss_distance = np.sum(
            [distance.pdist([cluster_centers[i], data[self.label == i][j]], 'sqeuclidean') for i in
             range(len(cluster_centers)) for j in range(len(data[self.label == i]))])

        if self.best_ss_distance > self.ss_distance:
            self.best_ss_distance = self.ss_distance
            self.best_position = self.cluster_centers


class PSClustering(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters=3, max_iter=5000, tol=1e-3,
                 verbose=0, population_lenght=20, scaling=True,
                 dataset_name="None", cognitive=1.49, social=1.49, inertia=0.72, max_cons_iter=500):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.scaling = scaling
        self.dataset_name = dataset_name
        self.population_lenght = population_lenght
        self.verbose = verbose

        self.population = None
        self.data = None
        self.cognitive = cognitive
        self.social = social
        self.inertia = inertia
        self.max_cons_iter = max_cons_iter
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

    def _initializePopulation(self):
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

    def _stopping_criterion(self, itr, count_cons_iter):

        if itr >= self.max_iter:
            if self.verbose:
                print("First stopping criterion, reach maximum iteration limit\n")
            return False

        if count_cons_iter >= self.max_cons_iter:
            if self.verbose:
                print("Third stopping criterion, reach maximum consecutive iteration without improvements\n")
            return False
        if self.verbose:
            print("Consecutive iteration without improvements", count_cons_iter)

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

    def _best_solution(self, ending=False):

        best = self.population[0].ss_distance
        best_index = 0
        for i in range(self.population_lenght - 1):
            trial = self.population[0].ss_distance
            if trial < best:
                best = trial
                best_index = i

        return self.population[best_index]

    def fit(self, data):
        self._prepareDataset(data)
        self._initializePopulation()

        count_iter = 0
        count_cons_iter = 0
        best_ss_distance = math.inf

        while self._stopping_criterion(count_iter, count_cons_iter):
            count_iter += 1
            if self.verbose:
                print("\n\n******* Iteration-" + str(count_iter) + " *******")

            global_best_position = self._get_global_best()
            for particle in self.population:
                new_velocitys = []
                new_centers = []
                mached_self_best, matched_global_best_position = self._matching(particle.cluster_centers,
                                                                                particle.best_position,
                                                                                global_best_position)
                for i in range(len(particle.cluster_centers)):
                    new_velocity = []
                    new_center = []

                    for j in range(len(particle.cluster_centers[i])):
                        r1 = np.random.uniform(low=0.0, high=1.0, size=1)[0]
                        r2 = np.random.uniform(low=0.0, high=1.0, size=1)[0]

                        new_velocity.append(self.inertia * particle.velocity[i][j] + self.cognitive * (
                                mached_self_best[i][j] - particle.cluster_centers[i][j]) * r1 + self.social * (
                                                    matched_global_best_position[i][j] - particle.cluster_centers[i][
                                                j]) * r2)

                        new_center.append(particle.cluster_centers[i][j] + new_velocity[j])

                    new_velocitys.append(new_velocity)
                    new_centers.append(new_center)

                particle.update(np.array(new_centers), self.data, new_velocitys)

            best_solution = self._best_solution(ending=True)
            if best_solution.ss_distance < best_ss_distance:
                best_ss_distance = self._best_solution().ss_distance
                count_cons_iter = 0
            else:
                count_cons_iter += 1

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
    dataset = np.loadtxt(dataset_name, delimiter=',')

    dataset = np.delete(dataset, 6, 1)
    dataset = np.delete(dataset, 5, 1)
    dataset = np.delete(dataset, 4, 1)
    dataset = np.delete(dataset, 3, 1)
    for _ in range(250):
        dataset = np.delete(dataset, -1, 0)
    dataset = np.delete(dataset, 2, 1)
    # dataset = np.delete(dataset, 1, 1)

    pso = PSClustering(n_clusters=3, max_iter=5000, verbose=1, scaling=True, dataset_name=dataset_name)

    membership = pso.fit_predict(dataset)
    print("\n\nMembership vector:")
    print(membership)
    print("\n\nCentroids:")
    print(pso.cluster_centers_)

# [[0.06059718  1.68454103]
#  [1.49680408 - 0.21855616]
#  [0.02352025 - 1.02842233]
#  [0.09208318  0.06859595]
# [-1.30955163 - 0.2205655]]

# [[ 1.56154266  0.3099195 ]
#  [ 0.53947026 -0.94654934]
#  [-0.24302265  1.74391844]
#  [-1.19552677 -0.62618128]
#  [-0.04631541  0.12639116]]

# small data in 2d
'''
Membership vector:
[2 0 0 2 0 1 0 0 1 0 0 0 0 2 1 2 0 2 2 2 1 1 0 1 2 1 0 0 2 0 2 0 1 2 1 2 0
 0 2 0 2 0 2 2 0 0 1 1 2 0 2 0 1 1 0 0 2 0 1 0 0 2 0 0 0 2 0 0 1 0 0 0 0 2
 0 2 2 1 0 1 2 0 2 1 1 1 0 1 2 2 0 0 1 0 0]


Centroids:
[[-0.45057064 -0.5547412 ]
 [ 1.30837001 -0.26142896]
 [-0.27664336  1.17191425]] 
'''
