'''
import kMeansDE

class KMeansDE():
    def __init__(self, n_clusters=2, max_iter=5000, tol=1e-4, verbose=0, population_lenght=150, kmeans_max_iter=5000, scaling=True, dataset_name="None"):
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
'''