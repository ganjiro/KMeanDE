# dataset_name =
# "bupa" for full bupa dataset
# "bupa2" for small bupa dataset R2)

from KMeansDE import KMeansDE
from PSOClustering import *
from Plotter import Plotter


class Tester:
    def __init__(self, n_clusters=3, max_iter=5000, verbose=1, scaling=False, dataset_name="bupa"):
        self.X = None
        self.Y = None

        self.n_clusters = np.array(n_clusters)
        self.max_iter = max_iter
        self.verbose = verbose
        self.scaling = scaling
        self.dataset_name = dataset_name
        self.save_path = None
        self.dataset = self.load_dataset(self.dataset_name)

        self.km = None
        self.pso = None

        self.run()

    def load_dataset(self, dataset_name):
        print("Launching Tester\nSearching for the dataset...\n")
        if dataset_name == "bupa":
            data = np.loadtxt("datasets/bupa.data", delimiter=',')
            data = np.delete(data, 6, 1)
            self.save_path = "not_used/results/bupa_dataset/"
        elif dataset_name == "bupa2":
            self.save_path = "not_used/results/bupa_dataset_R2/"
            data = np.delete(np.loadtxt("datasets/bupa.data", delimiter=','), np.s_[2:7], axis=1)
            self.X = data[:, 0]
            self.Y = data[:, 1]
        else:
            self.save_path = "not_used/results/bupa_dataset/"
            print("DATASET NAME INVALID!\nLoading default dataset bupa...")
            data = np.loadtxt("datasets/bupa.data", delimiter=',')
            data = np.delete(data, 6, 1)
        ret = data
        print("Dataset found\n")
        return ret

    def run(self):
        for k in self.n_clusters:
            self.km = KMeansDE(n_clusters=k, max_iter=self.max_iter, verbose=self.verbose, scaling=self.scaling,
                               dataset_name=self.dataset_name, test=True)
            self.km.fit_predict(self.dataset)
            if self.X is not None:
                plotter_KMeansDE = Plotter(self.X, self.Y, self.km.labels_)
                plotter_KMeansDE.plot()

            time.sleep(1)
            self.pso = PSClustering(n_clusters=k, max_iter=self.max_iter, verbose=self.verbose, scaling=self.scaling,
                                    dataset_name=self.dataset_name, test=True)
            self.pso.fit_predict(self.dataset)
            if self.X is not None:
                plotter_PSO = Plotter(self.X, self.Y, self.pso.labels_)
                plotter_PSO.plot()
