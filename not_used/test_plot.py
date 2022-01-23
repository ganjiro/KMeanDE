from KMeansDE import *
from PSOClustering import *
from Plotter import *

n_clusters = 5
dataset_name = "datasets/bupa.data"
bupa_data = np.delete(np.loadtxt(dataset_name, delimiter=','), np.s_[2:7], axis=1)
X = bupa_data[:, 0]
Y = bupa_data[:, 1]

# XXX KMeansDE
km = KMeansDE(n_clusters=n_clusters, max_iter=5000, verbose=1, scaling=True, dataset_name=dataset_name)
km.fit_predict(bupa_data)
membership_KMeansDE = km.get_membership()
plotter_KMeansDE = Plotter(X, Y, membership_KMeansDE)

pso = PSClustering(n_clusters=n_clusters, max_iter=5000, verbose=1, scaling=True, dataset_name=dataset_name)
membership_pso = pso.fit_predict(bupa_data)
plotter_pso = Plotter(X, Y, membership_pso)

plotter_KMeansDE.plot()
plotter_pso.plot()
