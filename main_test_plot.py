from KMeansDE import *
from PSOClustering import *
from Plotter import *

dataset_name = "bupa.data"
bupa_data = np.delete(np.loadtxt(dataset_name, delimiter=','), np.s_[2:7], axis=1)

X = bupa_data[:, 0]
Y = bupa_data[:, 1]

# XXX KMeansDE
km = KMeansDE(n_clusters=4, verbose=1, scaling=True, dataset_name=dataset_name)
km.fit_predict(bupa_data)
membership_KMeansDE = km.get_membership()
plotter_KMeansDE = Plotter(X, Y, membership_KMeansDE)

# # XXX KernelKMeans
# kkm = KernelKMeans(n_clusters=4, max_iter=5000, random_state=0, verbose=1, scaling=True, dataset_name=dataset_name)
# membership_KernelKMeans = kkm.fit_predict(dataset)
# plotter_KernelKMeans = Plotter(X, Y, membership_KernelKMeans)

pso = PSClustering(n_clusters=4, verbose=1, scaling=True, dataset_name=dataset_name, max_iter=1000)
membership_pso = pso.fit_predict(bupa_data)
plotter_pso = Plotter(X, Y, membership_pso)

print("\n\n*********** KMeansDE ***********")
print(membership_KMeansDE)
# print("\n\n*********** KernelKMeans ***********")
# print(membership_KernelKMeans)
print("\n\n*********** PSO ***********")
print(membership_pso)

# XXX PLOTTING
plotter_KMeansDE.plot()
# plotter_KernelKMeans.plot()
plotter_pso.plot()
