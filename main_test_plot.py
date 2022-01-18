from KernelKMeans import *
from KMeansDE import *
from Plotter import *
import numpy as np
from PSOClustering import *

dataset_name = "bupa.data"
bupa_data = np.loadtxt(dataset_name, delimiter=',')
bupa_data = np.delete(bupa_data, 6, 1)

# XXX
bupa_data = np.delete(bupa_data, 5, 1)
bupa_data = np.delete(bupa_data, 4, 1)
bupa_data = np.delete(bupa_data, 3, 1)
# for _ in range(250):
#    data = np.delete(data, -1, 0)
bupa_data = np.delete(bupa_data, 2, 1)

# TODO FIX
tmpX = bupa_data
tmpY = bupa_data
# tmpX = np.delete(tmpX, 5, 1)
# tmpX = np.delete(tmpX, 4, 1)
# tmpX = np.delete(tmpX, 3, 1)
# tmpX = np.delete(tmpX, 2, 1)
# tmpX = np.delete(tmpX, 1, 1)
# tmpY = np.delete(tmpY, 5, 1)
# tmpY = np.delete(tmpY, 4, 1)
# tmpY = np.delete(tmpY, 3, 1)
tmpX = np.delete(tmpX, 1, 1)
tmpY = np.delete(tmpY, 0, 1)
X = tmpX
Y = tmpY
# print(X)
# print(Y)

# XXX KMeansDE
km = KMeansDE(n_clusters=4, verbose=1, scaling=True, dataset_name=dataset_name)
km.fit_predict(bupa_data)
membership_KMeansDE = km.get_membership()
plotter_KMeansDE = Plotter(X, Y, membership_KMeansDE)

# XXX KernelKMeans
kkm = KernelKMeans(n_clusters=4, max_iter=5000, random_state=0, verbose=1, scaling=True, dataset_name=dataset_name)
membership_KernelKMeans = kkm.fit_predict(bupa_data)
plotter_KernelKMeans = Plotter(X, Y, membership_KernelKMeans)

pso = PSClustering(n_clusters=4, verbose=1, scaling=True, dataset_name=dataset_name, max_iter=50, kmeans_max_iter=500)
membership_pso = pso.fit_predict(bupa_data)
plotter_pso = Plotter(X, Y, membership_pso)


print("\n\n*********** KMeansDE ***********")
print(membership_KMeansDE)
print("\n\n*********** KernelKMeans ***********")
print(membership_KernelKMeans)
print("\n\n*********** PSO ***********")
print(membership_pso)

# XXX PLOTTING
plotter_KMeansDE.plot()
plotter_KernelKMeans.plot()
plotter_pso.plot()
