from sklearn.manifold import TSNE
from classifier import getAndFormatData
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def getDimReducedData():
    _, X, _, _ = getAndFormatData()
    X_embedded = TSNE(n_components=3, init='pca').fit_transform(X)
    return np.array(X_embedded)
def showData(data):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c='r')
    plt.show()

def main():
    data = getDimReducedData()
    showData(data)

main()