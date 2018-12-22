import matplotlib.pyplot as plt
import numpy as np
from classifier import getAndFormatData
from sklearn.manifold import TSNE


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