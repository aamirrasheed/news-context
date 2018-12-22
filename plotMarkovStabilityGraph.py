import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
from pretrainedModelWikipedia import getDataSets

def getTitles(trainingData):
    titles = []
    for i in range(len(trainingData)):
        titles.append(trainingData[i][0])
    return titles

def printClusters(N, C, index):
    trainingData, trainingContent, testData, testContent = getDataSets()
    titles = getTitles(trainingData)
    numClusters = N[index]
    print("Number of clusters is " + str(numClusters))
    clusters = [ [] for x in range(numClusters) ]
    for article in range(len(trainingData)):
        cluster = int(C[article][index])
        clusters[cluster].append(article)

    for cluster in range(numClusters):
        print("Clusters in cluster #" + str(cluster))
        for article in clusters[cluster]:
            print(titles[article])


def showGraph(N, VI):
    t = np.linspace(-1, 0, num=401)
    t = [10**x for x in t]
    fig, ax1 = plt.subplots()
    ax1.plot(t, N, 'r')
    plt.xscale('log')
    ax1.set_xlabel('Markov Time')
    ax1.set_ylabel('Number of Clusters', color='r')
    ax1.tick_params('y', colors='r')

    ax2 = ax1.twinx()
    ax2.plot(t, VI, 'b')
    ax2.set_ylabel('Variation of Information', color='b')
    ax2.tick_params('y', colors='b')

    fig.tight_layout()
    plt.show()


def main():
    mat_contents = sio.loadmat('Variation.mat')
    VI = mat_contents['VI']
    VI = VI[0]

    mat_contents = sio.loadmat('Number of Communities.mat')
    N = mat_contents['N']
    N = N[0]

    mat_contents = sio.loadmat('Cluster Labels.mat')
    C = mat_contents['C']

    #notes: index = 159 has 308 clusters and the local min of variation of Information
    #notes: index = 168 has 105 clusters
    #notes: index = 175 has 25 clusters

    printClusters(N, C, 159)
    showGraph(N, VI)



if __name__== "__main__":
    main()
