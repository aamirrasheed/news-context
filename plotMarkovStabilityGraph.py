'''
This file reads the results of the .mat files outputted by the Matlba Markov
Stability code and includes functions that:
1. list the articles in each cluster
2. show a graph of variation of informaiton and number of clusters as functions
of Markov time
3. show histogram of cluster sizes (distribution of how many articles are in each cluster)
'''

import csv
import random
import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt

# returns title and article contents from our stories.csv file
def getDataFromFile(input_file):
    data = []
    for idx, row in enumerate(input_file):
        data.append((row["Title"], row["Content no HTML"]))
    return data

# given a data output from getDataFromFile(), this method will extract the article content
def getContent(data):
    content = []
    for pair in data:
        content.append(pair[1])
    return content

# this method is responsible for retrieving the data and dividing it into train and test sets
def getDataSets():
    dataSetSize = 1100

    # get dataset
    input_file = csv.DictReader(open("data/Stories.csv"))
    data = getDataFromFile(input_file)

    # shuffle data
    random.seed(1)
    random.shuffle(data)

    data = data[:dataSetSize]
    content = getContent(data)
    return data, content

# this method returns the titles of the data given as input
def getTitles(data):
    titles = []
    for i in range(len(data)):
        titles.append(data[i][0])
    return titles

# this method returns a 2D array of the clusters. The row n represents the
# nth cluster and the column m represents the mth element in that clusterself.
# The elements of the 2D array are the indices of the articles in the array "data".
def getClusters(data, N, C, index):
    titles = getTitles(data)
    numClusters = N[index]
    clusters = [ [] for x in range(numClusters) ]
    for article in range(len(data)):
        cluster = int(C[article][index])
        clusters[cluster].append(article)
    return clusters

# prints the number of clusters and the titles of the articles in each cluster
def printClusters(data, clusters):
    titles = getTitles(data)
    numClusters = len(clusters)
    print("Number of clusters is " + str(numClusters))
    for cluster in range(numClusters):
        print("Articles in cluster #" + str(cluster))
        for article in clusters[cluster]:
            print(titles[article])

# shows a histogram of the cluster sizes
def showHistogramOfClusterSizes(clusters):
    clusterSizes = []
    for cluster in clusters:
        clusterSizes.append(len(cluster))
    plt.hist(clusterSizes, bins = 10, range = (0, 30))
    plt.title("Size of Clusters")
    plt.xlabel("Size of Clusters")
    plt.ylabel("Number of Cluster with That Size")
    plt.show()

# shows a graph of the variation of information and the numbere of clusters with
# respect to Markov Time.
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
    mat_contents = sio.loadmat('Variation - L500.mat')
    VI = mat_contents['VI']
    VI = VI[0]

    mat_contents = sio.loadmat('Number of Communities - L500.mat')
    N = mat_contents['N']
    N = N[0]

    mat_contents = sio.loadmat('Cluster Labels - L500.mat')
    C = mat_contents['C']

    #notes: For 250 optimizations => index = 159 has 308 clusters and the local min of variation of Information
    #notes: For 250 optimizations =>index = 168 has 105 clusters
    #notes: For 250 optimizations => index = 175 has 25 clusters
    #notes: or 500 optimizations => index 235
    data, content = getDataSets()
    clusters = getClusters(data, N, C, 235)

    # Commented out examples of how some functions in this file would be used:
        # printClusters(trainingData, clusters)
        # showHistogramOfClusterSizes(clusters)
        # showGraph(N, VI)



if __name__== "__main__":
    main()
