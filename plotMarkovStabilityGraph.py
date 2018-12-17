import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
from pretrainedModelWikipedia import getDataSets
from wordcloud import WordCloud

def getTitles(trainingData):
    titles = []
    for i in range(len(trainingData)):
        titles.append(trainingData[i][0])
    return titles

def getContent(trainingData):
    contents = []
    for i in range(len(trainingData)):
        contents.append(trainingData[i][1])
    return contents

def writeClustersForWordClouds(N, C, index):
    fileOutput = []
    trainingData, trainingContent, testData, testContent = getDataSets()
    titles = getTitles(trainingData)
    contents = getContent(trainingData)
    numClusters = N[index]
    print("Number of clusters is " + str(numClusters))
    clusters = [ [] for x in range(numClusters) ]
    for article in range(len(trainingData)):
        cluster = int(C[article][index])
        clusters[cluster].append(article)

    for cluster in range(numClusters):
        if len(clusters[cluster]) < 7 or len(clusters[cluster]) > 50:
            continue

        wordCloudString = ""

        clusterTitle = "Cluster #" + str(cluster) + ": " + str(len(clusters[cluster])) + " stories"
        fileOutput.append(clusterTitle)

        for article in clusters[cluster]:
            title = titles[article]
            content = contents[article]

            # delete all newlines in article content
            content = content.replace("\n", "")

            # delete sources part
            startSources = content.find("Sources:")
            content = content[:startSources]

            # delete the word said
            content = content.replace("said", "")

            # put together string
            string = title + ".\n" + content + "\n"

            # add to array
            fileOutput.append(string)

            # add to wordcloud string
            wordCloudString += string

        wordcloud = WordCloud(background_color="white").generate(wordCloudString)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig("wordClouds/" + clusterTitle, transparent = True, bbox_inches = 'tight', pad_inches = 0)

    with open('output/Clustered Articles.txt', 'w') as f:
        for item in fileOutput:
            f.write("%s\n" % item)


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
    ax2.set_ylabel('output/StabilityAlgorithmOutputL500/Variation of Information', color='b')
    ax2.tick_params('y', colors='b')

    fig.tight_layout()
    plt.show()

def findMinimums(data):
    indices = [10*x for x in range(40)]
    print(indices)
    mins = []
    for i in indices:
        if i > 0:
            minVal = 10000000
            minIndex = 0
            for j in range(i-5, i+5):
                if data[j] < minVal:
                    minVal = data[j]
                    minIndex = j
            mins.append((minVal, minIndex))
    print(mins)

def main():
    mat_contents = sio.loadmat('output/StabilityAlgorithmOutputL500/Variation.mat')
    VI = mat_contents['VI']
    VI = VI[0]

    mat_contents = sio.loadmat('output/StabilityAlgorithmOutputL500/Number of Communities.mat')
    N = mat_contents['N']
    N = N[0]

    mat_contents = sio.loadmat('output/StabilityAlgorithmOutputL500/Cluster Labels.mat')
    C = mat_contents['C']

    #notes: index = 159 has 308 clusters and the local min of variation of Information
    #notes: index = 168 has 105 clusters
    #notes: index = 175 has 25 clusters

    #findMinimums(VI)
    writeClustersForWordClouds(N, C, 235)
    #showGraph(N, VI)



if __name__== "__main__":
    main()
