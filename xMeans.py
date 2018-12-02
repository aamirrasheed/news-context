'''
This file clusters our document embeddings using the xmeans method from the pyclustering package and 
a trained model called "dv.model". It prints the clustered articles to the console.
'''
from gensim.models.doc2vec import Doc2Vec
from pyclustering.cluster.xmeans import xmeans, splitting_type
from main import getDataSets

# this gets document embeddings based on article content
def getModelVectors(model, trainingDataSize):
    vecs = [];
    for i in range(trainingDataSize):
        # model.docvecs[str(i)] gets the ith document embedding in our dataset
        vecs.append(model.docvecs[str(i)])
    return vecs


def main():
    # initialize variables
    trainingData, trainingContent, testData, testContent = getDataSets()
    model = Doc2Vec.load("dv.model")
    vectors = getModelVectors(model, len(trainingData))

    # run xmeans and get clusters
    xmeans_instance = xmeans(vectors)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    numOfClusters = len(clusters)

    # print out all clusters 
    trainingLabeled = [0 for _ in range(len(trainingData))]
    for i in range(numOfClusters):
        print("Cluster " + str(i + 1) + " (" + str(len(clusters[i])) + " articles in this cluster): ")
        for j in range(len(clusters[i])):
            trainingLabeled[clusters[i][j]] = i
            print(trainingData[clusters[i][j]][0])
        print("")  # just to seperate clusters when printing

if __name__== "__main__":
    main()