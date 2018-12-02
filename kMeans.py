'''
This file clusters document embeddings using a trained model given by "dv.model" using k-means
for k=20. To adjust the number of clusters, adjust numOfClusters in the main() function.
It prints the clustered articles to the console.
'''
from gensim.models.doc2vec import Doc2Vec
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans, kmeans_observer, kmeans_visualizer
from main import getDataSets

# this gets document embeddings based on article content
def getModelVectors(model, trainingDataSize):
    vecs = [];
    for i in range(trainingDataSize):
        # model.docvecs[str(i)] gets the ith document embedding in our dataset
        vecs.append(model.docvecs[str(i)])
    return vecs

def main(numOfClusters = 20):
    # initialize variables
    trainingData, trainingContent, testData, testContent = getDataSets()
    model = Doc2Vec.load("dv.model")
    vectors = getModelVectors(model, len(trainingData))

    # run kmeans and get clusters
    centers = kmeans_plusplus_initializer(vectors, numOfClusters).initialize();
    kmeans_instance = kmeans(vectors, centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()

    numOfClusters = len(clusters)

    trainingLabeled = [0 for _ in range(len(trainingData))]

    # print out all clusters
    for i in range(numOfClusters):
        print("Cluster " + str(i + 1) + " (" + str(len(clusters[i])) + " articles in this cluster): ")
        for j in range(len(clusters[i])):
            trainingLabeled[clusters[i][j]] = i
            print(trainingData[clusters[i][j]][0])
        print("") #just to seperate clusters when printing


if __name__== "__main__":
    main()
