import csv
from gensim.models.doc2vec import Doc2Vec
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans, kmeans_observer, kmeans_visualizer

def getarticlecontent(input_file):
    data = []
    titles = []
    for idx, row in enumerate(input_file):
        titles.append(row["Title"])
        data.append(row["Content no HTML"])
    return data, titles

def getModelVectors(model):
    vecs = [];
    for i in range(0, 1100):
        vecs.append(model.docvecs[str(i)])
    return vecs


trainingDataSize = 1100
testDataSize = 100
numOfClusters = 25

input_file = csv.DictReader(open("Stories.csv"))
data, titles = getarticlecontent(input_file)

model = Doc2Vec.load("dv.model")

vectors = getModelVectors(model)
centers = kmeans_plusplus_initializer(vectors, numOfClusters).initialize();
kmeans_instance = kmeans(vectors, centers)
kmeans_instance.process()
clusters = kmeans_instance.get_clusters()

numOfClusters = len(clusters)

trainingLabeled = [0 for x in range(trainingDataSize)]
for i in range(numOfClusters):
    print("Cluster " + str(i + 1) + ": ")
    for j in range(len(clusters[i])):
        trainingLabeled[clusters[i][j]] = i
        print(titles[clusters[i][j]])


