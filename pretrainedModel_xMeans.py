import csv
from gensim.models.doc2vec import Doc2Vec
from pyclustering.cluster.xmeans import xmeans, splitting_type
from nltk.tokenize import word_tokenize
from main import getDataSets

# this gets document embeddings based on article content
def getModelVectors(model, trainingContent):
    embeddings = []
    for article_content in trainingContent:
        tokenized_vec = word_tokenize(article_content.lower())
        embeddings.append(model.infer_vector(tokenized_vec))
    return embeddings


def main():
    # initialize variables
    trainingData, trainingContent, testData, testContent = getDataSets()
    model = Doc2Vec.load("wikiModel.bin")
    vectors = getModelVectors(model, trainingContent)

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
