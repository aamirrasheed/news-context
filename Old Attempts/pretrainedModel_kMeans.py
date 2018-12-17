'''
This file clusters document embeddings inferred from the trained model "wikiModel.bin" using k-means
for k=20. To adjust the number of clusters, adjust numOfClusters in the main() function.
It prints the clustered articles to the console and labels the training data according to those clusters.
'''
import csv
from gensim.models.doc2vec import Doc2Vec
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans, kmeans_observer, kmeans_visualizer
from nltk.tokenize import word_tokenize
from scipy import spatial
import random

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
    # get dataset
    input_file = csv.DictReader(open("../data/Stories.csv"))
    data = getDataFromFile(input_file)

    # shuffle data
    random.seed(1)
    random.shuffle(data)

    content = getContent(data)
    return data, content

# this gets document embeddings based on article content
def getModelVectors(model, content):
    embeddings = []
    for article_content in content:
        tokenized_vec = word_tokenize(article_content.lower())
        embeddings.append(model.infer_vector(tokenized_vec))
    return embeddings

# this method prints out all clusters and returns the labeled training data
def getLabeledDatAndPrintClusters(numOfClusters, clusters, data):
    labeledData = [0 for _ in range(len(data))]
    for i in range(numOfClusters):
        print("Cluster " + str(i + 1) + " (" + str(len(clusters[i])) + " articles in this cluster): ")
        for j in range(len(clusters[i])):
            labeledData[clusters[i][j]] = i
            print(data[clusters[i][j]][0])
        print("") #just to seperate clusters when printing
    return labeledData

def main(numOfClusters = 20):
    # initialize variables
    data, content = getDataSets()
    model = Doc2Vec.load("../wikiModel.bin")
    embeddings = getModelVectors(model, content)

    # run kmeans and get clusters
    centers = kmeans_plusplus_initializer(embeddings, numOfClusters).initialize();
    kmeans_instance = kmeans(embeddings, centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()

    numOfClusters = len(clusters)

    trainingData = getLabeledDatAndPrintClusters(numOfClusters, clusters, data)



if __name__== "__main__":
    main()
