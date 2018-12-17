'''
This file clusters our document embeddings using the xmeans method from the pyclustering package and
a trained model called "wikiModel.bin". It prints the clustered articles to the console and labels
the training data according to those clusters.
'''
import csv
import random
from gensim.models.doc2vec import Doc2Vec
from pyclustering.cluster.xmeans import xmeans, splitting_type
from nltk.tokenize import word_tokenize

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

def main():
    # initialize variables
    data, content = getDataSets()
    model = Doc2Vec.load("../wikiModel.bin")
    vectors = getModelVectors(model, content)

    # run xmeans and get clusters
    xmeans_instance = xmeans(vectors)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    numOfClusters = len(clusters)

    labeledData = getLabeledDatAndPrintClusters(numOfClusters, clusters, data)


if __name__== "__main__":
    main()
