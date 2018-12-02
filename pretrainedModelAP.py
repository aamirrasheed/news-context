import csv
from gensim.models.doc2vec import Doc2Vec
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans, kmeans_observer, kmeans_visualizer
from nltk.tokenize import word_tokenize
from scipy import spatial
import random
from matplotlib import pyplot as plt


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
    # specify size of train set
    trainingDataSize = 1100

    # get dataset
    input_file = csv.DictReader(open("Stories.csv"))
    data = getDataFromFile(input_file)

    # shuffle data
    random.seed(1)
    random.shuffle(data)

    # set variables for each set and return
    trainingData = data[0:trainingDataSize]
    trainingContent = getContent(trainingData)
    testData = data[trainingDataSize::]
    testContent = getContent(testData)
    return trainingData, trainingContent, testData, testContent

# this gets document embeddings based on article content
def getModelVectors(model, trainingContent):
    embeddings = []
    for article_content in trainingContent:
        tokenized_vec = word_tokenize(article_content.lower())
        embeddings.append(model.infer_vector(tokenized_vec))
    return embeddings

def printMostSimilarArticles(n, embeddings, trainingData, show):
    similarities = []
    for article1 in range(len(trainingData)):
        for article2 in range(len(trainingData)):
            similarity = 1 - spatial.distance.cosine(embeddings[article1], embeddings[article2])
            similarities.append(similarity)

    print("Most similar two articles:")

    trainingDataLen = len(trainingData)
    indicesOfTopN = sorted(range(len(similarities)), key=lambda i: similarities[i])[-(n + trainingDataLen):]
    for i in range(trainingDataLen, trainingDataLen + n):
        index = indicesOfTopN[len(indicesOfTopN) - i - 1]
        print("#" + str(i - trainingDataLen + 1) + ":")
        article1 = int(index) // int(len(trainingData))
        print(trainingData[article1][0])
        article2 = int(index) % int(len(trainingData))
        print(trainingData[article2][0])
        print(similarities[index])
        print("")

    if (show):
        plt.hist(similarities, bins = 20)
        plt.title("Cosine Similarity between Every Two Articles")
        plt.show()


def main():
    # initialize variables
    trainingData, trainingContent, testData, testContent = getDataSets()
    model = Doc2Vec.load("APModel.bin")
    vectors = getModelVectors(model, trainingContent)

    printMostSimilarArticles(30, vectors, trainingData, False)



if __name__== "__main__":
    main()
