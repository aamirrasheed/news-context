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

def getSimilarityMatrix(embeddings, trainingData):
    similaritiesMatrix = []
    for article1 in range(len(trainingData)):
        articleArr = []
        for article2 in range(len(trainingData)):
            similarity = 1 - spatial.distance.cosine(embeddings[article1], embeddings[article2])
            articleArr.append(similarity)
        similaritiesMatrix.append(articleArr)
    return similaritiesMatrix

    # Code that write the matrix to a text file
    with open('similarityMatrix.txt', 'w') as f:
        for arr in similaritiesMatrix:
            for item in arr[:-1]:
                f.write("%s," % item)
            f.write("%s\n" % arr[len(arr) -1])



def printMostSimilarArticles(n, embeddings, trainingData, showHistogram, dissimilarTitles, minDissimilarityPercentage):
    similaritiesArr = []
    for article1 in range(len(trainingData)):
        for article2 in range(len(trainingData)):
            similarity = 1 - spatial.distance.cosine(embeddings[article1], embeddings[article2])
            similaritiesArr.append(similarity)

    if (dissimilarTitles):
        print("Most similar two articles with dissimilar titles")
        indicesOfTopN = sorted(range(len(similaritiesArr)), key=lambda i: similaritiesArr[i])
        count = 0
        indexOfIndicesArr = len(trainingData) + 1
        while (count < n):
            indexOfSimilaritiesArr = indicesOfTopN[-indexOfIndicesArr]
            article1 = int(indexOfSimilaritiesArr) // int(len(trainingData))
            article1Title = trainingData[article1][0]
            article2 = int(indexOfSimilaritiesArr) % int(len(trainingData))
            article2Title = trainingData[article2][0]
            if (indexOfIndicesArr < len(similaritiesArr)):
                indexOfIndicesArr += 1
            else:
                print ("Could only find " + str(count + 1) + "articles with dissimilar titles!")
                break
            sameWords = set.intersection(set(article1Title.lower().split(" ")), set(article2Title.lower().split(" ")))
            if (len(sameWords) / min(len(article1Title), len(article2Title)) < 1 - minDissimilarityPercentage):
                print("#" + str(count + 1) + ":")
                print(article1Title)
                print(article2Title)
                print(similaritiesArr[indexOfSimilaritiesArr])
                print("")
                count += 1
    else:
        print("Most similar two articles")
        trainingDataLen = len(trainingData)
        indicesOfTopN = sorted(range(len(similaritiesArr)), key=lambda i: similaritiesArr[i])[-(n + trainingDataLen):]
        for i in range(trainingDataLen, trainingDataLen + n):
            index = indicesOfTopN[len(indicesOfTopN) - i - 1]
            print("#" + str(i - trainingDataLen + 1) + ":")
            article1 = int(index) // int(len(trainingData))
            print(trainingData[article1][0])
            article2 = int(index) % int(len(trainingData))
            print(trainingData[article2][0])
            print(similaritiesArr[index])
            print("")

    if (showHistogram):
        plt.hist(similaritiesArr, bins = 20)
        plt.title("Cosine Similarity between Every Two Articles")
        plt.show()

def printTopNSimilarities(indices, n, embeddings, trainingData):
    for index in indices:
        similaritiesArr = []
        for article in range(len(trainingData)):
            similarity = 1 - spatial.distance.cosine(embeddings[index], embeddings[article])
            similaritiesArr.append(similarity)

        print("Most similar articles to \"" + trainingData[index][0] + "\":")
        indicesOfTopN = sorted(range(len(similaritiesArr)), key=lambda i: similaritiesArr[i])[-(n + 1):]
        for i in range(1, 1 + n):
            index = indicesOfTopN[len(indicesOfTopN) - i - 1]
            print("#" + str(i) + ":")
            print(trainingData[index][0])
            print(similaritiesArr[index])
            print("")

def printAllArticles(trainingData):
    for i in range(len(trainingData)):
        print(str(i) + ": " + trainingData[i][0])

def main():
    # initialize variables
    trainingData, trainingContent, testData, testContent = getDataSets()
    model = Doc2Vec.load("wikiModel.bin")
    vectors = getModelVectors(model, trainingContent)

    printMostSimilarArticles(30, vectors, trainingData, False, True, 0.5)

    # matrix = getSimilarityMatrix(vectors, trainingData)

    # use the following function to figure out the index of the desired article
    # printAllArticles(trainingData)

    # printTopNSimilarities([581, 306, 759, 830] , 10, vectors, trainingData)




if __name__== "__main__":
    main()
