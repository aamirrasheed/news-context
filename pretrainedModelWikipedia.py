'''
This file includes all of the methods needed to find the most similar articles
(with dissimilar titles or regardless of titles), print similarity histograms
(which utalize cosine similarity), and the similarity matrix. The embeddings
for the articles are the embeddings inferred from the pretrained model "wikiModel.bin".
'''

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
    # get dataset
    input_file = csv.DictReader(open("data/Stories.csv"))
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

# this method returns the similarity matrix between every two articles
def getSimilarityMatrix(embeddings, data):
    similaritiesMatrix = []
    for article1 in range(len(data)):
        articleArr = []
        for article2 in range(len(data)):
            similarity = 1 - spatial.distance.cosine(embeddings[article1], embeddings[article2])
            articleArr.append(similarity)
        similaritiesMatrix.append(articleArr)

    # Code that write the matrix to a text file
    with open('output/similarityMatrix.txt', 'w') as f:
        for arr in similaritiesMatrix:
            for item in arr[:-1]:
                f.write("%s," % item)
            f.write("%s\n" % arr[len(arr) -1])

    return similaritiesMatrix

# this method prints the most similar articiles. Notes about parameters:
# n - the number of pairs of similar articles
# showHistogram - if true, the method should a histogram for the similarities
# dissimilarTitles - if true, the method prints titles that are dissimilar with at least the minDissimilarityPercentage
# minDissimilarityPercentage - the minimum percentage of words that need to be different between two titles. This is only used when dissimilarTitles is True.
def printMostSimilarArticles(n, embeddings, data, showHistogram, dissimilarTitles, minDissimilarityPercentage):
    similaritiesArr = []
    for article1 in range(len(data)):
        for article2 in range(len(data)):
            similarity = 1 - spatial.distance.cosine(embeddings[article1], embeddings[article2])
            similaritiesArr.append(similarity)

    # if dissimilarTitles is True, it prints only similar articles with dissimlar titles
    # (you can control the minimum dissimilarity percentage using the parameter
    # minDissimilarityPercentage). If dissimilarTitles is False, it prints the top n most
    # similar articles regardless of the similarity/dissimilarity of their titles.
    if (dissimilarTitles):
        print("Most similar two articles with dissimilar titles")
        indicesOfTopN = sorted(range(len(similaritiesArr)), key=lambda i: similaritiesArr[i])
        count = 0
        indexOfIndicesArr = len(data) + 1
        while (count < n):
            indexOfSimilaritiesArr = indicesOfTopN[-indexOfIndicesArr]
            article1 = int(indexOfSimilaritiesArr) // int(len(data))
            article1Title = data[article1][0]
            article2 = int(indexOfSimilaritiesArr) % int(len(data))
            article2Title = data[article2][0]
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
        dataLen = len(data)
        indicesOfTopN = sorted(range(len(similaritiesArr)), key=lambda i: similaritiesArr[i])[-(n + dataLen):]
        for i in range(dataLen, dataLen + n):
            index = indicesOfTopN[len(indicesOfTopN) - i - 1]
            print("#" + str(i - dataLen + 1) + ":")
            article1 = int(index) // int(len(data))
            print(data[article1][0])
            article2 = int(index) % int(len(data))
            print(data[article2][0])
            print(similaritiesArr[index])
            print("")

    # shows the histogram if showHistogram is True
    if (showHistogram):
        plt.hist(similaritiesArr, bins = 20)
        plt.title("Cosine Similarity between Every Two Articles (Wikipedia Model)")
        plt.xlabel("Value of Cosine Similarity")
        plt.ylabel("Number of Pairs of Articles")
        plt.show()

# prints the top n similar articles for the articles with indices corresponding to the
# values in the vector "indices".  The indices of the articles are determined by the indices of those
# articles in the list "embeddings".
def printTopNSimilarities(indices, n, embeddings, data):
    for j in range(len(indices)):
        index = indices[j]
        similaritiesArr = []
        for article in range(len(data)):
            similarity = 1 - spatial.distance.cosine(embeddings[index], embeddings[article])
            similaritiesArr.append(similarity)

        print(str(j + 1) + ". " + "Most similar articles to \"" + data[index][0] + "\":")
        indicesOfTopN = sorted(range(len(similaritiesArr)), key=lambda i: similaritiesArr[i])[-(n + 1):]
        for i in range(1, 1 + n):
            index = indicesOfTopN[len(indicesOfTopN) - i - 1]
            print("#" + str(i) + ":")
            print(data[index][0])
            print(similaritiesArr[index])
            print("")

# this method prints the indices of the articles along with the title of the article that each index corresponds to.
def printAllArticles(data):
    for i in range(len(data)):
        print(str(i) + ": " + data[i][0])

# this method prints the histogram of the word length of articles in the dataset
def printHistogramForArticleWordLength(content):
    articleWordLength = [len(article.split()) for article in content]
    plt.hist(articleWordLength, bins = 20)
    plt.title("Word Length of Articles in Dataset")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Articles")
    plt.show()

def main():
    # initialize variables
    data, content = getDataSets()
    model = Doc2Vec.load("wikiModel.bin")
    vectors = getModelVectors(model, content)

    #Commented out examples of to use some functions in this file:
        # printMostSimilarArticles(5, vectors, data, False, True, 0.5)
        # printTopNSimilarities([126] , 5, vectors, data)

    # you can use the following function to figure out the index of the desired article
        # printAllArticles(data)



if __name__== "__main__":
    main()
