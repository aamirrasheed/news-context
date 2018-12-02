import csv
import random
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from pyclustering.cluster.xmeans import xmeans, splitting_type

trainingDataSize = 1100
testDataSize = 100

def getarticlecontent(input_file):
    data = []
    for idx, row in enumerate(input_file):
        data.append(row["Title"])
    return data

def getModelVectors(model):
    vecs = []
    for i in range(0, trainingDataSize):
        vecs.append(model.docvecs[str(i)])
    return vecs

def getContent(data):
    content = []
    for pair in data:
        content.append(pair[1])
    return content


def getLabeledData():
    input_file = csv.DictReader(open("Stories.csv"))
    data = getarticlecontent(input_file)
    random.seed(1)
    random.shuffle(data)
    trainingData = data[0:trainingDataSize]
    testData = data[trainingDataSize::]

    tagged_training_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(trainingData)]

    max_epochs = 6
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
            alpha=alpha,
            min_alpha=0.00025,
            min_count=1,
            dm=1)

    model.build_vocab(tagged_training_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_training_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    vectors = getModelVectors(model)
    xmeans_instance = xmeans(vectors)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    numOfClusters = len(clusters)

    trainingLabeled = [0 for x in range(trainingDataSize)]
    for i in range(numOfClusters):
        print("Cluster " + str(i + 1) + ": ")
        for j in range(len(clusters[i])):
            trainingLabeled[clusters[i][j]] = i
            print(trainingData[clusters[i][j]])
    print(numOfClusters)
    return numOfClusters, trainingData, trainingLabeled, testData

getLabeledData()