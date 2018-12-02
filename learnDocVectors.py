from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import csv
import sys

csv.field_size_limit(sys.maxsize)

def readData():
    data = []
    with open("data/stories.csv") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            data.append(row["Title"])
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    return data, tagged_data
def train(data, max_epochs = 300):
    # initialize model for word training
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)

    model.build_vocab(data)

    # train model
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    model.save("d2vTitlesJensDataset.model")
    print("Model Saved")

    return model

def getTrainedVectors(data, tagged_data, model):
    numVectors = model.corpus_count
    data_array = []

    flag = True

    for i in range(numVectors):
        vector = model.docvecs[i]
        title = data[i]
        data_array.append({"title":title, "vector":vector})

    return data_array

# data, tagged_data = readData()
# model = train(data, 10)
# data_array = getTrainedVectors(data, tagged_data, model)
# print(data_array[0])

