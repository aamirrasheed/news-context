from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import csv
import sys

csv.field_size_limit(sys.maxsize)
data = []

def readData(readKaggle=False):
    data = []
    with open("../data/stories.csv") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            data.append(row["Content no HTML"])
        print("Num Entries in stories.csv: " + str(line_count))
    if(readKaggle):
        with open("../data/train.csv") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            print(type(csv_reader))
            line_count = 0
            for row in csv_reader:
                if(row["label"] == '0'):
                    line_count += 1
                    data.append(row["text"])
            print("Num Entries in train.csv: " + str(line_count))
    return data
def readTitleData():
    data = []
    with open("../data/stories.csv") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            data.append(row["Title"])
        print("Num Entries in stories.csv: " + str(line_count))
    return data
def train(max_epochs = 100):
    data = readData(readKaggle=True)

    # creates a data store of TaggedDocuments for each data entry
    # each TaggedDocument has words, and a tag for which entry in the data table it was
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

    # initialize model for word training
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)

    model.build_vocab(tagged_data)

    # train model
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    model.save("d2v.model")
    print("Model Saved")

def inference(dataEntryToMatch = 1221):
    data = readTitleData()
    model = Doc2Vec.load("d2v.model")

    topn = 100

    similar_doc = model.docvecs.most_similar(str(dataEntryToMatch), topn = topn)
    print("Title to match: ", data[dataEntryToMatch])
    for i in range(topn):
        if(int(similar_doc[i][0]) < 1225):
            print("Closest titles:" , data[int(similar_doc[i][0])])
inference()