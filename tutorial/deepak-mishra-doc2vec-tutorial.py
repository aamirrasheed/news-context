from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import csv

data = []
def readData():
    data = []
    with open("../stories.csv") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            data.append(row["Title"])
            line_count += 1
    return data
def train(max_epochs = 10):
    data = readData()

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
    data = readData()
    model = Doc2Vec.load("d2v.model")

    topn = 10

    similar_doc = model.docvecs.most_similar(str(dataEntryToMatch), topn = topn)
    print("Title to match: ", data[dataEntryToMatch])
    for i in range(topn):
        print("Closest titles:" , data[int(similar_doc[i][0])])

inference()