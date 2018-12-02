'''
This file can be run to train the doc2vec model. 
The model is saved in the same directory as "dv.model" and can be loaded later for inference.
To adjust the number of iterations we use for Doc2Vec, change the max_epochs variable in the main() function. 
We used this file to train various models with just titles, just articles, and with various numbers of iterations.
'''
import csv
import random
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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

# runs our model training
def main(max_epochs=1000, vec_size = 20, alpha = 0.025):

    # organize data
    trainingData, trainingContent, testData, testContent = getDataSets()
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(trainingContent)]

    # prepare model
    model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

    model.build_vocab(tagged_data)

    # train model
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    model.save("dv.model")

if __name__== "__main__":
    main()



