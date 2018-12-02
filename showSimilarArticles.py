'''
This is our baseline method. This loads the model trained from main.py and uses 
L2 distance norm to find the closest matching articles using their document embeddings.
By default, we run this on the first 10 articles of our dataset.
'''
import csv
from gensim.models.doc2vec import Doc2Vec
from main import getDataSets

def main(modelName="dv.model"):

	# initialize variables
    numberOfArticlesToShow = 10;
    trainingData, trainingContent, testData, testContent = getDataSets()
    model = Doc2Vec.load(modelName)

    # find closest documents using most_similar function and print out
    for index in range(1, numberOfArticlesToShow):
        print("Trying to match articles for: " + trainingData[index][0])
        similar_doc = model.docvecs.most_similar(str(index))
        for article in similar_doc:
            print("Matched with: " + trainingData[int(article[0])][0])

if __name__== "__main__":
    main()
