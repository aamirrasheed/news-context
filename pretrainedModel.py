import csv
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def getarticlecontent(input_file):
    data = []
    for idx, row in enumerate(input_file):
        data.append(row["Title"])
    return data


input_file = csv.DictReader(open("Stories.csv"))
data = getarticlecontent(input_file)

model = Doc2Vec.load("d2v.model")

for index in range(1200, 1224):
    index_to_match = index
    print("Trying to match articles for: " + data[index_to_match])
    similar_doc = model.docvecs.most_similar(str(index_to_match))
    for article in similar_doc:
        print("Matched with: " + data[int(article[0])])
    print("")

