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

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha

model.save("dv.model")
print("Model Saved")

model= Doc2Vec.load("dv.model")
# test_data = word_tokenize("Justice Kavanaugh recused in three high court cases so far".lower())
# v1 = model.infer_vector(test_data)
# print("V1_infer", v1)


index_to_match = 1
print("Trying to match articles for: " + data[index_to_match])
similar_doc = model.docvecs.most_similar(str(index_to_match))
for article in similar_doc:
    print("Matched with: " + data[int(article[0])])






