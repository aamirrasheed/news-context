import csv
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def getDataFromFile(input_file):
    data = []
    titles = []
    for idx, row in enumerate(input_file):
        titles.append(row["Title"])
        data.append(row["Content no HTML"])
    return data, titles

def main():
    input_file = csv.DictReader(open("Stories.csv"))
    data, titles = getDataFromFile(input_file)

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

    max_epochs = 1000
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
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

    model.save("dv2.model")

    model= Doc2Vec.load("dv.model")
    # test_data = word_tokenize("Justice Kavanaugh recused in three high court cases so far".lower())
    # v1 = model.infer_vector(test_data)
    # print("V1_infer", v1)


    for index in range(1200, 1224):
        index_to_match = index
        print("Trying to match articles for: " + titles[index_to_match])
        similar_doc = model.docvecs.most_similar(str(index_to_match))
        for article in similar_doc:
            print("Matched with: " + titles[int(article[0])])


if __name__== "__main__":
    main()
