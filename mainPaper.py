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
from gensim.corpora.wikicorpus import WikiCorpus


class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c.decode("utf-8") for c in content], [title])

# runs our model training
def main(max_epochs=10):

    wiki = WikiCorpus("wiki_articles.xml.bz2")
    tagged_data = TaggedWikiDocument(wiki)

    # prepare model
    model = Doc2Vec(vector_size=300,
                min_count=20,
                dm=0,
                sample=0.001,
                negative=5,
                epochs=max_epochs,
                window=5)

    model.build_vocab(tagged_data)

    # train model
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)

    model.save("paper_inspired_model.model")

if __name__== "__main__":
    main()
