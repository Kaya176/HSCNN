#import library
from nltk.tokenize import word_tokenize
from gensim.models import FastText
import pandas as pd

#load data

file = pd.read_csv("tmc_data.csv")
sentences = file["Text"]
corpus = []

#make corpus
for sent in sentences:
    corpus.append(word_tokenize(sent))

model = FastText(corpus,vector_size=100, workers=4, sg=1,window = 3)
model.train(corpus,total_examples=len(corpus),epochs= 10)

model.save("test.model")

#test
print(model.wv.most_similar("airpor",topn=5))
print(model.wv.most_similar("airpo",topn=5))
print(model.wv.most_similar("airtraffic",topn=5))
print(model.wv.most_similar("craft",topn=5))
print(model.wv.most_similar("acce",topn=5))