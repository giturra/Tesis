import pandas as pd
from river import stream
from river.linear_model import SoftmaxRegression
from mean_embedding import MeanWordEmbedding
from incremental_embedding import WordContextMatrix
from nltk import word_tokenize

X = pd.read_csv('https://raw.githubusercontent.com/dccuchile/CC6205/master/assignments/assignment_1/data/train/anger-train.txt', sep='\t', names=['id', 'tweet', 'class', 'sentiment_intensity'])

y = X.pop('sentiment_intensity')

data_stream = stream.iter_pandas(X, y)
wcm = WordContextMatrix(1000, 100, 3)
mwe = MeanWordEmbedding(wcm, word_tokenize)
clf = SoftmaxRegression()

for xi, yi in data_stream:
    we = mwe.transform_one(xi['tweet'])
    clf.learn_one(we, yi)
    print(clf.predict_one(we), yi)