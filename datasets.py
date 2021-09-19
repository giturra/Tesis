import pandas as pd
from river import stream

X = pd.read_csv('https://raw.githubusercontent.com/dccuchile/CC6205/master/assignments/assignment_1/data/train/anger-train.txt', sep='\t', names=['id', 'tweet', 'class', 'sentiment_intensity'])

y = X.pop('sentiment_intensity')

for xi, yi in stream.iter_pandas(X, y):
    print(xi, yi)