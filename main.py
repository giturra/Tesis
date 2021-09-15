from incremental_embedding import run, WordContextMatrix
from river.datasets import SMSSpam
from nltk import word_tokenize

dataset = SMSSpam()
wcm = WordContextMatrix(10_000, 100, 3, is_ppmi=False)

run(dataset, wcm, on='body', tokenizer=word_tokenize)