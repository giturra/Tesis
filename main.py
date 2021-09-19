from incremental_embedding import run, WordContextMatrix
from river.datasets import SMSSpam
from nltk import word_tokenize

dataset = SMSSpam()
wcm = WordContextMatrix(100, 5, 3, is_ppmi=False)

run(dataset, wcm, on='body', tokenizer=word_tokenize)
print(wcm.d)