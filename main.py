from incremental_embedding import run, WordContextMatrix
from river.datasets import SMSSpam

dataset = SMSSpam()
wcm = WordContextMatrix(10_000, 100, 3)

run(dataset, wcm, on='body') 