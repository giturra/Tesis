import numpy as np
from river.base.transformer import Transformer


class MeanWordEmbedding(Transformer):

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def transform_one(self, x):
        tokens = self.tokenizer(x)
        embeddings = np.zeros((len(tokens), self.model.c_size))
        for i, w in enumerate(tokens):
            self.model.learn_one(w, tokens=tokens)
            if w in self.model.vocabulary:
                embeddings[i, :] = self.model.transform_one(w)
            else:
                embeddings[i, :] = self.model.transform_one('unk')
        return {x: np.mean(embeddings, axis=0)}