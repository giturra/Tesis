import numpy as np
from river.base.transformer import Transformer
from river.utils import VectorDict


class MeanWordEmbedding(Transformer):

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    # def transform_one(self, x):
    #     tokens = self.tokenizer(x)
    #     embeddings = np.zeros((len(tokens), self.model.c_size))
    #     for i, w in enumerate(tokens):
    #         self.model.learn_one(w, tokens=tokens)
    #         if w in self.model.vocabulary:
    #             embeddings[i, :] = self.model.transform_one(w)
    #         else:
    #             embeddings[i, :] = self.model.transform_one('unk')
    #     return np.mean(embeddings, axis=0)

    def transform_one(self, x):
        tokens = self.tokenizer(x)
        n = len(tokens)
        embedding = VectorDict()
        for i, w in enumerate(tokens):
            self.model.learn_one(w, tokens=tokens)
            if w in self.model.vocabulary:
                embedding += self.model.transform_one(w)
            else:
                embedding += self.model.transform_one('unk')
        mean_embedding = embedding / n
        return mean_embedding

