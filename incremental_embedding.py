import numpy as np
from river.base.transformer import Transformer
from river.feature_extraction.vectorize import VectorizerMixin
from storage import Vocabulary, Context, WordRep
from scipy.spatial.distance import cosine


class IncrementalWordEmbedding(Transformer, VectorizerMixin):

    def __init__(
        self,
        v_size,
        c_size,
        w_size,
        normalize=True,
        on=None,
        strip_accents=True,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        ngram_range=(1, 1),
    ):

        super().__init__(
            on=on,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
        )
        self.v_size = v_size
        self.c_size = c_size
        self.w_size = w_size


class WordContextMatrix(IncrementalWordEmbedding):

    def __init__(
        self, 
        v_size, 
        c_size, 
        w_size, 
        normalize=True,
        on=None,
        strip_accents=True,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        ngram_range=(1, 1),
    ):
        super().__init__(
            v_size,
            c_size,
            w_size,
            on=on,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
        )
        self.vocabulary = Vocabulary(self.v_size)
        self.contexts = Context(self.c_size)
        self.d = 0
    
    def transform_one(self, x):
        return self.process_text(x)

    def learn_one(self, x):
        tokens = self.process_text(x)
        #print(tokens)
        for w in tokens:
            if w not in self.vocabulary:
                self.vocabulary.add(WordRep(w, self.c_size))
            self.d += 1
        for i, w in enumerate(tokens):
            contexts = _get_contexts(i, self.w_size, tokens)
            if w in self.vocabulary:
                self.vocabulary[w].counter += 1
            for c in contexts:
                if c not in self.contexts:
                    # if context full no add the word
                    self.contexts.add(c)
                if c in self.contexts:
                    self.vocabulary[w].add_context(c)
        return self
    
    def get_embedding(self, x):
        if x in self.vocabulary:
            word_rep = self.vocabulary[x]
            embedding = np.zeros(self.c_size, dtype=float)
            contexts = word_rep.contexts.items()
            for context, coocurence in contexts:
                ind_c = self.contexts[context]
                pmi = np.log2(
                    (coocurence * self.d) / (word_rep.counter * self.vocabulary[context].counter) 
                )
                embedding[ind_c] = max(0, pmi)
                # embedding[ind_c] = coocurence 
            return embedding
        return False


def _get_contexts(ind_word, w_size, tokens):
    # to do: agregar try para check que es posible obtener los elementos de los tokens
    slice_start = ind_word - w_size if (ind_word - w_size >= 0) else 0
    slice_end = len(tokens) if (ind_word + w_size + 1 >= len(tokens)) else ind_word + w_size + 1
    first_part = tokens[slice_start: ind_word]
    last_part = tokens[ind_word + 1: slice_end]
    contexts = tuple(first_part + last_part)
    return contexts


from river.datasets import SMSSpam

dataset = SMSSpam()

wcm = WordContextMatrix(10_000, 20, 3, on='body')

# question el vocab size debe ser siempre mayor o igual al context size?

for xi, y in dataset:
    wcm = wcm.learn_one(xi)
print(wcm.vocabulary['he'].contexts)
print(wcm.get_embedding('he'))
# print(len(wcm.contexts.values_storage.keys()))
print(len(wcm.vocabulary.values_storage.items()))
print(wcm.vocabulary.counter)
print(wcm.vocabulary.is_full())
#print(wcm.get_embedding('burger').shape)
# print('its' in wcm.vocabulary)
#print(cosine(wcm.get_embedding('sex'), wcm.get_embedding('bedroom')))
# print(wcm.vocabulary['until'].contexts)
# print(wcm.vocabulary.size)
# print(wcm.contexts.values_storage)
# print(wcm.vocabulary['go'].counter)