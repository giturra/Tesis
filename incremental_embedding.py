import numpy as np
import operator
import re
from pprint import pprint
from river.base.transformer import Transformer
from river.feature_extraction.vectorize import VectorizerMixin
from river.utils import VectorDict
from storage import Vocabulary, Context, WordRep
from scipy.spatial.distance import cosine
from nltk import word_tokenize


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
        is_ppmi=True
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

        self.is_ppmi = is_ppmi

        self.vocabulary.add(WordRep('unk', self.c_size))

        self.f = 0
    
    def transform_one(self, x):
        x = x if x in self.vocabulary else 'unk'
        word_rep = self.vocabulary[x]
        contexts = word_rep.contexts.items()
        if self.is_ppmi:
            embedding = {
                context: max(np.log2(
                        (coocurence * self.d) / (word_rep.counter * self.vocabulary[context].counter) 
                    ), 0) for context, coocurence in contexts
            }
        else:
            embedding = dict(word_rep.contexts)
        return VectorDict(embedding)


    def learn_one(self, x, **kwargs):
        tokens = kwargs['tokens']
        i = tokens.index(x)
        self.d += 1
        if x not in self.vocabulary:
            self.vocabulary.add(WordRep(x, self.c_size))
        contexts = _get_contexts(i, self.w_size, tokens)
        focus_word = self.vocabulary[x]
        # if x in self.vocabulary:
        #     self.vocabulary[x].counter += 1
        for c in contexts:
            if c not in self.contexts:
                self.contexts.add(c)
            if c not in self.contexts and len(self.contexts) == self.c_size and focus_word.word == 'unk':
                focus_word.add_context('unk')
            elif c not in self.contexts:
                focus_word.add_context('unk')
            elif c in self.contexts:
                focus_word.add_context(c)
        # print(f"{focus_word.word} {self.transform_one(focus_word.word)}")
        return self
    
    def get_embedding(self, x):
        if x in self.vocabulary:
            word_rep = self.vocabulary[x]
            embedding = np.zeros(self.c_size, dtype=float)
            contexts = word_rep.contexts.items()
            if self.is_ppmi:
                for context, coocurence in contexts:
                    ind_c = self.contexts[context]
                    pmi = np.log2(
                        (coocurence * self.d) / (word_rep.counter * self.vocabulary[context].counter) 
                    )
                    embedding[ind_c] = max(0, pmi)
            else:
                for context, coocurence in contexts:
                    ind_c = self.contexts[context]
                    embedding[ind_c] = coocurence 
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


def _preprocessing_streps(preprocessing_steps, x):
    for step in preprocessing_steps:
        x = step(x)
    return x

def run(stream_data, model, on=None, tokenizer=None, lower_case=True):
    # think if we need a class for this part
    preprocessing_steps = []
    if on is not None:
        preprocessing_steps.append(operator.itemgetter(on))
    if lower_case:
        preprocessing_steps.append(str.lower)
    preprocessing_steps.append(
        (re.compile(r"(?u)\b\w\w+\b").findall if tokenizer is None else tokenizer)
    )
    for text, y in stream_data:
        tokens = _preprocessing_streps(preprocessing_steps, text)
        for w in tokens:
            model = model.learn_one(w, tokens=tokens)
    #print(model.contexts.values_storage)
    pprint(cosine(model.get_embedding('she'), model.get_embedding('he')))
    #print(model.vocabulary['hello'].contexts)
    #pprint(model.get_embedding('j'))    
