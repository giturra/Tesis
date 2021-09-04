from river.base.transformer import Transformer
from river.feature_extraction.vectorize import VectorizerMixin


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

    def transform_one(x):
        ...


wcm = WordContextMatrix(10, 5, 3)