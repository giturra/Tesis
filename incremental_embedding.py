from river.base.transformer import Transformer
from river.feature_extraction.vectorize import 
# import Transformer

class IncrementalWordEmbedding(Transformer, VectorizeMixin):

    def __init__(
        self,
        v_size,
        c_size,
        w_size
        normalize=True,
        on: str = None,
        strip_accents=True,
        lowercase=True,
        preprocessor: typing.Callable = None,
        tokenizer: typing.Callable = None,
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

