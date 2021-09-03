import abc
from collections import defaultdict


class BaseStorage:

    def __init__(self, size):
        self.size = size
        self.counter = 0
        self.values_storage = defaultdict(int)
        self.values_word = tuple(self.values_storage.keys())


    def is_full(self):
        return self.counter == self.size
    
    @abc.abstractmethod
    def add(self, word):
        ...

    def __contains__(self, key):
        return key in self.storage
    
    def __len__(self):
        return len(self.values_word)

    def __repr__(self):
        return self.values_word.__repr__()
    
    def __getitem__(self, word):
        return self.values_storage[word]



class Vocabulary(BaseStorage):

    def __init__(self, size):
        super().__init__(size)
    
    def add(self, word_rep):
        if not self.is_full():
            self.values_storage[word_rep] = word_rep
            self.counter += 1
            self.values_storage = tuple(self.values_storage.keys())


class Context(BaseStorage):
    ....


