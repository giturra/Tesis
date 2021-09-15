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
        return key in self.values_storage
    
    def __len__(self):
        return len(self.values_word)

    def __repr__(self):
        return self.values_word.__repr__()
    
    def __getitem__(self, word):
        return self.values_storage[word]



class Vocabulary(BaseStorage):

    def __init__(self, v_size):
        super().__init__(v_size)
    
    def add(self, word_rep):
        if not self.is_full():
            self.values_storage[word_rep.word] = word_rep
            self.values_storage[word_rep.word].counter += 1
            self.counter += 1
            self.values_word = tuple(self.values_storage.keys())
    
    def __getitem__(self, word):
        if word in self.values_storage:
            return self.values_storage[word]
        return self.values_storage['unk']


class Context(BaseStorage):

    def __init__(self, c_size):
        super().__init__(c_size)
        self.counter = 1

    def add(self, word):
        if len(self) == 0:
                self.values_storage['unk'] = 0 
        if word not in self.values_storage and not self.is_full(): 
            self.values_storage[word] = self.counter
            self.counter += 1
            self.values_word = tuple(self.values_storage.keys())


class WordRep:

    def __init__(self, word, c_size):
        self.word = word
        self.c_size = c_size
        self.c_counter = 0
        # save counter between target and its contexts
        self.contexts = defaultdict(int)
        # for tracking number of tweets that appears
        #self.num_tweets = 0
        self.counter = 0
        
    def is_empty(self):
        return self.c_counter == 0

    def is_full(self):
        return self.c_counter == self.c_size

    def add_context(self, context):
        if context in self.contexts or self.is_full():
            if context in self.contexts:
                self.contexts[context] += 1
            else:
                self.contexts['unk'] += 1
        elif self.c_counter + 1 == self.c_size:
            self.contexts['unk'] += 1
            self.c_counter += 1
        else:
            self.contexts[context] += 1
            self.c_counter += 1
        # if not self.is_full() and context not in self.contexts:
        #     self.c_counter += 1
        #     self.contexts[context] += 1
        # elif context in self.contexts:
        #     self.contexts[context] += 1

    def __len__(self):
        return len(self.contexts.keys())

    def __repr__(self):
        return self.word

    def __getitem__(self, context):
        return self.contexts[context]
