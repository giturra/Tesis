from collections import defaultdict


class BaseStorage:

    def __ini__(self, size):
        self.size = size
        self.counter = 0
        self.values_storage = defaultdict(int)
        self.values_word = self.values_storage.l

    def is_full(self):
        return self.counter == self.size
    
    def __contains__(self, key):
        return key in self.storage
    
    def __len__(self):
        return len(self.values_word)

    def __repr__(self):
        return self.values_word.__repr__()
    
    def __getitem__(self, word):
        return self.values_storage[word]