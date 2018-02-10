import io
import glob
import unicodedata
import string

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


class Lang():
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.vocab_size = 0

        self.__init_dicts()

        
    def __init_dicts(self):
        all_letters = string.ascii_letters + " .,;'"
        for i, char in enumerate(all_letters):
            self.word2index[char] = i
            self.index2word[i] = char
            self.vocab_size += 1

            
    def seq2tensor(self, sequence, use_cuda = True, volatile = False):
        indices = [self.word2index[ele] for ele in sequence]
        tensor = torch.LongTensor(indices)
        if use_cuda:
            tensor = tensor.cuda()
        return Variable(tensor, volatile = volatile)

    
    def tensor2seq(self, sequence):
        numpy_seq = sequence.data.cpu().numpy()
        words = [self.index2word[ele] for ele in numpy_seq]
        return "".join(words)

    
class DataFeeder():
    def __init__(self):
        self.categories = []
        self.names_dict = {}
        self.all_letters = string.ascii_letters + " .,;'"
        
        self.__generate_names_dict()

        
    def __generate_names_dict(self):
        paths = glob.glob("data/names/*.txt")
        for filename in paths:
            category = filename.split("/")[-1].split(".")[0]
            self.categories += [category]
            lines = self.read_lines(filename)
            self.names_dict[category] = lines

            
    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters)

    
    def read_lines(self, filename):
        lines = io.open(filename, encoding="utf-8").read().strip().split("\n")
        return [self.unicodeToAscii(line) for line in lines]

    
    def get_batch(self):
        cat = np.random.randint(len(self.categories))
        index = np.random.randint(len(self.names_dict[self.categories[cat]]))

        return self.names_dict[self.categories[cat]][index], cat


if __name__ == "__main__":
    data_feeder = DataFeeder()
    lang = Lang()
    for i in range(1):
        name = data_feeder.get_random_name()
        indices = lang.seq2tensor(name)
        back_to_name = lang.tensor2seq(indices)
        print(name)
        print(indices)
        print(back_to_name)
        print(lang.vocab_size)
        
