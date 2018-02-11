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
        self.EOS = 0
        self.vocab_size = 1
        self.word2index = {"<EOS>" : self.EOS}
        self.index2word = {self.EOS : "<EOS>"}
        self.all_letters = string.ascii_letters + " .,;'"

        self.__init_dicts()

        
    def __init_dicts(self):
        
        for i, char in enumerate(self.all_letters):
            self.word2index[char] = i+1
            self.index2word[i+1] = char
            self.vocab_size += 1

            
    def seq2tensor(self, sequences, use_cuda = True, volatile = False):
        # indices = [self.word2index[ele] for ele in sequence]
        indices = [[self.word2index[ele] for ele in sequence] for sequence in sequences]
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
        self.lang = Lang()
        self.all_letters = self.lang.all_letters
        
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

    
    def get_random_name(self):
        cat = np.random.randint(len(self.categories))
        index = np.random.randint(len(self.names_dict[self.categories[cat]]))

        name = self.names_dict[self.categories[cat]][index]
        name = [char for char in name]

        inp = [name[0]]
        target = []
        for ele in name[1:]:
            inp += [ele]
            target += [ele]
        target += ["<EOS>"]
        return inp, target


    def __get_batch_strings(self, batch_size):
        inputs = []
        targets = []
        lengths = []
        for _ in range(batch_size):
            inp, target = self.get_random_name()
            inputs += [inp]
            targets += [target]
            lengths += [len(inp)]
        
        return inputs, targets, lengths
    
    def get_batch(self, batch_size = 1):
        input_names, target_names, lengths = self.__get_batch_strings(batch_size)
        
        seq_length = max(lengths)
        for inp, tar in zip(input_names, target_names):
            for _ in range(len(inp), seq_length):
                inp += ["<EOS>"]
                tar += ["<EOS>"]

        input_indices = self.lang.seq2tensor(input_names).t()
        target_indices = self.lang.seq2tensor(target_names).t()
        return input_indices, target_indices, lengths


if __name__ == "__main__":
    data_feeder = DataFeeder()
    for i in range(1):
        inputs, targets, lengths = data_feeder.get_batch(3)
        print(inputs)
        print(targets)
        print(lengths)
        
