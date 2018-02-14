import io
import glob
import unicodedata
import string
import os
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


class Lang():
    def __init__(self):
        self.EOS = 0
        self.SOS = 1
        self.EOS_string = "<EOS>"
        self.SOS_string = "<SOS>"
        
        with open('word2index.pkl', 'rb') as f:
            self.word2index = pickle.load(f)
        with open('index2word.pkl', 'rb') as f:
            self.index2word = pickle.load(f)
        with open('c_files_paths.pkl', 'rb') as f:
            self.c_files_paths = pickle.load(f)

        self.vocab_size = len(self.word2index)


    def init_c_paths(self):
        walk_dir = "/hdd/Data/LinuxSourceCode/linux/"
        c_files_paths = []
        for root, subdirs, files in os.walk(walk_dir):
            for filename in files:
                if filename.endswith(".c"):
                    c_files_paths += [os.path.join(root, filename)]
        return c_files_paths

    
    def init_dicts_from_scratch(self):
        seen = []
        real_c_paths = []
        for path in self.c_files_paths:
            with open(path) as f:
                try:
                    text = f.read()
                    real_c_paths += [path]
                    for ele in text:
                        if ele not in seen:
                            seen += [ele]
                            self.word2index[ele] = self.vocab_size
                            self.index2word[self.vocab_size] = ele
                            self.vocab_size += 1
                except Exception as e:
                    pass
                
        self.c_files_paths = real_c_paths

            
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
        self.lang = Lang()
        self.c_files_paths = self.lang.c_files_paths
    
    def read_lines(self, filename):
        lines = io.open(filename, encoding="utf-8").read().strip().split("\n")
        return [self.unicodeToAscii(line) for line in lines]

    
    def get_random_name(self):
        path = self.c_files_paths[np.random.randint(len(self.c_files_paths))]

        with open(path) as f:
            text = f.read()
            #text = [ele for ele in text]

        inp = [self.lang.SOS_string]
        target = []
        for ele in text:
            inp += [ele]
            target += [ele]
        target += [self.lang.EOS_string]
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
                inp += [self.lang.EOS_string]
                tar += [self.lang.EOS_string]

        input_indices = self.lang.seq2tensor(input_names).t()
        target_indices = self.lang.seq2tensor(target_names).t()
        return input_indices, target_indices, lengths

if __name__ == "__main__":
    data_feeder = DataFeeder()
    for i in range(100):
        input_indices, target_indices, lengths = data_feeder.get_batch(1)
        print(input_indices)
