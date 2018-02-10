import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
    
class RNNWithEmbeddings(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers, vocab_size, n_categories):
        super(RNNWithEmbeddings, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.char_embeddings = nn.Embedding(vocab_size, embedding_size)

        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_size, n_categories)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.n_layers, 1, self.hidden_size).cuda()),
                Variable(torch.zeros(self.n_layers, 1, self.hidden_size).cuda()))

    def forward(self, sequence):
        embeds = self.char_embeddings(sequence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sequence), 1, -1), self.hidden)
        linear_out = self.linear(lstm_out.view(len(sequence), -1))
        linear_out = linear_out.view(len(sequence), 1, -1)
        return linear_out
