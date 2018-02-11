import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
    
class RNNWithEmbeddings(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers, vocab_size):
        super(RNNWithEmbeddings, self).__init__()
        self.hidden = None
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers)

        self.dropout = nn.Dropout(0.5)
        
        # The linear layer that maps from hidden state space to tag space
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self.fc3 = nn.Linear(2*vocab_size, vocab_size)
        
    def init_hidden(self, batch_size = 1, use_cuda = True):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        tensor = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if use_cuda:
            tensor = tensor.cuda()
        return Variable(tensor)

    def forward(self, sequence):
        self.gru.flatten_parameters()
        
        seq_length, batch_size = sequence.size()
        embeds = self.embeddings(sequence)
        gru_out, self.hidden = self.gru(embeds, self.hidden)
        out = self.dropout(gru_out)
        fc1 = self.dropout(self.fc1(out.view(-1, self.hidden_size)))
        fc2 = self.dropout(self.fc2(out.view(-1, self.hidden_size)))
        fc3 = self.fc3(torch.cat((fc1, fc2), dim=1))
        return fc3.view(seq_length, batch_size, -1)

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()


    def forward(self, preds, targets, lengths):
        batch_size = len(lengths)
        preds = preds.permute(1,0,2)
        targets = targets.permute(1,0)

        loss = 0
        for pred, target, i in zip(preds, targets, lengths):
            pred = pred[:i]
            target = target[:i]
            loss += self.cross_entropy(pred, target)
            
        return loss / batch_size
