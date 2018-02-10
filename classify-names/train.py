import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data_feeder import DataFeeder, Lang
from network import RNNWithEmbeddings
from utils import graph_losses

if __name__ == "__main__":
    data_feeder = DataFeeder()
    lang = Lang()
    model = RNNWithEmbeddings(64, 128, 2, lang.vocab_size, len(data_feeder.categories)).cuda()

    loss = nn.CrossEntropyLoss().cuda()

    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum = 0.9, weight_decay=0.00001)

    losses = []
    losses_ewma = []
    ewma = 2.8
    alpha = 0.90
    for i in range(20000):
        model.zero_grad()
        
        name, cat = data_feeder.get_batch()
        indices = lang.seq2tensor(name)
        target = Variable(torch.LongTensor([cat]).cuda())

        model.hidden = model.init_hidden()
        pred = model(indices)[-1]

        total_loss = loss(pred, target)
        
        total_loss.backward()
        optimizer.step()

        numpy_loss = total_loss.data.cpu().numpy()[0]
        losses += [numpy_loss]
        ewma = ewma*alpha + (1-alpha)*numpy_loss
        losses_ewma += [ewma]
        if i % 100 == 0:
            print(i)
            pass

        # decrease learning rate
        if i in [40000]:
            learning_rate /= 10
            print("updated learning rate: current lr:", learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
                
        # if i % 10000 == 0 and i != 0:
        #     torch.save(model, "savedir/facenet_"+version+"_it"+str(i//1000)+"k.pt")
            
    graph_losses(losses, losses_ewma)
