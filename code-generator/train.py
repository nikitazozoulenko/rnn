import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data_feeder import DataFeeder, Lang
from network import RNNWithEmbeddings, Loss
from utils import graph_losses


if __name__ == "__main__":
    data_feeder = DataFeeder()
    lang = data_feeder.lang
    
    model = RNNWithEmbeddings(64, 128, 2, lang.vocab_size).cuda()
    loss = Loss().cuda()

    learning_rate = 0.1
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum = 0.9, weight_decay=0.00001)

    losses = []
    batch_size = 16
    num_iterations = 40000
    for i in range(num_iterations):
        model.zero_grad()

        input_indices, target_indices, lengths = data_feeder.get_batch(batch_size)

        model.hidden = model.init_hidden(batch_size)
        preds = model(input_indices)

        total_loss = loss(preds, target_indices, lengths)
        
        total_loss.backward()
        optimizer.step()

        numpy_loss = total_loss.data.cpu().numpy()[0]
        losses += [numpy_loss]
        
        if i % 100 == 0:
            print(i)
            pass

        # decrease learning rate
        if i in [35000]:
            learning_rate /= 10
            print("updated learning rate: current lr:", learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        if i % 1000 == 0 and i != 0:
            torch.save(model, "savedir/name_generator_it"+str(i//1000)+"k.pt")
                
        # if i % 10000 == 0 and i != 0:
        #     torch.save(model, "savedir/facenet_"+version+"_it"+str(i//1000)+"k.pt")
            
    graph_losses(losses)
