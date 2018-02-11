import torch
from torch.autograd import Variable
import numpy as np

from data_feeder import Lang, DataFeeder
from network import RNNWithEmbeddings

if __name__ == "__main__":
    model = torch.load("savedir/name_generator_it40k.pt")
    lang = Lang()
    batch_size = 1

    for i in range(20):
        name = ""
        inp = Variable(torch.cuda.LongTensor([[lang.SOS]]), volatile = True)
        model.hidden = model.init_hidden(batch_size)
        while(len(name)<100):
            char_pred = model(inp)
            value, index = char_pred.topk(1)
            pick = np.random.randint(1)
            index = index[:,:,pick:pick+1]
            numpy_index = index.data.cpu().numpy()[0]
            if numpy_index == lang.EOS:
                break
            name += lang.index2word[numpy_index[0][0]]

            inp = Variable(torch.from_numpy(numpy_index).cuda(), volatile = True)
        print(name)
        
    
