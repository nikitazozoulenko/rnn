import argparse
from argparse import ArgumentError

import torch
from torch.autograd import Variable
import numpy as np

from data_feeder import Lang, DataFeeder
from network import RNNWithEmbeddings

def make_parser():
    parser = argparse.ArgumentParser(description="Random name generator from a GRU RNN model")
    parser.add_argument("--start_character", type=str, default = "?", help="To only generate names that start with a specific character")
    parser.add_argument("--num_names", type=int, default = 100, help="How many names to generate. Default 100")
    parser.add_argument("--randomness", type=int, default = 3, help="Argument for torch.topk() to get top k results from the RNN. Default 3, Range [1,57]")
    parser.add_argument("--use_cuda", type=bool, default = torch.cuda.is_available(),  help="To use cuda or not. Default torch.cuda.is_available()")
    return parser


def assert_args_are_correct(args, all_letters):
    if args.start_character not in all_letters and args.start_character != "?":
        raise ArgumentError(__spec__,
                            "'start_character' has to be one of the following: {}".format(
                                all_letters))

    if args.randomness not in range(1, len(all_letters)+1):
        raise ArgumentError(__spec__,
                            "Randomness variable has to be in range [1, 57]")

    
def generate_name(model, lang, start_char, randomness, use_cuda):
    if start_char == "?":
        start_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        start_idx = np.random.randint(len(start_chars))
        start_char = start_chars[start_idx]
        
    name = start_char
    tensor_idx = lang.word2index[name]
    inp = torch.LongTensor([[tensor_idx]])
    if use_cuda:
        inp = inp.cuda()
    inp = Variable(inp, volatile = True)
    model.hidden = model.init_hidden(batch_size = 1, use_cuda = use_cuda)
    while(len(name)<50):
        char_pred = model(inp)
        value, index = char_pred.topk(randomness)
        pick = np.random.randint(randomness)
        index = index[:,:,pick:pick+1]
        numpy_index = index.data.cpu().numpy()[0]
        
        if numpy_index == lang.EOS:
            break
        name += lang.index2word[numpy_index[0][0]]

        inp = torch.from_numpy(numpy_index)
        if use_cuda:
            inp = inp.cuda()
        inp = Variable(inp, volatile = True)
    return name


def main():
    lang = Lang()
    parser = make_parser()
    args = parser.parse_args()
    assert_args_are_correct(args, lang.all_letters)
    model = torch.load("savedir/name_generator_it40k.pt")
    if args.use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    for i in range(args.num_names):
        name = generate_name(model, lang, args.start_character, args.randomness, args.use_cuda)
        print(name)

        
if __name__ == "__main__":
    main()
        
    
