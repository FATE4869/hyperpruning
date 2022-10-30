import argparse
import os
import time
import torch

def ed(param_name, default=None):
    return os.environ.get(param_name, default)

class Args():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Network Pruning')
        # parser.add_argument("--local_rank", type=int, default=0)
        args = parser.parse_args()
        # general hyper-parameters
        use_cuda = torch.cuda.is_available()
        args.device = torch.device("cuda" if use_cuda else "cpu")

        # dataset
        args.data = '../dataset/PTB/penn/'  # location of the data corpus
        args.epochs = 100  # upper epoch limit
        args.batch_size = 20  # batch size
        args.eval_batch_size = 20  # evaluate batch size
        args.bptt = 35  # sequence length


        # model hyper-parameters
        args.model = 'LSTM'  # type of recurrent net (RHN, LSTM)
        args.emsize = 1500  # size of word embeddings
        args.nhid = 1500  # number of hidden units per layer
        args.nlayers = 2  # number of layers
        args.dropout = 0.65  # dropout applied to layers (0 = no dropout)
        args.dropouth = 0.25  # dropout for rnn hidden units (0 = no dropout)
        args.dropouti = 0.65  # dropout for input embedding layers (0 = no dropout)
        args.dropoute = 0.2  # dropout to remove words from embedding layer (0 = no dropout)
        args.tied = False
        args.couple = True

        # args.model = 'RHN'
        # args.emsize = 830
        # args.nhid = 830
        # args.nrecurrence_depth = 10
        # args.nlayers = 1
        # args.dropout = 0.65  # dropout applied to layers (0 = no dropout)
        # args.dropouth = 0.25  # dropout for rnn hidden units (0 = no dropout)
        # args.dropouti = 0.65  # dropout for input embedding layers (0 = no dropout)
        # args.dropoute = 0.2  # dropout to remove words from embedding layer (0 = no dropout)
        # args.tied = True
        # args.couple = True


        args.log_interval = 200
        # args.evaluate = '../models/lstm_PTB/___e2___00000.pt'
        args.evaluate = ''  # path to pre-trained model (default: none)
        randomhash = ''.join(str(time.time()).split('.'))
        args.save = randomhash + '.pt'
        args.sparse = True
        args.seed = 42
        args.log_interval = 200
        args.keep_train_from_path = None
        args.verbose = False

        # optimizer hyper-parameters
        args.lr = 15  # initial learning rate
        args.clip = 0.25  # gradient clipping
        args.optimizer = 'sgd'
        args.momentum = 0.9  # SGD momentum (default: 0.9)
        args.wdecay = 1.2e-6

        self.args = args
        self.sparse()
        # self.rigL()
        self.selfish_rnn()

    def sparse(self):
        self.args.fix = False
        self.args.sparse = True
        self.args.sparsity = 0.66
        self.args.density = 0.33
        self.selfish_rnn()
        self.args.update_frequency = 100


    def selfish_rnn(self):
        self.args.beta = 1  # beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)
        self.args.nonmono = 5
        self.args.death_rate = 0.8
        self.args.init = 'uniform'
        self.args.growth = 'random'
        self.args.death = 'magnitude'
        self.args.redistribution = 'none'
        self.args.death_rate = 0.5
        self.args.decay_schedule = 'cosine'
        self.args.dense_allocation = None

        # RigL
    def rigL(self):
        self.args.sparsity = 0.1
        self.args.delta = 100
        self.args.alpha = 0.3
        self.args.static_topo = False # if 1, use random sparisty topo and remain static, else 0
        self.args.grad_accumulation_n = 1

def main():
    args = Args().args
    print(args)
if __name__ == '__main__':
    main()