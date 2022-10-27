'''
This code will load trained model and calculate the LEs

gru model are stored in 'trials/gru/models/...'
each .pickle file contains 50 trials and each trial is trained with 20 epochs, the model and accuracy
after each epoch of training is saved. You can just load the model to perform the LEs calculation

lstm model:
'''
import argparse
import time
import math
import pickle
import torch
import torch.nn as nn
from models import Stacked_LSTM, RHN
from lyapunov import calc_LEs_an
import os
import data
from args import Args
# from train import batchify, get_batch, repackage_hidden
from dataloader import PBT_Dataloader

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(args, model, data_source, dataloader, model_type='lstm', LE_require=True):
    criterion = nn.CrossEntropyLoss()
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = dataloader.ntokens
    hidden = model.init_hidden(args.eval_batch_size)

    with torch.no_grad():
        start = time.time()
        for i, idx in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
            samples, targets = dataloader.get_batch(data_source, idx, args.bptt)
            output, hidden = model(samples, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(samples) * criterion(output_flat, targets).item()
            if LE_require:
                if i < 1:
                    if model_type == 'lstm':
                        h = torch.zeros(args.nlayers, samples.size(1), args.nhid).to(args.device)
                        c = torch.zeros(args.nlayers, samples.size(1), args.nhid).to(args.device)
                        emb = model.drop(model.encoder(samples))
                        params = (emb, (h, c))
                    # elif model_type == 'rhn':
                    #     emb = model.lockdrop(model.embedding(data), model.input_dp)
                    #     h = model.init_hidden(args.eval_batch_size)
                    #     params = (emb, h)
                    LEs, _ = calc_LEs_an(*params, model=model, rec_layer=model_type)
                    if i == 0:
                        LE_list = LEs
                    else:
                        LE_list = torch.cat([LE_list, LEs], dim=0)
        end = time.time()
        if LE_require:
            print(f"Time to calculate LE with {LE_list.shape[0]} samples: {end - start}")
            LEs_avg = torch.mean(LE_list, dim=0)
            return total_loss / (len(data_source)), LEs_avg
        else:
            print("LE is not required to calculate...")
            return total_loss / (len(data_source) - 1)

def cal_LEs_from_trained_model(args, model, val_data, test_data, dataloader, trial_num=None, epoch=100):
    # local path
    # path_models_des = f'../models/LSTM_PTB_full'
    # path_LEs_des = f'../LEs/LSTM_PTB_full'
    path_models_des = f'../models/LSTM_PTB_pruned'
    path_LEs_des = f'../LEs/LSTM_PTB_pruned'
    # path_models_des = f'../models/RHN_PTB_pruned/'
    # path_LEs_des = f'../LEs/RHN_PTB_pruned'
    # path_models_des = f'../models/RHN_PTB_full'
    # path_LEs_des = f'../LEs/RHN_PTB_full'

    # Load the saved model.
    for i in range(0, epoch):
        start = time.time()
        path_saved = f"{path_models_des}/___e{i}___{trial_num}.pt"
        # print(path_saved)
        if not os.path.exists(path_saved):
            continue
        else:
            with open(path_saved, 'rb') as f:
                saved = torch.load(path_saved)
                model.load_state_dict(saved)
                # Calculate LS from validation set
                val_loss, LEs_avg = evaluate(args, model, val_data, dataloader, model_type='lstm', LE_require=True)
                # val_loss = evaluate(args, model, val_data, dataloader, criterion, model_type='lstm', LE_require=False)
                # LEs_avg = 0

                # Just report the accuracy from testing set, do not need to calculate the LS
                test_loss = evaluate(args, model, test_data, dataloader, model_type='lstm', LE_require=False)
                print('=' * 89)
                print(f'| trial_num {trial_num} | At epoch {i} | val loss {val_loss:5.2f} | val ppl {math.exp(val_loss):8.2f} '
                      f'| test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f}')
                print('=' * 89)

                LEs_stats = {}
                LEs_stats['LEs'] = LEs_avg
                LEs_stats['current_loss'] = val_loss
                LEs_stats['current_perplexity'] = math.exp(val_loss)
                LEs_stats['test_loss'] = test_loss
                LEs_stats['test_perplexity'] = math.exp(test_loss)

                if not os.path.exists(path_LEs_des):
                    os.mkdir(path_LEs_des)
                pickle.dump(LEs_stats, open(f'{path_LEs_des}/___e{i}___{args.trial_num}.pickle', 'wb'))
        end = time.time()
        print(f"Time elpased: {end - start} s")

def LE_main(args):
    dataloader = PBT_Dataloader(data_path=args.data, batch_size=args.batch_size,
                                test_batch_size=args.eval_batch_size, bptt=args.bptt)
    # train_data = dataloader.train_data.to(args.device)
    val_data = dataloader.val_data.to(args.device)
    test_data = dataloader.test_data.to(args.device)
    ntokens = dataloader.ntokens
    model = Stacked_LSTM(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(args.device)
    cal_LEs_from_trained_model(args=args, model=model, val_data=val_data, test_data=test_data, dataloader=dataloader,
                               trial_num=args.trial_num, epoch=args.epochs)

# def LE_main_rhn(args):
#     corpus = data.Corpus(args.data)
#     ntokens = len(corpus.dictionary)
#     args.eval_batch_size = 20
#     val_data = batchify(corpus.valid, args.eval_batch_size).to(args.device)
#     test_data = batchify(corpus.test, args.eval_batch_size).to(args.device)
#     model = RHN(vocab_sz=ntokens, embedding_dim=args.emsize, hidden_dim=args.nhid,
#                 recurrence_depth=args.nrecurrence_depth, num_layers=args.nlayers, input_dp=args.dropouti,
#                 output_dp=args.dropout, hidden_dp=args.dropouth, embed_dp=args.dropoute,
#                 tie_weights=args.tied, couple=args.couple).to(args.device)
#     print(args)
#     cal_LEs_from_trained_model(args=args, model=model, val_data=val_data, test_data=test_data,corpus=corpus, trial_num=args.trial_num)
#     print(f'{torch.cuda.max_memory_allocated(args.device) / (10**9):.2f} out of 12 GB')

if __name__ == "__main__":
    args = Args().args
    args.data = '../dataset/PTB/penn/'
    args.save = '00000'
    args.epochs = 2
    args.eval_batch_size = 2


    # trial_nums = ['100002', '100005', '100006', '100007', '100012']
    trial_nums = ['00000']
    for trial_num in trial_nums:
        args.trial_num = trial_num
        LE_main(args)
        # LE_main_rhn(args)

