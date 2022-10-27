import sys
import args
import time
import math
import torch
import torch.nn as nn
from models import Stacked_LSTM
from Sparse_ASGD import Sparse_ASGD
from dataloader import PBT_Dataloader
from sparse_rnn_core import Masking, CosineDecay


def model_save(model, fn):
    torch.save(model.state_dict(), fn)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train_main(args):
    dataloader = PBT_Dataloader(data_path=args.data, batch_size=args.batch_size,
                                test_batch_size=args.eval_batch_size, bptt=args.bptt)
    model = Stacked_LSTM(args.model, dataloader.ntokens, args.emsize,
                         args.nhid, args.nlayers, args.dropout, args.tied).to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    mask = None
    if args.sparse:
        full_training_epoch = 100
        decay = CosineDecay(args.death_rate, full_training_epoch * len(dataloader.train_data) // args.bptt)
        mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death,
                       death_rate_decay=decay, growth_mode=args.growth,
                       redistribution_mode=args.redistribution, model=args.model, args=args)
        mask.add_module(model, density=args.density, sparse_init=args.init)

    torch.manual_seed(args.seed)
    ###############################################################################
    # Load data
    ###############################################################################
    train_data = dataloader.train_data.to(args.device)
    val_data = dataloader.val_data.to(args.device)
    test_data = dataloader.test_data.to(args.device)
    ntokens = dataloader.ntokens
    criterion = nn.CrossEntropyLoss()
    ###############################################################################
    # Train and evaluate code
    ###############################################################################
    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        if args.model != 'Transformer':
            hidden = model.init_hidden(args.eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                samples, targets = dataloader.get_batch(data_source, i, bptt=args.bptt)
                output, hidden = model(samples, hidden)
                hidden = repackage_hidden(hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(samples) * criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)

    def train(mask=None, epoch=0):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.
        start_time = time.time()
        if args.model != 'Transformer':
            hidden = model.init_hidden(args.batch_size)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            samples, targets = dataloader.get_batch(train_data, i, args.bptt)
            optimizer.zero_grad()
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            output, hidden = model(samples, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if mask is None:
                optimizer.step()
            else:
                mask.step()
            total_loss += loss.item()
            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch+1, batch, len(train_data) // args.bptt, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
                sys.stdout.flush()

    ###############################################################################
    # Evaluating
    ###############################################################################
    if args.evaluate:
        print("=> loading checkpoint '{}'".format(args.evaluate))
        model.load_state_dict(torch.load(args.evaluate))
        print('=> testing...')
        test_loss = evaluate(test_data)
        print('=' * 89)
        print('| Final test | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)
        sys.stdout.flush()
        return test_loss

    ###############################################################################
    # Training
    ###############################################################################
    # else:
    lr = args.lr
    best_val_loss = []
    best_test_loss = []
    stored_loss = 100000000
    starting_epoch = 0
    save_path = f'../models/LSTM_PTB_pruned/___e{starting_epoch}___{args.save}.pt'
    model_save(model, save_path)
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        # Loop over epochs.
        for epoch in range(starting_epoch, starting_epoch + args.epochs):
            epoch_start_time = time.time()
            train(mask, epoch=epoch)

            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()
                val_loss2 = evaluate(val_data)
                test_loss2 = evaluate(test_data)
                print('-' * 89)
                print(f'| end of epoch {epoch+1:3d} | time: {(time.time() - epoch_start_time):5.2f}s | '
                      f'valid loss {val_loss2:5.2f} | valid ppl {math.exp(val_loss2):8.2f} | '
                      f'valid bpc {val_loss2 / math.log(2):8.3f} '
                      f'| test loss {test_loss2:5.2f} | test ppl {math.exp(test_loss2):8.2f} | '
                      f'test bpc {test_loss2 / math.log(2):8.3f} ')
                print('-' * 89)

                if val_loss2 < stored_loss:
                    # if epoch > 80:
                    save_path = f'../models/LSTM_PTB_pruned/___e{epoch+1}___{args.save}.pt'
                    model_save(model, save_path)
                    print('Saving Averaged!')
                    stored_loss = val_loss2
                    stored_test_loss = test_loss

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

                if args.sparse and epoch < args.epochs + 1:
                    mask.at_end_of_epoch(epoch)

            else:
                val_loss = evaluate(val_data)
                test_loss = evaluate(test_data)
                print('-' * 89)
                print(f'| end of epoch {epoch+1:3d} | time: {(time.time() - epoch_start_time):5.2f}s | '
                      f'valid loss {val_loss:5.2f} | valid ppl {math.exp(val_loss):8.2f} | '
                      f'valid bpc {val_loss / math.log(2):8.3f} '
                      f'| test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f} | '
                      f'test bpc {test_loss / math.log(2):8.3f} ')
                print('-' * 89)

                if val_loss < stored_loss:
                    save_path = f'../models/LSTM_PTB_pruned/___e{epoch+1}___{args.save}.pt'
                    model_save(model, save_path)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss
                    stored_test_loss = test_loss

                if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                        len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                    print('Switching to ASGD')
                    optimizer = Sparse_ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                    mask.optimizer = optimizer
                    mask.init_optimizer_mask()

                if args.sparse and 't0' not in optimizer.param_groups[0]:
                    mask.at_end_of_epoch(epoch)

                best_val_loss.append(val_loss)

            print(f"PROGRESS: ({epoch+1} / {starting_epoch + args.epochs}) = {(epoch+1) / (starting_epoch + args.epochs) * 100}%")






            # val_loss = evaluate(val_data)
            # test_loss = evaluate(test_data)
            # print('-' * 89)
            # print(f'| end of epoch {epoch+1:3d} | time: {(time.time() - epoch_start_time):5.2f}s | '
            #       f'valid loss {val_loss:5.2f} | valid ppl {math.exp(val_loss):8.2f} | '
            #       f'valid bpc {val_loss / math.log(2):8.3f} '
            #       f'| test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f} | '
            #       f'test bpc {test_loss / math.log(2):8.3f} ')
            # print('-' * 89)
            #
            # if val_loss < stored_loss:
            #     # if epoch > 80:
            #     save_path = f'../models/LSTM_PTB_pruned/___e{epoch+1}___{args.save}.pt'
            #     model_save(model, save_path)
            #     stored_loss = val_loss
            #     stored_test_loss = test_loss
            #     best_val_loss.append(val_loss)
            #     best_test_loss.append(test_loss)
            #
            # # Using ASGD
            # if 't0' in optimizer.param_groups[0]:
            #     print('Saving Averaged!')
            #     tmp = {}
            #     for prm in model.parameters():
            #         tmp[prm] = prm.data.clone()
            #         prm.data = optimizer.state[prm]['ax'].clone()
            #     for prm in model.parameters():
            #         prm.data = tmp[prm].clone()
            #     if args.sparse and epoch < args.epochs + 1:
            #         mask.at_end_of_epoch(epoch)
            # # Using SGD
            # else:
            #     if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
            #             len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
            #         print('Switching to ASGD')
            #         optimizer = Sparse_ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            #         mask.optimizer = optimizer
            #         mask.init_optimizer_mask()
            #     if args.sparse and 't0' not in optimizer.param_groups[0]:
            #         mask.at_end_of_epoch(epoch)
            # print(f"PROGRESS: ({epoch+1} / {starting_epoch + args.epochs}) = {(epoch+1) / (starting_epoch + args.epochs) * 100}%\n")
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    return stored_loss, stored_test_loss


if __name__ == '__main__':
    args = args.Args().args
    args.data = '../dataset/PTB/penn/'
    args.save = '00000'
    args.epochs = 2

    train_main(args)
    # save_path = f'../models/rhn_PTB/___e{0}___{args.save}.pt'
    # model_save(model, save_path)
    # print("Training a new model.")
    # starting_epoch = 1
