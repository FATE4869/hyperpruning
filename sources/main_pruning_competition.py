import collections
import sys
import torch
import torch.nn as nn
import numpy as np
from train import train_main
from args import Args
from LE_calculation import *
from util import *

def continue_competition(args):
    parser = argparse.ArgumentParser(description="continue the competition")
    parser.add_argument("-ind", "--starting_idx", type=int, default=120000, required=False)
    parser.add_argument("--max_evals", type=int, default=2, required=False)
    parser.add_argument("--LE_based", type=str, default='True', required=False)
    parser.add_argument("-e0", type=int, default=3, required=False, help="number of epochs for the first round")
    parser.add_argument("-ei", type=int, default=3, required=False, help="number of epochs for the future rounds")
    parser.add_argument("--initial_indices", nargs="*", type=int, default=None)
    parser.add_argument("--hp_opt", type=str, default='tpe', required=False)

    args = parser.parse_args(args)
    LE_based = args.LE_based.lower()

    # epoch start with e0 + ei, and round start with 1
    e0 = args.e0
    ei = args.ei
    epoch = e0 + ei
    round = 1

    # loading result from the first round
    trials = pickle.load(open(f'../trials/LSTM_PTB/LE_{args.hp_opt}_trials_num_{args.max_evals}_ind_{args.starting_idx}.pickle', 'rb'))
    new_trials = collections.defaultdict(list)
    print(args.initial_indices)

    # select candidates based on the input [initial_indices].
    # if None, use all candidates in the trials.
    if args.initial_indices is not None:
        for trial_key in trials:
            if (trials[trial_key]['args'].trial_num - args.starting_idx) in args.initial_indices:
                vals = []
                for val in trial_key:
                    vals.append(val)
                new_trials[tuple(vals)] = trials[trial_key]
                print(trials[trial_key]['args'].trial_num - args.starting_idx)
    trials = new_trials

    # print(trials)
    # stop until only two candidates are left
    while len(trials) > 2:
        previous_indices = []
        distances = {}

        # sort trials based on their distance/loss
        for trial_key in trials:
            distances[trials[trial_key]['loss']] = trial_key
            previous_indices.append(trials[trial_key]['args'].save)
        distances_sorted = dict(sorted(distances.items()))
        new_trials = collections.defaultdict(list)

        # only keep half of previous candidates
        for i, distance in enumerate(distances_sorted):
            if i > len(distances) / 2:
                break
            else:
                new_key = distances_sorted[distance]
                vals = []
                for val in new_key:
                    vals.append(val)
                new_trials[tuple(vals)] = trials[new_key]

        # get indices of remaining candidates
        remaining_indices = []
        for trial_key in new_trials:
            remaining_indices.append(trials[trial_key]['args'].save)

        # Summary
        print(f"-------------------- Round: {round} --------------------\n"
              f"previous remaining indices: {previous_indices}...\n"
              f"current remaining indices: {remaining_indices}...\n"
              f"each will be trained for {epoch} epochs...\n"
              f"Keeping {len(new_trials)} out of {len(trials)}...\n"
              f"-----------------------------------------------------\n")
        # train remaining trails
        trials = new_trials
        for i, trial_key in enumerate(trials):
            args = trials[trial_key]['args']
            args.data = '../dataset/PTB/penn/'
            # args.seed = 42
            args.epochs = epoch
            args.eval_batch_size = 20
            print(args)
            # val_loss, test_loss = 0, 0
            val_loss, test_loss = train_main(args)
            trials[trial_key]['val_loss'] = val_loss
            trials[trial_key]['test_loss'] = test_loss
            if LE_based == 'true':
                if epoch < 50:
                    args.eval_batch_size = 5
                    LE_main(args)
                    LE_distance, _, _ = LE_distance_main(int(args.save), num_epochs=epoch)

            else:
                LE_distance = 0
            trials[trial_key]['loss'] = LE_distance
            print(
                f"count: {args.save}, val_loss: {math.exp(val_loss)}, test_loss: {math.exp(test_loss)}, LE_distance: {LE_distance}")
        epoch += ei
        round += 1

    # check the final selections
    for trial_key in trials:
        print(trials[trial_key])

    # extensively train the final selections
    epoch = 100
    val_losses = {}
    test_losses = {}
    for i, trial_key in enumerate(trials):
        args = trials[trial_key]['args']
        args.data = '../dataset/PTB/penn/'
        # args.seed = 42
        args.epochs = epoch
        args.eval_batch_size = 20
        print(args)
        val_loss, test_loss = train_main(args)
        val_losses[args.trial_num] = val_loss
        test_losses[args.trial_num] = test_loss
    print(f"val losses: {val_losses}")
    print(f"test losses: {test_losses}")


if __name__ == '__main__':
    continue_competition(sys.argv[1:])
