from first_round import *
from main_pruning_competition import future_rounds
import train
from args import Args
import argparse
import sys

def main(args):
    parser = argparse.ArgumentParser(description="Hyperpruning")
    parser.add_argument("--starting_idx", type=int, default=100, required=False)
    parser.add_argument("--max_evals", type=int, default=40, required=False)
    parser.add_argument("--LE_based", type=str, default='True', required=False)
    parser.add_argument("--e0", type=int, default=3, required=False, help="number of epochs for the first round")
    parser.add_argument("--ei", type=int, default=3, required=False, help="number of epochs for the future rounds")
    parser.add_argument("--initial_indices", nargs="*", type=int, default=None)
    parser.add_argument("--hp_opt", type=str, default='tpe', required=False)
    parser.add_argument("--initial_pool", type=bool, default=False, required=False)
    parser.add_argument("--evaluate", type=str, default='../models/LSTM_PTB_pruned/___e100___136.pt', required=False)
    code_args = parser.parse_args(args)
    print(f"code_args: {code_args}")
    # count = code_args.starting_idx
    #
    # # just evaluating the given model without running hyperpruning
    if code_args.evaluate:
        args = Args().args
        args.evaluate = code_args.evaluate
        train_main(args)
    else:
        if not code_args.initial_pool: # need to generate the initial pool
            trials = round1(num_epochs=code_args.e0, max_evals=code_args.max_evals,
                            LE_based=code_args.LE_based, count=count)
            simplify_trials(trials=trials, max_evals=code_args.max_evals,
                            ind=code_args.starting_idx, LE_based=code_args.LE_based)
        future_rounds(code_args)

if __name__ == '__main__':
    main(sys.argv[1:])