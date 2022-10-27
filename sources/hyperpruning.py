from first_round import *
from main_pruning_competition import future_rounds
from args import Args
import argparse
import sys
import os
import time
import torch

def main(args):
    parser = argparse.ArgumentParser(description="continue the competition")
    parser.add_argument("-ind", "--starting_idx", type=int, default=100, required=False)
    parser.add_argument("--max_evals", type=int, default=3, required=False)
    parser.add_argument("--LE_based", type=str, default='True', required=False)
    parser.add_argument("-e0", type=int, default=2, required=False, help="number of epochs for the first round")
    parser.add_argument("-ei", type=int, default=1, required=False, help="number of epochs for the future rounds")
    parser.add_argument("--initial_indices", nargs="*", type=int, default=None)
    parser.add_argument("--hp_opt", type=str, default='tpe', required=False)
    code_args = parser.parse_args(args)
    print(code_args)
    count = code_args.starting_idx
    trials = round1(num_epochs=code_args.e0, max_evals=code_args.max_evals,
                    LE_based=code_args.LE_based, count=count)
    simplify_trials(trials=trials, max_evals=code_args.max_evals,
                    ind=code_args.starting_idx, LE_based=code_args.LE_based)
    future_rounds(code_args)

if __name__ == '__main__':
    main(sys.argv[1:])