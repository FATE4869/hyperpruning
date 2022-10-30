#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python hyperpruning.py --starting_idx 100 --max_evals 40 --LE_based 'True' -e0 3 -ei 3 --hp_opt 'tpe'