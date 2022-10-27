#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main_pruning_competition.py -ind 18000 --max_evals 40 --LE_based 'True' -e0 3 -ei 3 --initial_indices 15 21 --hp_opt 'tpe'
