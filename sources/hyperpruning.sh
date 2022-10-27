#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python future_rounds.py -ind 18000 --max_evals 40 --LE_based 'True' -e0 3 -ei 3 --hp_opt 'tpe' --initial_indices 15 21
CUDA_VISIBLE_DEVICES=0 python hyperpruning.py -ind 18000 --max_evals 40 --LE_based 'True' -e0 1 -ei 1 --hp_opt 'tpe'