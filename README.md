# Hyperpruning

This repository is the official implementation of Hyperpruning: Hyperpruning first generates an initial candidate pool based on hyperparameter optimization algorithms (HPO) and LS-based or loss-based metric. It then iteratively excludes candidates based on LS distance or loss. Remaining candidates are extensively trained until their accuracies converge.


![](Hyperpruning.png)

## System Requirements
Our code is run on Ubuntu 20.04. No non-standard hardware is required.

## Installation Guide
### requirement
- Python 3.8
- Pytorch 1.11.0
- Additional requirements in requirements.txt
  - ```bash
    conda create --name hyperpruning python=3.8
    # installation could take a couple minutes
    pip install -r requirements.txt
    
## Selecting the optimal candidate
  Following is the code for Stacked-LSTM
  ````
  python hyperpruning.py -ind 1000 --max_evals 40 --LE_based 'True' -e0 1 -ei 1 --hp_opt 'tpe'
  ````
Options:
* -ind: it is for tracking different experiments and does not have any impact on the experiments
* --max_evals: is the number of candidates in the initial pool, --hp_opt defines the HPO
* --LS_based: if 'True', it uses LS distance as the metric, otherwise uses current loss
* -e0: the number of epochs for round 1
* -e1: the number of epochs for future rounds
* -hp_opt: it decides which hyperparameter optimization algorithms (HPO) to use ('tpe', 'atpe')

## Selected Candidate

You can download the pretrained selected Selfish stacked-LSTM models here:

* Selfish stacked-LSTM trained on PTB.

This model gives 69.95 test perplexity on PTB dataset at sparsity of 0.67.

[//]: # (To evaluate this pre-trained model, you need to run:)