import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import stats
import pickle
from LE_calculation import *
from util import *
import os


def scatter_plot():
    indices = [102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 122, 130, 132, 133, 135, 136, 137, 138, 139]
    epochs = [3, 15]

    early_ppl = []
    early_LS = []
    mid_ppl = []
    mid_LS = []
    for index in indices:
        path = f'../LEs/LSTM_PTB_pruned/___e{epochs[0]}___{index}.pickle'
        LEs = pickle.load(open(path, 'rb'))
        early_ppl.append(LEs['test_perplexity'])

        path = f'../LEs/LSTM_PTB_pruned/___e{epochs[1]}___{index}.pickle'
        temp = epochs[1]
        while not os.path.exists(path):
            temp -= 1
            path = f'../LEs/LSTM_PTB_pruned/___e{temp}___{index}.pickle'
        LEs = pickle.load(open(path, 'rb'))
        mid_ppl.append(LEs['test_perplexity'])

        LE_distance, _, _ = LE_distance_main(index, num_epochs=epochs[0])
        early_LS.append(LE_distance)

        LE_distance, _, _ = LE_distance_main(index, num_epochs=epochs[1])
        mid_LS.append(LE_distance)
    early_ppl_normalized = [(i-min(early_ppl)) / (max(early_ppl) - min(early_ppl)) for i in early_ppl]
    mid_ppl_normalized = [(i - min(mid_ppl)) / (max(mid_ppl) - min(mid_ppl)) for i in mid_ppl]
    early_LS_normalized = [(i - min(early_LS)) / (max(early_LS) - min(early_LS)) for i in early_LS]
    mid_LS_normalized = [(i - min(mid_LS)) / (max(mid_LS) - min(mid_LS)) for i in mid_LS]

    late_ppl0 = [73.26, 73.54, 72.87, 79.77, 82.68, 71.89, 73.02, 80.65, 72.2, 72.15, 82.52, 81.77, 89.46, 72.71, 80.37, 79.71, 71.06, 71.75, 70.57, 70.87, 72.05, 70.89, 69.73, 71.94, 72.02, 71.12]


    print(stats.spearmanr(early_ppl_normalized, late_ppl0))
    print(stats.spearmanr(early_LS_normalized, late_ppl0))
    print(stats.spearmanr(mid_ppl_normalized, late_ppl0))
    print(stats.spearmanr(mid_LS_normalized, late_ppl0))

    plt.figure(1, figsize=(3, 2))
    plt.scatter(late_ppl0, early_ppl_normalized)
    plt.xlim([69, 91])
    plt.ylim([-.1, 1.1])
    plt.xticks([70, 75, 80, 85, 90], [])
    plt.yticks([0, 0.5, 1], [])
    plt.title('early ppl vs. late ppl')
    plt.show()

    plt.figure(2, figsize=(3, 2))
    plt.scatter(late_ppl0, mid_ppl_normalized)
    plt.xlim([69, 91])
    plt.ylim([-.1, 1.1])
    plt.xticks([70, 75, 80, 85, 90], [])
    plt.yticks([0, 0.5, 1], [])
    plt.title('mid ppl vs. late ppl')
    plt.show()

    plt.figure(3, figsize=(3, 2))
    plt.scatter(late_ppl0, early_LS_normalized)
    plt.xlim([69, 91])
    plt.ylim([-.1, 1.1])
    plt.xticks([70, 75, 80, 85, 90], [])
    plt.yticks([0, 0.5, 1], [])
    plt.title('early LS vs. late ppl')
    plt.show()

    plt.figure(4, figsize=(3, 2))
    plt.scatter(late_ppl0, mid_LS_normalized)
    plt.xlim([69, 91])
    plt.ylim([-.1, 1.1])
    plt.xticks([70, 75, 80, 85, 90], [])
    plt.yticks([0, 0.5, 1], [])
    plt.title('mid LS vs. late ppl')
    plt.show()

def LS_space():
    folder_names = ['LEs/LSTM_PTB_pruned', 'LEs/LSTM_PTB_full']
    indices = [136, 161]
    num_epochs = 100
    divider_indices = [0]
    LE, val_ppls, test_ppl = LE_loading(folder_names[0], indices[0], num_epochs)
    LE_full, val_ppls_full, _ = LE_loading(folder_names[1], indices[1], num_epochs)
    divider_indices.append(LE_full.shape[0])
    LEs = torch.cat([LE_full, LE])
    divider_indices.append(LEs.shape[0])
    print(f"divider_indices: {divider_indices}")
    distance = tsne(LEs, dim=2, divider_indices=divider_indices, use_tsne=False, plot=True, trial_nums=indices)

def epochs_vs_ppl():
    index = 136
    num_epochs = 100
    ppls = []
    for epoch in range(num_epochs):
        path = f'../LEs/LSTM_PTB_pruned/___e{epoch}___{index}.pickle'
        if os.path.exists(path):
            LEs = pickle.load(open(path, 'rb'))
            ppl = LEs['test_perplexity']
        ppls.append(ppl)
    plt.figure(1)
    plt.plot(range(5, num_epochs), ppls[5:])
    plt.show()
    # print(len(ppls))

if __name__ == '__main__':
    # scatter_plot()
    # LS_space()
    epochs_vs_ppl()