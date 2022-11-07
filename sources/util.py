import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.cm as cmx
import matplotlib.colors as colors
from numpy.linalg import norm

def LE_plot(x_embedded, divider_indices, trial_nums):
    # colormap = cmx.Set1.colors
    colors = cmx.rainbow(np.linspace(0, 1, len(divider_indices) - 2))
    plt.figure()
    trial_nums = [100000] + trial_nums
    # colors = ['g', 'b', 'y', 'cyan']
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], color='r')
    for i, indices in enumerate(divider_indices):
        if i == 1:
            plt.plot(x_embedded[divider_indices[i-1]:divider_indices[i], 0],
                        x_embedded[divider_indices[i-1]:divider_indices[i], 1],
                        color='k',  linewidth=5, label=trial_nums[i-1])
        elif i != 0:
            plt.plot(x_embedded[divider_indices[i-1]:divider_indices[i], 0],
                        x_embedded[divider_indices[i-1]:divider_indices[i], 1],
                        color=colors[i-2],  linewidth=5, label=trial_nums[i-1])

    for i, indices in enumerate(divider_indices):
        if i != 0:
            plt.scatter(x_embedded[divider_indices[i - 1], 0],
                        x_embedded[divider_indices[i - 1], 1],
                        color='k')
    plt.legend()
    plt.show()

def return_candidates(distances, num_trials, starting_epoch=3, incremental_epoch=2):
    distance2ind = {}
    distances_check = distances
    for i in range(num_trials):
        distance2ind[distances_check[i, -1]] = i
    checking_epoch = starting_epoch
    # indices_original = np.linspace(0, 23, 24, dtype=int)
    indices_remaining = np.linspace(0, num_trials-1, num_trials, dtype=int)
    while len(indices_remaining) > 3:
        # indices_sorted = np.argsort(distances_check[:, checking_epoch])
        indices_sorted = np.argsort(distances_check[:, checking_epoch])
        remaining_trials = int(len(indices_sorted) / 2)
        indices_remaining = indices_sorted[:remaining_trials]
        distances_check = distances_check[indices_remaining]
        checking_epoch += incremental_epoch
    indices_candidates = []
    for distance in distances_check[:, -1]:
        indices_candidates.append(distance2ind[distance])
    return indices_candidates

def cal_distance(data, divider_indices, num_trials, num_epochs, starting_epoch=0):
    distances = np.zeros([num_epochs, num_trials])
    plt.figure(1)
    for a in range(num_epochs):
        for j in range(num_trials):
            distances[a, j] = torch.sqrt(torch.sum(torch.square(data[a + starting_epoch]
                                                       - data[a + divider_indices[j+1] + starting_epoch])))

        plt.plot(range(9), distances[a, :], label=f'{a + starting_epoch}')
    plt.legend()
    plt.show()

def perplexity_compare(names, indices, num_epochs, labels=None, indices_candidates=[]):
    if labels is None:
        labels = names
    colors = cmx.rainbow(np.linspace(0, 1, len(names)))
    # print(colors)
    # colors = cmx.rainbow(np.linspace(0, 1, 4))
    ppls = torch.zeros([len(names), num_epochs])
    divider_indices = []
    plt.figure()
    count = 0
    starting_epoch = 5
    ending_epoch = num_epochs
    for i, name in enumerate(names):
        # print(i, name)
        previous_ppl = 0
        for epoch in range(num_epochs):
            file_path = f'../LEs/{name}/___e{epoch}___{indices[i]}.pickle'
            print(file_path)
            if os.path.exists(file_path):
                LEs = pickle.load(open(file_path, 'rb'))
                # print(LEs)
                ppls[i, epoch] = LEs['current_perplexity']
                previous_ppl = LEs['current_perplexity']
            else:
                ppls[i, epoch] = previous_ppl

        divider_indices.append(num_epochs * i)

    # if i in range(len(indices)):
        plt.plot(range(starting_epoch, ending_epoch), ppls[i, starting_epoch: ending_epoch], label=indices[i], color=colors[count], linewidth=3)
        count += 1
    # else:
    #     plt.plot(range(6, num_epochs), ppls[i, 6: num_epochs], label=labels[i], color='k')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("perplexity")
    print(ppls)
    print(ppls[:, -1])
    # print(ppls[indices_candidates, -1])
    plt.show()
    # return_candidates(ppls[1:].numpy(), num_trials=24, starting_epoch=4, incremental_epoch=2)
    # cal_distance(torch.flatten(ppls), divider_indices, num_trials=len(names)-1, num_epochs=10, starting_epoch=0)

def LE_loading(folder_name, trial_index, num_epochs=None, epochs=None):
    if num_epochs is not None:
        epochs = np.arange(num_epochs+1)
    print(f'epochs calculating LS distance: {epochs}')
    val_ppls = []
    test_ppls = []
    for i, epoch in enumerate(epochs):
        file_name = f'../{folder_name}/___e{epoch}___{trial_index}.pickle'
        if os.path.exists(file_name):
            saved = pickle.load(open(file_name, 'rb'))
            LE = torch.transpose(torch.unsqueeze(saved['LEs'], 1), 0, 1)
            val_ppl = saved['current_perplexity']
            if 'test_perplexity' in saved:
                test_ppl = saved['test_perplexity']
            else:
                test_ppl = val_ppl
            if i == 0:
                LEs = LE
            else:
                LEs = torch.cat([LEs, LE])
        val_ppls.append(val_ppl)
        test_ppls.append(test_ppl)

    LEs = LEs.detach().cpu()
    return LEs, val_ppls, test_ppls

def tsne(X, dim=2, divider_indices=None, names=None, labels=None, use_tsne=True, plot=False, limited_samples=None, trial_nums=None):
    if labels is None:
        labels = names
    z = torch.zeros((), device=X.device, dtype=X.dtype)
    inf_indices = torch.where(torch.isinf(X))
    for i, inf_index in enumerate(range(inf_indices[0].shape[0])):
        X[inf_indices[0][i], inf_indices[1][i]] = 0
    # dimension reduction
    if use_tsne: # T-SNE
        tsne_model = TSNE(perplexity=10, n_components=2, random_state=1)
        x_embedded = tsne_model.fit_transform(X.detach().numpy())
    else: # PCA
        U,S,V = torch.pca_lowrank(X)
        x_embedded = torch.matmul(X, V[:, :dim]).detach().numpy()
    if plot:
        LE_plot(x_embedded, divider_indices, trial_nums)

    # l2 distance
    distance = np.sqrt(np.sum(
        np.square(x_embedded[divider_indices[1] - 1] - x_embedded[divider_indices[2] - 1])
    ))
    return distance

    # cosine distance
    # distance = np.dot(x_embedded[divider_indices[1] - 1], x_embedded[divider_indices[2] - 1]) / \
    #            (norm(x_embedded[divider_indices[1] - 1]) * norm(x_embedded[divider_indices[2] - 1]))
    # return distance * 100

def main():
    names = ['LEs/stacked_LSTM_full'] + ['LEs/stacked_LSTM_pruned'] * 24 # + ['LEs/RigL'] * 10
    # indices = [161, 5000, 5007, 5009, 5012, 5021, 5022]
    indices = [161] + np.linspace(5000, 5023, 24, dtype=int).tolist()
              # + np.linspace(1000, 1009, 10, dtype=int).tolist()

def LE_distance_main(trial_num, num_epochs, epoch=None, last_epoch_ref=True):

    folder_names = ['LEs/LSTM_PTB_pruned', 'LEs/LSTM_PTB_full']
    indices = [trial_num, 161]

    divider_indices = [0]
    LE, val_ppls, test_ppl = LE_loading(folder_names[0], indices[0], num_epochs)
    # LE_full, val_ppls_full, _ = LE_loading(folder_names[1], indices[1], epochs=[49])
    # LE_full, val_ppls_full, _ = LE_loading(folder_names[1], indices[1], num_epochs=49)
    LE_full, val_ppls_full, _ = LE_loading(folder_names[1], indices[1], num_epochs)
    divider_indices.append(LE_full.shape[0])
    LEs = torch.cat([LE_full, LE])
    divider_indices.append(LEs.shape[0])
    print(f"divider_indices: {divider_indices}")
    distance = tsne(LEs, dim=2, divider_indices=divider_indices, use_tsne=False)
    return distance, val_ppls[-1], val_ppls_full[-1]
    # return 0, 0, 0


def LE_multiple(trial_nums, num_epochs, last_epoch_ref=True):

    # LEs = LE_loading('LEs/RHN_full', 100000, num_epochs=num_epochs)
    # LEs, _, _ = LE_loading('LEs/RHN_PTB_full', 110000, epochs = [29])
    LEs, _, _ = LE_loading('LEs/RHN_PTB_full/110000', 110000, num_epochs=200)
    divider_indices = [0]
    distance_sum = LEs.shape[0]
    divider_indices.append(distance_sum)
    for i, index in enumerate(trial_nums):
        LE, val_ppls, test_ppls = LE_loading(f'LEs/RHN_PTB_pruned/{index}', index, num_epochs)
        # LE = LE_loading(name, index, num_epochs=None, epochs=[num_epochs])

        LEs = torch.cat([LEs, LE])
        distance_sum += LE.shape[0]
        divider_indices.append(distance_sum)
    print(divider_indices)
    distance = tsne(LEs, divider_indices=divider_indices, use_tsne=False, plot=True, trial_nums=trial_nums)
    # return distance

def generate_dic():
    trial_num = 110000
    name = f'RHN_PTB_full/{trial_num}/'

    num_epochs = 500
    info_dic = {'epochs': [], 'ppls': []}
    for i in range(num_epochs):
        file_name = f'../LEs/{name}/___e{i}___{trial_num}.pickle'
        if os.path.exists(file_name):
            saved = pickle.load(open(file_name, 'rb'))
            LE = torch.transpose(torch.unsqueeze(saved['LEs'], 1), 0, 1)
            info_dic['epochs'].append(i)
            info_dic['ppls'].append(saved['current_perplexity'])
        if i == 0:
            LEs = LE
        else:
            LEs = torch.cat([LEs, LE])
    info_dic['LEs'] = LEs
    # print(info_dic)
    pickle.dump(info_dic, open(f'../LEs/RHN_PTB_full/{trial_num}/info_dic_{trial_num}.pickle', 'wb'))
    return info_dic

if __name__ == '__main__':
    trial_nums = [18015, 18021, 18022]
    epoch = 9
    val_ppls = []
    distances = []
    for trial_num in trial_nums:
        distance, val_ppl, val_ppl_full = LE_distance_main(trial_num, num_epochs=epoch)
        print(f'Trial Num: {trial_num}, at epoch: {epoch} --- > ref val perplexity: {val_ppl_full:.1f}    val perplexity: {val_ppl:.1f}   distance: {distance:.4f}')
        # val_ppls.append(val_ppl)
        # distances.append(distance)
    # val = np.sort(val_ppls)
    # indices = np.argsort(val_ppls)
    # distances_sorted = np.sort(distances)
    # indices = np.argsort(distances)
    # print(distances_sorted, indices)