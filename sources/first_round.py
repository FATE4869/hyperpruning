import pickle
import os
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from train import train_main
import collections
from LE_calculation import *
from util import *

count = 10

def main(num_epochs = 2, max_evals = 2, LE_based=False):
    space = {
        "sparse_init": hp.choice("sparse_init", ['uniform', 'ER']),
        "growth": hp.choice("growth", ['random']),
        "death": hp.choice("death", ['magnitude', 'SET', 'threshold', 'global_magnitude']),
        "redistribution": hp.choice("redistribution", ['magnitude', 'nonzeros', 'none']),
        "death_rate": hp.randint('death_rate', 6), # Returns a random integer in the range [0, upper)
    }
    trials = Trials()

    # define an objective function
    def objective(params):
        args = Args().args
        args.sparsity = 0.67
        args.density = 1 - args.sparsity
        args.epochs = num_epochs
        args.eval_batch_size = 20
        args.seed = 42
        global count
        count_local = count
        args.save = f'{count}'
        # print(args.save)
        args.init = params['sparse_init']
        args.growth = params['growth']
        args.death = params['death']
        args.redistribution = params['redistribution']
        args.death_rate = 0.1 * (params['death_rate'] + 4)
        args.verbose = False
        print(f'{count}: {args}')
        val_loss, test_loss = train_main(args)
        if not LE_based:
            count += 1
            return {"loss": math.exp(val_loss), "status": STATUS_OK, 'val_loss': math.exp(val_loss),
                    'test_loss': math.exp(test_loss), 'args': args}
        else:
            args.trial_num = count
            args.eval_batch_size = 2
            LE_main(args)
            # LE_main_rhn(args)
            LE_distance, _, _ = LE_distance_main(count, num_epochs=num_epochs)
            print(f"count: {count} \t LE_distance: {LE_distance}")
            count += 1
            return {"loss": LE_distance, "status": STATUS_OK, 'val_loss': math.exp(val_loss),
                    'test_loss': math.exp(test_loss), 'args': args}

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials)

    print(best)
    print(trials)
    trials_path = '../trials/'
    count_local = 10
    if not os.path.exists(trials_path):
        os.mkdir(trials_path)
    if LE_based:
        print("saving LE trials")
        print(os.path.exists(f'{trials_path}'))
        pickle.dump(trials, open(f'{trials_path}/LE_tpe_trials_num_{max_evals}_ind_{count_local}.pickle', 'wb'))
    else:
        print(os.path.exists(f'{trials_path}'))
        print("saving PPL trials")
        pickle.dump(trials, open(f'{trials_path}/PPL_tpe_trials_num_{max_evals}_ind_{count_local}.pickle', 'wb'))
    return trials

if __name__ == '__main__':
    num_epochs = 2
    max_evals = 2
    # LE_based = True
    # trials = main(num_epochs, max_evals, LE_based)
    # LE_distances = []
    # val_losses = []
    trials = pickle.load(open(f'../trials/stacked_LSTM/LE_tpe_trials_num_40_ind_18000.pickle', 'rb'))
    new_trials = collections.defaultdict(dict)
    for i in range(len(trials.results)):
        print(i)
        # if i > 8:
        #     break
        # else:
        new_trial_key = []
        new_trial_key.append(trials.results[i]['args'].init)
        new_trial_key.append(trials.results[i]['args'].growth)
        new_trial_key.append(trials.results[i]['args'].death)
        new_trial_key.append(trials.results[i]['args'].redistribution)
        death_rate = trials.results[i]['args'].death_rate
        new_trial_key.append(f'{death_rate:.3f}')
        print(trials.results[i]['args'].save[:-3])
        trials.results[i]['args'].save = trials.results[i]['args'].save[:-3]
        new_trials[tuple(new_trial_key)] = trials.results[i]
    print(f'there are {len(new_trials)} unique candidates...')
    pickle.dump(new_trials, open(f'../trials/LSTM_PTB/LE_tpe_trials_num_{40}_ind_{18000}.pickle', 'wb'))
    # if LE_based:
    #     for i in range(max_evals):
    #         LE_distance = round(trials.results[i]['loss'], 2)
    #         LE_distances.append(LE_distance)
    #         print(LE_distance)
    #     print("\nval_losses: ")
    #     for i in range(max_evals):
    #         val_loss = round(trials.results[i]['val_loss'], 2)
    #         val_losses.append(val_loss)
    #         print(val_loss)
    #     print(f"LE_distances: {LE_distances}")
    #     print(f"val_losses: {val_losses}")
    # else:
    #     for i in range(max_evals):
    #         val_loss = round(trials.results[i]['val_loss'], 2)
    #         val_losses.append(val_loss)
    #         print(val_loss)
    #         print(f"val_losses: {val_losses}")