import os
import pickle
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from train import train_main
import collections
from LE_calculation import *
from util import *


def round1(num_epochs = 2, max_evals = 2, LE_based=False, count=100):
    space = {
        "sparse_init": hp.choice("sparse_init", ['uniform', 'ER']),
        "growth": hp.choice("growth", ['random']),
        "death": hp.choice("death", ['magnitude', 'SET', 'threshold', 'global_magnitude']),
        "redistribution": hp.choice("redistribution", ['magnitude', 'nonzeros', 'none']),
        "death_rate": hp.randint('death_rate', 6), # Returns a random integer in the range [0, upper)
    }

    pickle.dump(count, open('counter.txt', 'wb'))

    # define an objective function
    def objective(params):
        args = Args().args

        # customize change args
        args.sparsity = 0.67
        args.density = 1 - args.sparsity
        args.epochs = num_epochs
        args.eval_batch_size = 20
        args.seed = 42
        # print(params)
        # this allows change of global variable 'count'
        # global count_local
        count = pickle.load(open('counter.txt', 'rb'),)
        args.save = f'{count}'

        # set methodological and non-methodological hyperparameter according to params selection.
        args.init = params['sparse_init']
        args.growth = params['growth']
        args.death = params['death']
        args.redistribution = params['redistribution']
        args.death_rate = 0.1 * (params['death_rate'] + 4)

        args.verbose = False
        print(f'{count}: {args}')

        # count += 1
        # pickle.dump(count, open('counter.txt', 'wb'))
        # return {"loss": 0, "status": STATUS_OK, 'val_loss': math.exp(0),
        #         'test_loss': math.exp(0), 'args': args}

        # val_loss, test_loss = train_main(args)
        val_loss, test_loss = 0, 0
        if not LE_based:
            count += 1
            pickle.dump(count, open('counter.txt', 'wb'))
            return {"loss": math.exp(val_loss), "status": STATUS_OK, 'val_loss': math.exp(val_loss),
                    'test_loss': math.exp(test_loss), 'args': args}
        else:
            args.trial_num = count
            args.eval_batch_size = 2
            # calculate the LE
            LE_main(args)
            # calculate the LE distance
            LE_distance, _, _ = LE_distance_main(count, num_epochs=num_epochs)
            print(f"count: {count} \t LE_distance: {LE_distance}")
            count += 1
            pickle.dump(count, open('counter.txt', 'wb'))
            return {"loss": LE_distance, "status": STATUS_OK, 'val_loss': math.exp(val_loss),
                    'test_loss': math.exp(test_loss), 'args': args}

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials)
    return trials


# Simplifies the trials by creating a new dict called 'new_trials'
# the key of new_trials is a list of [init, growth, death, redistribution, death_rate]
# the value of new_trials is the result in trials which includes loss, args, etc
def simplify_trials(trials, max_evals, LE_based, ind):
    new_trials = collections.defaultdict(list)
    for i in range(len(trials.results)):
        death_rate = trials.results[i]['args'].death_rate
        new_trial_key = [trials.results[i]['args'].init, trials.results[i]['args'].growth,
                         trials.results[i]['args'].death, trials.results[i]['args'].redistribution,
                         f'{death_rate:.3f}']
        print(trials.results[i]['args'].save)
        new_trials[tuple(new_trial_key)] = trials.results[i]
    print(f'there are {len(new_trials)} unique candidates...')

    trials_path = '../trials/LSTM_PTB'
    if not os.path.exists(trials_path):
        os.mkdir(trials_path)

    # saving trials
    if LE_based:
        print("saving LE trials")
        print(os.path.exists(f'{trials_path}'))
        pickle.dump(new_trials, open(f'{trials_path}/LE_tpe_trials_num_{max_evals}_ind_{ind}.pickle', 'wb'))
    else:
        print(os.path.exists(f'{trials_path}'))
        print("saving PPL trials")
        pickle.dump(new_trials, open(f'{trials_path}/PPL_tpe_trials_num_{max_evals}_ind_{ind}.pickle', 'wb'))

# if __name__ == '__main__':
#     num_epochs = 1
#     max_evals = 10
#     count = 100
#     LE_based = True
#     count_local = count
#     trials = round1(num_epochs, max_evals, LE_based, count=count)
#     simplify_trials(trials=trials, max_evals=max_evals, ind=count_local, LE_based=LE_based)