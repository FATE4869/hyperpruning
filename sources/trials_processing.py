import collections
import pickle

def main():
    trials = pickle.load(open('../trials/LE_tpe_trials_num_2_ind_120000.pickle', 'rb'))
    new_trials = collections.defaultdict(dict)
    for i in range(len(trials.results)):
        new_trial_key = []
        new_trial_key.append(trials.results[i]['args'].init)
        new_trial_key.append(trials.results[i]['args'].growth)
        new_trial_key.append(trials.results[i]['args'].death)
        new_trial_key.append(trials.results[i]['args'].redistribution)
        death_rate = trials.results[i]['args'].death_rate
        new_trial_key.append(f'{death_rate:.3f}')
        new_trials[tuple(new_trial_key)] = trials.results[i]
if __name__ == '__main__':
    main()