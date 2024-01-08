import numpy as np
import os, itertools, argparse, time, json, datetime
from pyaromatics.stay_organized.submit_jobs import run_experiments
from lif_stability.neural_models import possible_pseudod


def final_experiments(seed):
    experiments = []

    ss = [10 ** i for i in np.linspace(-2, 2, 5)]
    options_a = ['1_taua:{}_originalpseudod_vt/t'.format(s) for s in ss] #['1_taua:{}'.format(s) for s in ss]
    options_v = []  # ['1_tauv:{}'.format(s) for s in ss]
    experiment = {
        'task_name': ['ptb', ], 'net_name': ['LSNN'],  # ['maLSNN', 'LSNN'],
        'n_neurons': [1000], 'batch_size': [32], 'n_dt_per_step': [1], 'stack': [1, ],
        'comments': [*options_a, *options_v
                     ],
        'embedding': ['zero_mean:None:100:100'], 'seed': [seed], 'epochs': [None], 'steps_per_epoch': [None],
    }
    experiments.append(experiment)
    #

    #
    # ss = [10 ** i for i in np.linspace(-1, 1, 5)]
    # options = ['2_{}_sharpn:{}'.format(*o) for o in list(itertools.product(possible_pseudod, ss))]
    # experiment = {
    #     'task_name': ['ptb', ], 'net_name': ['maLSNN', 'LSNN'],
    #     'n_neurons': [1000], 'batch_size': [32], 'n_dt_per_step': [1], 'stack': [1, ],
    #     'comments': [*options],
    #     'embedding': ['zero_mean:None:100:100'], 'seed': [seed], 'epochs': [None], 'steps_per_epoch': [None],
    # }
    # experiments.append(experiment)

    # ss = np.linspace(.5, 1.5, 5)
    # options = ['3_{}_dampf:{}'.format(*o) for o in list(itertools.product(possible_pseudod, ss))]
    # experiment = {
    #     'task_name': ['ptb', ], 'net_name': ['maLSNN', 'LSNN'],
    #     'n_neurons': [1000], 'batch_size': [32], 'n_dt_per_step': [1], 'stack': [1, ],
    #     'comments': [*options],
    #     'embedding': ['zero_mean:None:100:100'], 'seed': [seed], 'epochs': [None], 'steps_per_epoch': [None],
    # }
    # experiments.append(experiment)

    # experiment = {
    #     'task_name': ['ptb', ], 'net_name': ['maLSNN', ],
    #     'n_neurons': [1000, 1500, 2000, 2500, 3000, 3500, 4000], 'batch_size': [32], 'n_dt_per_step': [1], 'stack': [1, ],
    #     'comments': ['4_'],
    #     'embedding': ['zero_mean:None:100:100'], 'seed': [seed], 'epochs': [None], 'steps_per_epoch': [None],
    # }
    # experiments.append(experiment)
    #
    # experiment = {
    #     'task_name': ['ptb', ], 'net_name': ['maLSNN', ],
    #     'n_neurons': [1000, ], 'batch_size': [32], 'n_dt_per_step': [1], 'stack': [1, 3, 5, 7],
    #     'comments': ['5_'],
    #     'embedding': ['zero_mean:None:100:100'], 'seed': [seed], 'epochs': [None], 'steps_per_epoch': [None],
    # }
    # experiments.append(experiment)

    experiment = {
        'task_name': ['ptb', ], 'net_name': ['maLSNN', ],
        'n_neurons': [1000], 'batch_size': [32], 'n_dt_per_step': [1, 2, 3, 4], 'stack': [1, ],
        'comments': ['6_simplereadout'],
        'embedding': ['zero_mean:None:100:100'], 'seed': [seed], 'epochs': [None], 'steps_per_epoch': [None],
    }
    experiments.append(experiment)

    init_command = 'python language_main.py with stop_time=72000 '
    run_experiments(experiments, init_command=init_command, run_string='sbatch run_23h.sh ')

    # train 3 days
    experiments = []

    experiment = {
        'task_name': ['ptb', ], 'net_name': ['maLSNN', ],
        'n_neurons': [1000, ], 'batch_size': [32], 'n_dt_per_step': [1], 'stack': [5, 7],
        'comments': ['5_'],
        'embedding': ['zero_mean:None:100:100'], 'seed': [seed], 'epochs': [None], 'steps_per_epoch': [None],
    }
    experiments.append(experiment)

    experiment = {
        'task_name': ['ptb', ], 'net_name': ['maLSNN', 'LSNN'],
        'n_neurons': [1000, ], 'batch_size': [32], 'n_dt_per_step': [1], 'stack': [5],  # [2, 3, 4],
        'comments': [
            # '7_addrestrellis_transformer_dilation_drop1',
            # '7_addrestrellis_transformer_drop1',
            # '7_addrestrellisshared_transformer_drop1',
            '7_addrestrellis_drop1',
            # '7_stack_drop1',
            # '7_stack_drop1_vt/t',
        ],
        'embedding': ['zero_mean:None:100:100'], 'seed': [seed], 'epochs': [None], 'steps_per_epoch': [None],
    }
    experiments.append(experiment)

    init_command = 'python language_main.py with stop_time=245000 '
    run_experiments(experiments, init_command=init_command, run_string='sbatch run_3d.sh ')
