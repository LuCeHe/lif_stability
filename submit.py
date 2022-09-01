import os, itertools, argparse, time, json, datetime, socket
from tqdm import tqdm

from GenericTools.keras_tools.esoteric_layers.surrogated_step import possible_pseudod
from GenericTools.stay_organized.VeryCustomSacred import summarize
from GenericTools.stay_organized.submit_jobs import run_experiments
from GenericTools.stay_organized.utils import summarize_logs

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

parser = argparse.ArgumentParser(description='main')

# types: spiking_transformers, send, summary, scancel:x:y
parser.add_argument('--type', default='send', type=str, help='main behavior')
args = parser.parse_args()

if args.type == 'send':
    # send experiments on Compute Canada for surrogate gradients paper
    n_seeds = 4
    seed = 104
    seeds = [seed + i for i in range(n_seeds)]
    # seeds = [seed + 3, seed + 4]

    save_model = False
    # final_experiments(seed)
    experiments = []
    send_fs = ['adaptsg']  # 1, 2, 3, 4, 5, 6 'extra' sparsity adaptsg

    # f1
    if 1 in send_fs:
        pairs = [
            # ['maLSNN', '1_embproj_noalif_nogradreset_dropout:.3_timerepeat:2_'],
            # ['maLSNN', '1_embproj_nogradreset_dropout:.3_timerepeat:2_'],
            ['spikingLSTM', '1_embproj_dropout:.3_timerepeat:2_'],
        ]

        for n_name, incomplete_comments in pairs:
            comments = [incomplete_comments + p for p in possible_pseudod]
            experiment = {
                'task_name': ['heidelberg', ], 'net_name': [n_name],
                'comments': comments, 'seed': seeds,
            }
            experiments.append(experiment)

        run_experiments(
            experiments,
            init_command='python surrogate_gradient.py with epochs=None steps_per_epoch=None batch_size=None '
                         'stack=None n_neurons=None ',
            run_string=None,
            sh_location=os.path.join(CDIR, 'experiments'),
            py_location=CDIR, duration={'days': 0, 'hours': 48, 'minutes': 0, 'prestop_training_hours': 3},
            account='def-jrouat', env_name='denv2', n_gpus=1, id='sg'
        )

        experiments = []

        pairs = [
            # ['maLSNN', '1_embproj_noalif_nogradreset_dropout:.3_timerepeat:2_'],
            ['maLSNN', '1_embproj_nogradreset_dropout:.3_timerepeat:2_'],
            # ['spikingLSTM', '1_embproj_dropout:.3_timerepeat:2_'],
        ]

        for n_name, incomplete_comments in pairs:
            comments = [incomplete_comments + p for p in possible_pseudod]
            experiment = {
                'task_name': ['heidelberg', ], 'net_name': [n_name],
                'comments': comments, 'seed': seeds,
            }
            experiments.append(experiment)

    if 2 in send_fs:
        # possible_pseudod = [
        #     'fastsigmoidpseudod',
        # 'cappedskippseudod',
        # ]
        # f2
        options = [f'{sg}_dampf:1._sharpn:1.' for sg in possible_pseudod]  # sorted(damp_options + sharp_options)
        # comment_architecture = '2_noalif_timerepeat:2_multreset2_nogradreset_'
        comment_architecture = '2_'

        ds_comments = [comment_architecture + o for o in options]
        experiment = {
            'task_name': ['lca', 'heidelberg'], 'net_name': ['maLSNN'], 'stack': [2], 'seed': seeds,
            'comments': ds_comments,
        }
        experiments.append(experiment)

        ds_comments = [comment_architecture + 'spikelca_' + o for o in options]
        experiment = {
            'task_name': ['lca', ], 'net_name': ['maLSNN'], 'stack': [2], 'seed': seeds,
            'comments': ds_comments,
        }
        experiments.append(experiment)

        # possible_pseudod = [
        #     'fastsigmoidpseudod',
        # ]

        dd = [.1, .25, .5, .75, 1., 1.25, 1.5]  # [.75, 1.25]
        ss = [.1, .25, .5, .75, 1., 1.25, 1.5]  # [.75, 1.25]
        damp_options = ['{}_dampf:{}_sharpn:1.'.format(*o) for o in list(itertools.product(possible_pseudod, dd))]
        sharp_options = ['{}_sharpn:{}_dampf:1.'.format(*o) for o in list(itertools.product(possible_pseudod, ss))]
        options = sorted(damp_options + sharp_options)
        # comment_architecture = '2_noalif_timerepeat:2_multreset2_nogradreset_'
        comment_architecture = '2_'

        ds_comments = [comment_architecture + o for o in options]
        experiment = {
            'task_name': ['lca', 'heidelberg'], 'net_name': ['maLSNN'], 'stack': [2], 'seed': seeds,
            'comments': ds_comments,
        }
        experiments.append(experiment)

        ds_comments = [comment_architecture + 'spikelca_' + o for o in options]
        experiment = {
            'task_name': ['lca', ], 'net_name': ['maLSNN'], 'stack': [2], 'seed': seeds,
            'comments': ds_comments,
        }
        experiments.append(experiment)

        # experiment = {
        #     'task_name': ['lca'], 'net_name': ['maLSNN'], 'stack': [2], 'seed': seeds,
        #     'comments': [
        #         comment_architecture + '_ntailpseudod_tailvalue:{}'.format(1 + 10 ** tv)
        #         for tv in np.linspace(-2, 1.2, 10)
        #     ],
        # }
        # experiments.append(experiment)

    if 3 in send_fs:
        # 'GlorotUniform', 'GlorotNormal', 'HeUniform', 'HeNormal', 'Orthogonal'
        # 'GlorotBiGamma', 'HeBiGamma', 'OrthogonalBiGamma'
        # f3

        experiment = {
            'task_name': ['heidelberg'], 'net_name': ['maLSNN'], 'stack': [2], 'seed': [seed],
            'comments': ['3_noalif_timerepeat:2_multreset2_nogradreset_fastsigmoidpseudod',
                         ],
            'initializer': ['BiGammaOrthogonal']
        }
        experiments.append(experiment)

        experiment = {
            'task_name': ['heidelberg'], 'net_name': ['maLSNN'], 'stack': [2], 'seed': [seed],
            'comments': ['3_noalif_timerepeat:2_multreset2_nogradreset_fastsigmoidpseudod',
                         '3_noalif_timerepeat:2_multreset2_nogradreset_gaussianpseudod',
                         '3_noalif_timerepeat:2_multreset2_nogradreset_originalpseudod',
                         ],
            'initializer': ['HeBiGamma']
        }
        experiments.append(experiment)

        experiment = {
            'task_name': ['heidelberg'], 'net_name': ['maLSNN'], 'stack': [2], 'seed': seeds[:2],
            'comments': ['3_noalif_timerepeat:2_multreset2_nogradreset_cappedskippseudod',
                         '3_noalif_timerepeat:2_multreset2_nogradreset_exponentialpseudod',
                         ],
            'initializer': ['HeBiGamma']
        }
        experiments.append(experiment)

    if 4 in send_fs:
        # f4

        for annealing_type in ['pea', 'ea', 'ha']:
            experiment = {
                'task_name': ['heidelberg'], 'net_name': ['maLSNN'],
                'batch_size': [32], 'stack': [2],
                'seed': seeds, 'comments': [
                    '4_embproj_annealing:{}_{}'.format(annealing_type, sg)
                    for sg in possible_pseudod
                ],
            }
            experiments.append(experiment)

    if 5 in send_fs:
        # f5
        combinations_conditions = [
            'naive',
            'conditionI_',
            'conditionII_',
            'conditionIII_',
            'conditionIV_',
            'conditionI_conditionII_',
            'conditionI_conditionII_conditionIII_',
            'conditionI_conditionII_conditionIII_conditionIV_',
        ]

        # experiment = {
        #     'task_name': ['heidelberg', 'sl_mnist'], 'net_name': ['maLSNN'], 'initializer': ['Orthogonal'],
        #     'stack': [2],
        #     'seed': seeds, 'comments':
        #         [
        #             '5_noalif_exponentialpseudod_nogradreset_multreset2_dropout:.1_timerepeat:2_{}'.format(
        #                 conditions)
        #             for conditions in combinations_conditions
        #         ],
        # }
        # experiments.append(experiment)

        experiment = {
            'task_name': ['wordptb', ], 'net_name': ['maLSNN'],
            'stack': ['1700:300'], 'n_neurons': [300],
            'seed': [seed], 'comments':
                [
                    '5_embproj_noalif_exponentialpseudod_nogradreset_multreset2_dropout:.3_timerepeat:2_{}'
                        .format(conditions)
                    for conditions in combinations_conditions
                ],
        }
        experiments.append(experiment)

    if 6 in send_fs:
        incomplete_comments = '6_embproj_noalif_nogradreset_dropout:.3_timerepeat:2_'
        comments = [incomplete_comments + p for p in possible_pseudod]

        experiments = []
        experiment = {
            'task_name': ['heidelberg', ], 'net_name': ['spikingLSTM'],
            'comments': ['6_embproj_dropout:.3_timerepeat:2_' + p for p in possible_pseudod], 'seed': seeds,
            'lr': [1e-2, 3.16e-3, 1e-3, 3.16e-4, 1e-4, 3.16e-5, 1e-5]
        }
        experiments.append(experiment)

        run_experiments(
            experiments,
            init_command='python surrogate_gradient.py with epochs=None steps_per_epoch=None batch_size=None '
                         'stack=None n_neurons=None ',
            run_string=None,
            sh_location=os.path.join(CDIR, 'experiments'),
            py_location=CDIR, duration={'days': 0, 'hours': 13, 'minutes': 0, 'prestop_training_hours': 3},
            account='def-jrouat', env_name='denv2', n_gpus=1, id='sg'
        )
        experiments = []

    if 'sparsity' in send_fs:
        incomplete_comment = '7_embproj_noalif_nogradreset_dropout:.3_timerepeat:2_v0m_'

        incomplete_comments = [incomplete_comment + f'adjfi:{i}_' for i in [.01, .1, .3, .5, .7]]

        comments = []
        for ff in  ['', 'adjff:.1', 'adjff:.01']: # ['', ]
            comments.extend([c + ff for c in incomplete_comments])

        experiment = {
            'task_name': ['heidelberg'], 'net_name': ['maLSNN'],
            'comments': comments, 'seed': seeds
        }
        experiments.append(experiment)

    if 'adaptsg' in send_fs:
        incomplete_comments = '8_embproj_noalif_nogradreset_dropout:.3_timerepeat:2_movedgauss_'
        # lif_comments = [incomplete_comments + t for t in ['', 'adaptsg', 'readaptsg:3', 'readaptsg:10']]
        lif_comments = [incomplete_comments + t for t in ['adaptsg', 'readaptsg:3']]
        alif_comments = [c.replace('noalif_', '') for c in lif_comments]
        experiment = {
            'task_name': ['sl_mnist', 'heidelberg', 'wordptb'], 'net_name': ['maLSNN'],
            'comments': lif_comments + alif_comments, 'seed': seeds
        }
        experiments.append(experiment)

    if 'extra' in send_fs:
        comments = ['6_embproj_noalif_nogradreset_multreset2_dropout:.3_timerepeat:2_tenb_' + p for p in
                    possible_pseudod]
        experiment = {
            'task_name': ['wordptb'], 'net_name': ['maLSNN'],
            'stack': ['1700:300'], 'n_neurons': [300],
            'seed': [seed], 'comments':
                [
                    *comments,
                    '6_embproj_noalif_learnablepseudod_nogradreset_multreset2_dropout:.3_timerepeat:2_tenb',
                    '6_embproj_noalif_learnablepseudod_nogradreset_multreset2_dropout:.3_timerepeat:2_tenb_mgauss',
                    '6_embproj_noalif_learnablepseudod_nogradreset_multreset2_dropout:.3_timerepeat:2_tenb_mtail',
                    '6_embproj_noalif_learnablepseudod_nogradreset_multreset2_dropout:.35_timerepeat:2_tenb',
                    '6_embproj_noalif_learnablepseudod_nogradreset_multreset2_dropout:.35_timerepeat:2_tenb_mgauss',
                    '6_embproj_noalif_learnablepseudod_nogradreset_multreset2_dropout:.35_timerepeat:2_tenb_mtail',
                ],
        }
        experiments.append(experiment)

    run_experiments(
        experiments,
        init_command='python surrogate_gradient.py with epochs=None steps_per_epoch=None batch_size=None '
                     'stack=None n_neurons=None ',
        run_string=None,
        sh_location=os.path.join(CDIR, 'experiments'),
        py_location=CDIR, duration={'days': 0, 'hours': 13, 'minutes': 0, 'prestop_training_hours': 3},
        account='def-jrouat', env_name='denv2', n_gpus=1, id='sg'
    )


elif args.type in ['initcond']:
    experiments = []
    experiment = {
        'task_name': ['sl_mnist', 'wordptb', 'heidelberg'], 'n_seeds': [10], 'conditions': ['all'],
        'init_seed': [i * 10 for i in range(10)], 'tests': [0],
        'steps_per_epoch': [-1], 'plot': [0], 'redoseeds': [0]
    }
    experiments.append(experiment)
    # experiment = {
    #     'task_name': ['wordptb', 'heidelberg'], 'n_seeds': [5],
    #     'steps_per_epoch': [-1],
    # }
    # experiments.append(experiment)

    run_experiments(
        experiments,
        init_command='python conditions_initialization.py ',
        run_string=None, is_argparse=True,
        sh_location=os.path.join(CDIR, 'experiments'),
        py_location=CDIR, duration={'days': 0, 'hours': 2, 'minutes': 0, 'prestop_training_hours': -1},
        account='def-jrouat', env_name='denv2', n_gpus=1, id='initcond'
    )


elif args.type in ['cev']:
    # send experiments on Compute Canada for stochastic differential equations paper

    # experiments = []
    # import numpy as np
    #
    # rhos = np.linspace(-2, -.1, 20)
    # experiment = {
    #     'noise_type': ['cev', ], 'theta': ['position', 'amplitude'],
    #     'rho': rhos, 'n_evaluations': [int(1e6)],
    # }
    # experiments.append(experiment)
    #
    # run_experiments(
    #     experiments, init_command='python figure_2_toyoizumiamari.py ', is_argparse=True,
    #     run_string='sbatch run_small.sh '
    # )

    run_experiments(None, init_command='python figure_1.py', run_string='sbatch run_3d.sh ')

elif 'continue' in args.type:
    # continue experiment unfinished

    # beluga
    exps_to_continue = [
        '2021-04-26--16-07-18--2511-mnl_',
    ]

    print('Number jobs: {}'.format(len(exps_to_continue)))
    for e in exps_to_continue:
        path_config = os.path.join(CDIR, 'experiments', e, '1', 'config.json')

        with open(path_config) as f:
            data = json.load(f)
        data['continue_training'] = 'continue_' + e
        config_update = ' '.join(['{}={}'.format(k, str(v)) for k, v in data.items()])
        print(config_update)
        run_string = 'sbatch run_tf2.sh '
        # command = 'python nonstochastic_language_main_old_psheid.py with ' + config_update
        command = 'python language_main.py with ' + config_update

        run_string += "'{}'".format(command)
        print(run_string)
        # os.system(command)
        os.system(run_string)
    print('Number jobs: {}'.format(len(exps_to_continue)))

elif 'summary' in args.type:
    # summarize content of ungoing training sessions
    # track_params = ['net_name', 'task_name', 'n_dt_per_step', 'spike_dropout', 'stack', 'n_neurons',
    #                 'batch_size', 'comments', 'continue_training', 'initializer']
    # summarize(CDIR, track_params)
    summarize_logs(CDIR)

elif 'scancel' in args.type:

    # the string should look like args.type=scancel:1000:2000
    initial = int(args.type.split(':')[1])
    final = int(args.type.split(':')[2])
    for i in range(initial, final):
        os.system(f'scancel {i}')


elif 'ziptheunzipped' in args.type:
    import shutil

    EXPERIMENTS = os.path.join(CDIR, 'experiments')
    ds = [d for d in os.listdir(EXPERIMENTS) if not '.zip' in d and not '.sh' in d and not '.txt' in d]
    for d in tqdm(ds):
        path = os.path.join(EXPERIMENTS, d)

        shutil.make_archive(path, 'zip', path)

else:
    raise NotImplementedError
