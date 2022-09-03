import os, json, argparse, copy
from datetime import timedelta, datetime

from tensorflow import reduce_prod

from GenericTools.keras_tools.esoteric_initializers import glorotcolor, orthogonalcolor, hecolor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
FMT = '%Y-%m-%dT%H:%M:%S'

from tqdm import tqdm
import pandas as pd
import matplotlib as mpl
import numpy as np
from scipy.stats import mannwhitneyu
from matplotlib.lines import Line2D

import pickle
import matplotlib.pyplot as plt

from GenericTools.keras_tools.convergence_metric import convergence_estimation
from GenericTools.keras_tools.esoteric_layers.surrogated_step import possible_pseudod, clean_pseudname, \
    clean_pseudo_name, pseudod_color
from GenericTools.keras_tools.plot_tools import plot_history, TensorboardToNumpy
# from GenericTools.PlotTools.mpl_tools import load_plot_settings
from GenericTools.stay_organized.unzip import unzip_good_exps
from GenericTools.stay_organized.plot_tricks import large_num_to_reasonable_string
from GenericTools.stay_organized.statistics import significance_to_star
from GenericTools.stay_organized.utils import timeStructured, str2val
from GenericTools.stay_organized.mpl_tools import load_plot_settings

from GenericTools.keras_tools.esoteric_tasks.time_task_redirection import Task
# from stochastic_spiking.language_main import build_model
# from sg_design_lif.neural_models import clean_pseudo_name, pseudod_color
from sg_design_lif.visualization_tools.plotting_tools import smart_plot, postprocess_results

mpl, pd = load_plot_settings(mpl=mpl, pd=pd)

CDIR = os.path.dirname(os.path.realpath(__file__))
EXPERIMENTS = os.path.join(CDIR, 'experiments')

CSVPATH = os.path.join(EXPERIMENTS, 'means.h5')
HSITORIESPATH = os.path.join(EXPERIMENTS, 'histories.json')

metric_sort = 'v_ppl'
reduce_samples = True

parser = argparse.ArgumentParser(description='main')
parser.add_argument(
    '--type', default='nothing', type=str, help='main behavior',
    choices=[
        'excel', 'histories', 'interactive_histories', 'activities', 'weights', 'continue', 'robustness', 'init_sg',
        'pseudod', 'move_folders', 'conventional2spike', 'n_tail', 'task_net_dependence', 'sharpness_dampening',
        'conditions', 'lr_sg'
    ]
)
args = parser.parse_args()

GEXPERIMENTS = [
    r'C:\Users\PlasticDiscobolus\work\sg_design_lif\good_experiments',
    # r'C:\Users\PlasticDiscobolus\work\stochastic_spiking\good_experiments\2022-02-10--best-ptb-sofar',
    # r'C:\Users\PlasticDiscobolus\work\stochastic_spiking\good_experiments\2022-02-11--final_for_lif',
    # r'D:\work\stochastic_spiking\good_experiments\2022-02-16--verygood-ptb',
    # r'C:\Users\PlasticDiscobolus\work\stochastic_spiking\good_experiments\2022-02-16--verygood-ptb'
]

_, starts_at_s = timeStructured(False, True)


# reduce_prod
def history_pick(k, v):
    if any([n in k for n in ['loss', 'perplexity', 'entropy', 'bpc']]):
        o = np.nanmin(v[10:])
    elif any([n in k for n in ['acc']]):
        o = np.nanmax(v[10:])
    else:
        o = f'{round(v[0], 3)}/{round(v[-1], 3)}'

    return o


if not os.path.exists(CSVPATH):
    ds = unzip_good_exps(
        GEXPERIMENTS, EXPERIMENTS,
        exp_identifiers=['mnl'], except_folders=[],
        unzip_what=['run.json', ]  # 'png_content', 'train_model']
    )

    # first check how many have the history file:
    nohistoryds = [d for d in ds if not os.path.exists(os.path.join(d, 'other_outputs', 'history.json'))]
    ds = [d for d in ds if not d in nohistoryds]

    histories = {}
    df = pd.DataFrame()
    list_results = []
    for d in tqdm(ds):
        print(d)
        # d_path = os.path.join(EXPERIMENTS, d)
        history_path = os.path.join(d, 'other_outputs', 'history.json')
        hyperparams_path = os.path.join(d, 'other_outputs', 'results.json')
        config_path = os.path.join(d, '1', 'config.json')
        run_path = os.path.join(d, '1', 'run.json')

        with open(history_path) as f:
            history = json.load(f)

        if len(history['loss']) > 5:

            with open(config_path) as f:
                config = json.load(f)

            with open(run_path) as f:
                run = json.load(f)

            results = {}
            results.update({'where': run['host']['hostname']})

            if 'stop_time' in run.keys():
                results.update({'duration_experiment':
                                    datetime.strptime(run['stop_time'].split('.')[0], FMT) - datetime.strptime(
                                        run['start_time'].split('.')[0], FMT)
                                })
            results.update({k: history_pick(k, v) for k, v in history.items()})
            results.update({k: v for k, v in config.items()})
            results.update({'d_name': d})

            if os.path.exists(hyperparams_path):
                with open(hyperparams_path) as f:
                    hyperparams = json.load(f)
                    if 'comments' in hyperparams.keys():
                        hyperparams['final_comments'] = hyperparams['comments']
                        hyperparams.pop('comments', None)

                results.update({k: postprocess_results(k, v) for k, v in hyperparams.items()})

            list_results.append(results)
            # small_df = pd.DataFrame([results])

            # df = df.append(small_df)
            history = {k.replace('val_', ''): v for k, v in history.items() if 'val' in k}

            histories[d] = history

    df = pd.DataFrame.from_records(list_results)
    df.loc[df['comments'].str.contains('noalif'), 'net_name'] = 'LIF'
    df.loc[df['net_name'].str.contains('maLSNN'), 'net_name'] = 'ALIF'
    df.loc[df['net_name'].str.contains('spikingLSTM'), 'net_name'] = 'sLSTM'

    df.loc[df['task_name'].str.contains('wordptb'), 'task_name'] = 'PTB'
    df.loc[df['task_name'].str.contains('heidelberg'), 'task_name'] = 'SHD'
    df.loc[df['task_name'].str.contains('sl_mnist'), 'task_name'] = 'sl-MNIST'

    cols = list(df)
    cols.insert(0, cols.pop(cols.index('convergence')))
    df = df.loc[:, cols]

    df = df.sort_values(by='comments')

    df.to_hdf(CSVPATH, key='df', mode='w')
    json.dump(histories, open(HSITORIESPATH, "w"))
else:
    # mdf = pd.read_csv(CSVPATH)
    df = pd.read_hdf(CSVPATH, 'df')  # load it
    with open(HSITORIESPATH) as f:
        histories = json.load(f)

df = df.sort_values(by='v_sparse_mode_accuracy', ascending=False)

history_keys = [
    'v_perplexity', 'v_sparse_mode_accuracy', 'v_firing_rate', 'v_loss',
    't_perplexity', 't_sparse_mode_accuracy', 't_firing_rate', 't_loss',
    'v_firing_rate_ma_lsnn', 'v_firing_rate_ma_lsnn_1',
    'firing_rate_ma_lsnn', 'firing_rate_ma_lsnn_1',
]

config_keys = [
    'comments', 'initializer', 'optimizer_name', 'seed',
    'weight_decay', 'clipnorm', 'task_name', 'net_name',  # 'lr_schedule'  # 'continue_training',
]
hyperparams_keys = [
    'n_params', 'final_epochs', 'duration_experiment', 'convergence', 'lr', 'stack', 'n_neurons', 'embedding',
    'batch_size',
]
extras = ['d_name', 'where']  # , 'where', 'main_file','accumulated_epochs',

keep_columns = history_keys + config_keys + hyperparams_keys + extras
remove_columns = [k for k in df.columns if k not in keep_columns]
df.drop(columns=remove_columns, inplace=True)

df = df.rename(
    columns={
        'val_sparse_categorical_accuracy': 'val_zacc', 'val_sparse_mode_accuracy': 'val_macc',
        'sparse_categorical_accuracy_test': 'test_macc', 'val_perplexity': 'val_ppl',
        'perplexity_test': 'test_ppl',
        't_sparse_mode_accuracy': 't_macc', 't_perplexity': 't_ppl',
        'v_sparse_mode_accuracy': 'v_macc', 'v_perplexity': 'v_ppl',
    },
    inplace=False
)
# df = df[df['task_name'].str.contains('PTB')]
# df = df[df['final_epochs'] == 3]

# df['comments'] = df['comments'].str.replace('_dampf:.3', '')
# df['comments'] = df['comments'].str.replace('_dropout:.3', '')
# df = df[(df['comments'].str.contains('ptb2')) | (df['task_name'].str.contains('SHD')) | (
#     df['task_name'].str.contains('sl-MNIST'))]
df['comments'] = df['comments'].str.replace('_ptb2', '')
# df = df[df['comments'].str.contains('_v0m')]
df = df[df['d_name'] > r'C:\Users\PlasticDiscobolus\work\sg_design_lif\experiments\2022-08-13']
# df = df[~(df['d_name'].str.contains('2022-08-10--')) | (df['d_name'].str.contains('2022-08-11--'))]

for ps in possible_pseudod:
    df['comments'] = df['comments'].str.replace('timerepeat:2' + ps, 'timerepeat:2_' + ps)

# df = df[(df['d_name'].str.contains('2022-08-12--'))|(df['d_name'].str.contains('2022-08-13--'))]
# df = df[(df['d_name'].str.contains('2022-08-27--'))]
df['comments'] = df['comments'].replace({'1_embproj_nogradres': '6_embproj_nogradres'}, regex=True)

# df = df.dropna(subset=['t_ppl'])


early_cols = ['task_name', 'net_name', 'n_params', 'final_epochs', 'comments', 'firing_rate_ma_lsnn',
              'firing_rate_ma_lsnn_1']
some_cols = [n for n in list(df.columns) if not n in early_cols]
df = df[early_cols + some_cols]

group_cols = ['net_name', 'task_name', 'initializer', 'comments', 'lr']
# only 4 experiments of the same type, so they have comparable statistics

if reduce_samples:
    df = df.sort_values(by='d_name', ascending=True)
    df = df.groupby(group_cols).sample(4, replace=True)

print(df.to_string())

counts = df.groupby(group_cols).size().reset_index(name='counts')
metrics_oi = ['v_ppl', 'v_macc', 't_ppl', 't_macc']

mdf = df.groupby(
    group_cols, as_index=False
).agg({m: ['mean', 'std'] for m in metrics_oi})

for metric in metrics_oi:
    mdf['mean_{}'.format(metric)] = mdf[metric]['mean']
    mdf['std_{}'.format(metric)] = mdf[metric]['std']
    mdf = mdf.drop([metric], axis=1)

mdf['counts'] = counts['counts']
mdf = mdf.sort_values(by='mean_' + metric_sort, ascending=False)

print(mdf.to_string())

_, ends_at_s = timeStructured(False, True)
duration_experiment = timedelta(seconds=ends_at_s - starts_at_s)
print('Time to load the data: ', str(duration_experiment))

# print(mdf.to_string())
if args.type == 'excel':

    # df = df[df['d_name'].str.contains('2021-12-29')]
    tasks = np.unique(df['task_name'])

    for task in tasks:
        print(task)
        idf = df[df["task_name"] == task]
        idf = idf.sort_values(
            by=['val_macc' if not task in ['wordptb', 'PTB'] else 'val_bpc'],
            ascending=False if not task in ['wordptb', 'PTB'] else True
        )
        print(idf.to_string(index=False))
        print('\n\n')

    # print(df.to_string(index=False))
    print(df.shape)


elif args.type == 'n_tail':
    idf = df[df['comments'].str.contains('2_')]
    idf = df[df['comments'].str.contains('_tailvalue')]
    counts = idf.groupby(['comments', ]).size().reset_index(name='counts')
    left = counts[counts['counts'] < 4]
    done = counts[counts['counts'] == 4]['comments'].values
    print(done)

    print()
    idf = mdf[mdf['comments'].str.contains('2_')]
    idf = idf.loc[idf['comments'].isin(done)]
    idf = idf.sort_values(by='mean_val_macc', ascending=False)
    print(idf.to_string(index=False))

    idf = idf[idf['comments'].str.contains('tailvalue')]
    tails = idf['comments'].str.replace('2_noalif_timerepeat:2_multreset2_nogradreset__ntailpseudod_tailvalue:',
                                        '').values.astype(float)

    sorted_idx = tails.argsort()
    accs = idf['mean_val_macc'].values[sorted_idx]
    stds = idf['std_val_macc'].values[sorted_idx]
    tails = tails[sorted_idx]

    cm = plt.get_cmap('Oranges')
    # print(idf.to_string(index=False))
    fig, axs = plt.subplots(1, 1, gridspec_kw={'wspace': .0, 'hspace': 0.}, figsize=(6, 6))
    axs.plot(tails, accs, color=cm(.5))
    axs.fill_between(tails, accs - stds, accs + stds, alpha=0.5, color=cm(.5))

    # max_means = {k: [] for k in tails}
    # for index, row in idf.iterrows():
    #     tail = str2val(row['comments'], 'tailvalue')
    #     d = row['d_name']
    #     event_dir = os.path.join(d, 'other_outputs', 'train')
    #     event_filename = os.path.join(event_dir, [p for p in os.listdir(event_dir) if 'events' in p][0])
    #
    #     means, stds = TensorboardToNumpy(event_filename, id_selection='grad')
    #
    #     mean_mean = np.array([np.median(np.abs(list(v.values()))) for v in means.values()])[::2]
    #     mean_std = np.array([np.median(np.abs(list(v.values()))) for v in stds.values()])[::2]
    #     # axs[1].plot(mean_mean)
    #     max_means[tail].append(np.mean(mean_mean))
    #
    # means = np.array([np.mean(v) for _, v in max_means.items()])
    # stds = np.array([np.std(v) for _, v in max_means.items()])
    # axs[1].plot(tails, means)
    # axs[1].fill_between(tails, means - stds, means + stds, alpha=0.5)

    for pos in ['right', 'left', 'bottom', 'top']:
        axs.spines[pos].set_visible(False)

    axs.set_xlabel('$q$ tail fatness')
    axs.set_xscale('log')
    axs.set_yticks([0.89, .9, .91])

    # axs[1].set_ylabel('mean gradient\nmagnitude')
    axs.set_ylabel('accuracy')
    plot_filename = r'experiments/figure2_tails.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

    plt.show()

    print(counts)
    print(left)
    print(done)


elif args.type == 'sharpness_dampening':
    feature = 'sharpness'  # sharpness dampening

    for feature in ['dampening', 'sharpness']:
        feature_oi = 'dampf' if feature == 'dampening' else 'sharpn'

        idf = mdf[mdf['comments'].str.contains('2_')]

        idf = idf[idf['comments'].str.contains(feature_oi)]
        idf = idf[idf['net_name'].str.contains('LIF')]
        idf = idf[idf['task_name'].str.contains('sl-MNIST')]
        print(idf.to_string(index=False))

        comments = np.unique(mdf['comments'])
        fig, axs = plt.subplots(1, 1, gridspec_kw={'wspace': .0, 'hspace': 0.}, figsize=(6, 6))

        for pn in possible_pseudod:
            iidf = idf[idf['comments'].str.contains(pn)].sort_values(by='comments', ascending=False)
            foi_values = [str2val(d, feature_oi) for d in iidf['comments']]
            accs = iidf['mean_val_macc'].values
            stds = iidf['std_val_macc'].values
            stds = np.nan_to_num(stds)
            axs.plot(foi_values, accs, color=pseudod_color(pn))
            axs.fill_between(foi_values, accs - stds, accs + stds, alpha=0.5, color=pseudod_color(pn))

        value = .204 if feature == 'dampening' else 1.02
        axs.axvline(x=value, color='k', linestyle='--')
        for pos in ['right', 'left', 'bottom', 'top']:
            axs.spines[pos].set_visible(False)

        axs.set_xlabel(feature)
        axs.set_ylabel('accuracy')
        plot_filename = r'experiments/figure2_{}.pdf'.format(feature)
        fig.savefig(plot_filename, bbox_inches='tight')

        plt.show()

        idf = df[df['comments'].str.contains('1_')]
        idf = idf[idf['comments'].str.contains(feature_oi)]
        idf = idf[idf['net_name'].str.contains('ALIF')]
        idf = idf[idf['task_name'].str.contains('sl-MNIST')]

        counts = idf.groupby(['comments', ]).size().reset_index(name='counts')
        left = counts[counts['counts'] < 4]
        print(counts)
        print(left)


elif args.type == 'lr_sg':
    def sensitivity_metric(out_vars, in_vars, name='diff'):
        assert out_vars.keys() == in_vars.keys()
        lrs = out_vars.keys()
        if name == 'ratio':
            metric = np.mean([out_vars[lr] / in_vars[lr] for lr in lrs])
        elif name == 'mean':
            metric = np.mean([out_vars[lr] for lr in lrs])
        elif name == 'diff':
            metric = np.mean([abs(out_vars[lr] - in_vars[lr]) for lr in lrs])
        else:
            raise NotImplementedError

        return metric


    per_task_variability = {}
    metric = 'v_ppl'

    net_name = 'LIF'  # LIF sLSTM
    tasks = ['sl-MNIST', 'SHD', 'PTB']  # for LIF
    nets = ['LIF', 'ALIF', 'sLSTM']
    task_sensitivity = {}
    net_sensitivity = {}
    task_sensitivity_std = {}
    net_sensitivity_std = {}

    fig, axs = plt.subplots(
        2, len(tasks) + 1, figsize=(12, 7),
        gridspec_kw={'wspace': .5, 'hspace': .5, 'width_ratios': [1, 1, 1, 2]}
    )

    # if not isinstance(axs, list):
    #     axs = [axs]

    # mdf = mdf[mdf['comments'].str.contains('6_')]
    mdf = mdf[mdf['comments'].str.contains('_dropout:.3')]
    df = df[df['comments'].str.contains('_dropout:.3')]

    # plot lr vs metric
    for i, task in enumerate(tasks):
        idf = mdf
        idf = idf[idf['net_name'].eq(net_name)]
        idf = idf[idf['task_name'].str.contains(task)]
        idf = idf.sort_values(by=['mean_' + metric], ascending=False)

        # print(idf.to_string(index=False))

        comments = np.unique(mdf['comments'])
        for pn in possible_pseudod:
            iidf = idf[idf['comments'].str.contains(pn)]
            lrs = np.unique(iidf['lr'])

            accs = []
            stds = []
            for lr in lrs:
                ldf = iidf[iidf['lr'] == lr]
                accs.append(ldf['mean_' + metric].values[0])
                stds.append(ldf['std_' + metric].values[0] / 2)

            stds = np.nan_to_num(stds)

            axs[0, i].plot(lrs, accs, color=pseudod_color(pn))
            axs[0, i].fill_between(lrs, accs - stds, accs + stds, alpha=0.5, color=pseudod_color(pn))

        axs[0, i].set_title(task)

    # compute task sensitivities
    for task in tasks:
        print('-' * 30)
        print(task)
        idf2 = df[df['net_name'].eq(net_name)]
        idf2 = idf2[idf2['comments'].str.contains('6_embproj_noalif_nogradreset_dropout:.3_timerepeat:2_')]
        idf2 = idf2[idf2['task_name'].str.contains(task)]

        # print(idf2.to_string())
        items = -1
        if task == 'sl-MNIST':
            items = 10
        elif task == 'SHD':
            items = 20
        elif task == 'PTB':
            items = 10000

        lrs = np.unique(idf2['lr'])
        out_vars = {}
        for lr in lrs:
            iidf2 = idf2[idf2['lr'].eq(lr)]
            out_vars[lr] = np.std(iidf2[metric]) / items

        pn_vars = {}
        for pn in possible_pseudod:
            iidf = idf[idf['comments'].str.contains(pn)]
            lrs = np.unique(iidf['lr'])
            pn_vars[pn] = {}

            for lr in lrs:
                iidf2 = iidf[iidf['lr'].eq(lr)]
                pn_vars[pn][lr] = iidf2['std_' + metric]

        in_vars = {lr: np.mean([pn_vars[pn][lr] for pn in possible_pseudod]) for lr in lrs}

        # print(out_vars)
        task_sensitivity[task] = sensitivity_metric(out_vars, in_vars)
        print(task, task_sensitivity[task])
        task_sensitivity_std[task] = np.std([out_vars[lr] for lr in lrs])

    task = 'SHD'
    for i, net_name in enumerate(nets):
        idf = mdf
        idf = idf[idf['net_name'].eq(net_name)]
        idf = idf[idf['comments'].str.contains('6_')]
        idf = idf[idf['task_name'].str.contains(task)]
        idf = idf.sort_values(by=['mean_' + metric], ascending=False)

        # print(idf.to_string(index=False))

        comments = np.unique(mdf['comments'])

        for pn in possible_pseudod:
            iidf = idf[idf['comments'].str.contains(pn)]
            lrs = np.unique(iidf['lr'])

            accs = []
            stds = []
            for lr in lrs:
                ldf = iidf[iidf['lr'] == lr]
                accs.append(ldf['mean_' + metric].values[0])
                stds.append(ldf['std_' + metric].values[0] / 2)

            stds = np.nan_to_num(stds)
            # print(accs, stds, lrs)
            axs[1, i].plot(lrs, accs, color=pseudod_color(pn))
            axs[1, i].fill_between(lrs, accs - stds, accs + stds, alpha=0.5, color=pseudod_color(pn))

        axs[1, i].set_title(net_name)
    # if len(tasks) > 1 and tasks[2] == 'PTB':
    #     axs[2].set_ylim([80, 800])

    # compute sensitivities to net
    task = 'SHD'
    items = 20
    for net_name in nets:
        print('-' * 30)
        print(net_name)

        idf = mdf
        idf = idf[idf['net_name'].eq(net_name)]
        idf = idf[idf['comments'].str.contains('6_')]
        idf = idf[idf['task_name'].str.contains(task)]
        idf = idf.sort_values(by=['mean_' + metric], ascending=False)

        idf2 = df[df['net_name'].eq(net_name)]
        idf2 = idf2[idf2['comments'].str.contains('6_embproj_')]
        idf2 = idf2[idf2['task_name'].str.contains(task)]
        lrs = np.unique(idf2['lr'])
        out_vars = {}
        for lr in lrs:
            iidf2 = idf2[idf2['lr'].eq(lr)]
            out_vars[lr] = np.std(iidf2[metric]) / items

        pn_vars = {}
        for pn in possible_pseudod:
            iidf = idf[idf['comments'].str.contains(pn)]
            lrs = np.unique(iidf['lr'])
            pn_vars[pn] = {}

            for lr in lrs:
                iidf2 = iidf[iidf['lr'].eq(lr)]
                pn_vars[pn][lr] = iidf2['std_' + metric]

        in_vars = {lr: np.mean([pn_vars[pn][lr] for pn in possible_pseudod]) for lr in lrs}
        ratio = np.mean([out_vars[lr] / in_vars[lr] for lr in lrs])
        metric_2 = np.mean([out_vars[lr] for lr in lrs])
        print('out_vars: ', out_vars)
        print('in_vars:  ', in_vars)
        print(net_name, ratio, metric_2)
        net_sensitivity[net_name] = sensitivity_metric(out_vars, in_vars)
        net_sensitivity_std[net_name] = np.std([out_vars[lr] for lr in lrs])

    for j in range(2):
        for i in range(len(tasks)):
            axs[j, i].set_xscale('log')
            axs[j, i].set_xticks([1e-2, 1e-3, 1e-4, 1e-5])

        for i in range(len(tasks) + 1):
            for pos in ['right', 'left', 'bottom', 'top']:
                axs[j, i].spines[pos].set_visible(False)

    axs[1, 2].set_xlabel('Learning rate')
    axs[0, 0].set_ylabel('Perplexity')

    axs[0, -1].bar(tasks, task_sensitivity.values(),
                   yerr=np.array(list(task_sensitivity_std.values())) / 2, color='maroon', width=0.4)
    axs[0, -1].set_ylabel('Sensitivity')
    axs[0, -1].set_xlabel('Task')

    axs[1, -1].bar(nets, net_sensitivity.values(),
                   yerr=np.array(list(net_sensitivity_std.values())) / 2, color='maroon', width=0.4)
    axs[1, -1].set_ylabel('Sensitivity')
    axs[1, -1].set_xlabel('Neural Model')

    axs[0, 0].text(-.7, .5, 'LIF network', fontsize=18,
                   horizontalalignment='center', verticalalignment='center', rotation='vertical',
                   transform=axs[0, 0].transAxes)
    axs[1, 0].text(-.7, .5, 'SHD task', fontsize=18,
                   horizontalalignment='center', verticalalignment='center', rotation='vertical',
                   transform=axs[1, 0].transAxes)

    for i in [0, 1]:
        box = axs[i, -1].get_position()
        box.x0 = box.x0 + 0.05
        box.x1 = box.x1 + 0.05
        axs[i, -1].set_position(box)

    for i, label in enumerate('abcg'):
        axs[0, i].text(-.3, 1.2, f'{label})', fontsize=14, color='#535353',
                       horizontalalignment='center', verticalalignment='center',
                       transform=axs[0, i].transAxes)

    for i, label in enumerate('defh'):
        axs[1, i].text(-0.3, 1.2, f'{label})', fontsize=14, color='#535353',
                       horizontalalignment='center', verticalalignment='center',
                       transform=axs[1, i].transAxes)

    legend_elements = [Line2D([0], [0], color=pseudod_color(n), lw=4, label=clean_pseudname(n))
                       for n in possible_pseudod]
    plt.legend(ncol=3, handles=legend_elements, loc='lower center', bbox_to_anchor=(-1.4, -.85))

    plt.show()
    plot_filename = f'experiments/lr_sg.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

elif args.type == 'init_sg':

    mini_df = df[df['comments'].str.contains('3_')]
    print(mini_df.to_string())
    idf = mdf[mdf['comments'].str.contains('3_')]

    idf['comments'] = idf['comments'].str.replace('3_embproj_snudecay_', '')
    idf['comments'] = idf['comments'].str.replace('3_noalif_timerepeat:2_multreset2_nogradreset_', '')
    idf['initializer'] = idf['initializer'].str.replace('BiGammaOrthogonal', 'OBiGamma')
    idf['initializer'] = idf['initializer'].str.replace('Orthogonal', 'OrthogonalNormal')
    idf['initializer'] = idf['initializer'].str.replace('OBiGamma', 'OrthogonalBiGamma')

    pseudods = possible_pseudod  # np.unique(idf['comments'])
    desired_initializers = ['HeNormal', 'HeUniform', 'HeBiGamma', 'GlorotNormal', 'GlorotUniform', 'GlorotBiGamma',
                            'OrthogonalNormal', 'OrthogonalBiGamma']
    # desired_initializers = ['HeBiGamma', 'GlorotBiGamma', 'OrthogonalBiGamma']

    print(mini_df.to_string(index=False))
    print(idf.to_string(index=False))
    idf = idf[['mean_val_macc', 'std_val_macc', 'comments', 'initializer']]
    idf = idf.loc[idf['initializer'].isin(desired_initializers)]
    idf['initializer'] = idf['initializer'].str.replace('BiGamma', ' BiGamma')
    idf['initializer'] = idf['initializer'].str.replace('Normal', ' Normal')
    idf['initializer'] = idf['initializer'].str.replace('Uniform', ' Uniform')

    idf = idf.sort_values(by=['mean_val_macc'], ascending=False)
    print(idf.to_string(index=False))

    # fig, axs = plt.subplots(1, 1, gridspec_kw={'wspace': .0, 'hspace': 0.}, figsize=(15, 5))

    fig = plt.figure(figsize=(15, 5))

    gs = fig.add_gridspec(1, 4)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])

    x = np.arange(len(desired_initializers))  # the label locations
    width = 1 / (len(pseudods) + 1)  # the width of the bars
    for i in range(len(pseudods)):
        c = pseudod_color(pseudods[i])
        # iidf = idf[idf['comments'] == pseudods[i]]
        iidf = idf[idf['comments'].str.contains(pseudods[i])]
        iidf = iidf.sort_values('initializer')
        print(iidf)
        if not iidf.empty:
            ax1.bar(
                x + i * width - (len(pseudods) - 1) / 2 * width,
                iidf['mean_val_macc'].values,
                yerr=iidf['std_val_macc'].values, width=width, color=c
            )
    ax1.set_ylim([.6, .9])
    clean_initializers_n = [
        d.replace('BiGammaOrthogonal', 'OrthogonalBiGamma').replace('Normal', ' Normal')
            .replace('Uniform', ' Uniform').replace('BiGamma', ' BiGamma')
        for d in desired_initializers]

    ax1.set_xticks(np.arange(len(desired_initializers)))
    ax1.set_xticklabels(clean_initializers_n)
    ax1.set_ylabel('accuracy')

    import seaborn as sns


    def init_color(name):
        if 'orot' in name:
            color = glorotcolor
        elif 'rthogonal' in name:
            color = orthogonalcolor
        else:
            color = hecolor
        return color


    idf['comments'] = idf['comments'].apply(clean_pseudname)
    palette = {p: init_color(p) for p in clean_initializers_n}
    sns.boxplot(y='mean_val_macc', x='initializer', data=idf, ax=ax2, palette=palette)
    ax2.set_ylabel('')
    # ax.set_xticklabels(x_labels, rotation='vertical', ha='center')
    ax2.set_xlabel('')
    # ax2.set_ylim([.7, .88])

    palette = {clean_pseudname(p): pseudod_color(p) for p in possible_pseudod}
    sns.boxplot(y='mean_val_macc', x='comments', data=idf, ax=ax3, palette=palette)
    ax3.set_ylabel('')
    ax3.set_xlabel('')

    for tick in [*ax2.get_xticklabels(), *ax3.get_xticklabels(), *ax1.get_xticklabels()]:
        tick.set_rotation(45)
        tick.set_ha('right')

    for pos in ['right', 'left', 'bottom', 'top']:
        ax1.spines[pos].set_visible(False)
        ax2.spines[pos].set_visible(False)
        ax3.spines[pos].set_visible(False)

    ax1.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='x', labelsize=14)

    plot_filename = r'experiments/figure3_initializations.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()

    idf = df[df['comments'].str.contains('3_')]
    idf['initializer'] = idf['initializer'].str.replace('BiGammaOrthogonal', 'OrthogonalBiGamma')

    idf = idf.loc[idf['initializer'].isin(desired_initializers)]
    counts = idf.groupby(['comments', 'initializer']).size().reset_index(name='counts')
    left = counts[counts['counts'] < 4]
    print(counts)
    print(left)


elif args.type == 'conditions':
    option = 2

    idf = mdf[mdf['comments'].str.contains('5_')]
    idf = idf[idf['task_name'].str.contains('SHD')]
    print(idf.to_string())
    idf['comments'] = idf['comments'].str.replace('condition', '')
    idf['comments'] = idf['comments'].str.replace('timerepeat:2_', '')
    idf['comments'] = idf['comments'].str.replace('5_noalif_exponentialpseudod_', '')
    # idf['comments'] = idf['comments'].str.replace('', 'naive')
    idf['comments'] = idf['comments'].str.replace(r'_$', '')
    idf['comments'] = idf['comments'].str.replace('_', '+')
    # idf['comments'] = idf['comments'].str.replace('II+I+', 'I+II\n')
    # idf = idf.replace(r'^\s*$', 'naive', regex=True)

    order_conditions = idf['comments']
    print(idf)
    order_conditions = ['naive', 'I', 'II', 'III', 'IV:b', 'II+I', 'II+I+III', 'II+I+III+IV:b']
    # order_conditions = ['naive', 'I', 'II', 'III', 'IV', 'I+II', 'I+II+III', 'I+II+III+IV']
    idf = idf[idf['comments'].isin(order_conditions)]
    idf['comments'] = idf['comments'].str.replace('IV:b', 'IV')
    # idf['comments'] = idf['comments'].str.replace('+III', '\n+III')
    idf['comments'] = idf['comments'].apply(lambda x: "I+II" + x[4:] if x.startswith("II+I") else x)
    idf['comments'] = idf['comments'].apply(lambda x: x.replace('+III', '\n+III'))

    idf = idf.sort_values(by='mean_test_macc', ascending=True)  # mean_test_macc mean_val_macc
    order_conditions = idf['comments'].values
    order_conditions = ['no\nconditions' if o == 'naive' else o for o in order_conditions]
    print(idf)
    print(order_conditions)
    means_val = idf['mean_val_macc'].values
    stds_val = idf['std_val_macc'].values
    means_test = idf['mean_test_macc'].values
    stds_test = idf['std_test_macc'].values

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    niceblue = '#0883E0'
    colors = [niceblue, niceblue, '#97A7B3', niceblue, niceblue, niceblue, niceblue, niceblue]

    axs[1].bar(range(len(means_val)), means_val, yerr=stds_val, width=.8, color=colors)
    axs[0].bar(range(len(means_test)), means_test, yerr=stds_test, width=.8, color=colors)
    axs[1].set_ylim([.8, .95])
    axs[1].set_yticks([.8, .85, .9, .95])
    axs[0].set_ylim([.5, .7])
    axs[0].set_yticks([.5, .6, .7])
    axs[0].set_xticks([])
    axs[1].set_xticks(np.arange(len(order_conditions)))
    axs[1].set_xticklabels(order_conditions, ha='center')
    axs[1].set_xlabel('conditions')
    axs[1].set_ylabel('validation\naccuracy')
    axs[0].set_ylabel('test\naccuracy')

    for ax in axs:
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    plot_filename = r'experiments/figure5_conditions.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()



elif args.type == 'task_net_dependence':

    idf = df[df['comments'].str.contains('1_')]
    idf = idf[~idf['comments'].str.contains('sharpn')]
    idf = idf[~idf['comments'].str.contains('dampf')]
    # idf = idf[idf['d_name'].str.contains('2021-12')]
    idf = idf[~idf['task_name'].str.contains('s_mnist')]

    # idf['comments'] = idf['comments'].str.replace('dampf:1.0', '')
    idf['comments'] = idf['comments'].str.replace('1_embproj_', '')
    idf['comments'] = idf['comments'].str.replace('snudecay_', '')
    idf['comments'] = idf['comments'].str.replace('noalif_', '')

    idf = idf[~idf['comments'].str.contains('dampf')]
    idf = idf[~idf['comments'].str.contains('cauchypseudod')]
    idf = idf[~idf['comments'].str.contains('annealing')]
    idf = idf[idf['comments'].str.contains('pseudod')]

    figsize = (16, 8)

    # print(idf.to_string(index=False))
    # print(idf.shape, ' when it should be ', )
    counts = idf.groupby(['task_name', 'net_name', 'comments']).size().reset_index(name='counts')
    # print(counts.to_string(index=False))

    what2plot = ['net', 'task', ]
    fig, axs = plt.subplots(len(what2plot), 3, gridspec_kw={'wspace': 0.1, 'hspace': .4}, figsize=figsize, sharey=False)
    # plt.subplots_adjust(right=0.9)
    # tasks:
    for ci, choice in enumerate(what2plot):  # ['task', 'net']:
        axs[ci, 0].set_ylabel('loss')

        print('\nDependency on {} choice'.format(choice))
        if choice == 'task':
            choices = ['SHD', 'sl-MNIST', 'PTB']  # np.unique(idf['task_name'])
            iidf = idf[idf['net_name'] == 'LIF']
            min_loss_len = 0
        elif choice == 'net':
            choices = ['LIF', 'ALIF', 'sLSTM']  # np.unique(idf['net_name'])
            iidf = idf[idf['task_name'].str.contains('SHD')]
            min_loss_len = 100

        for a_i, c in enumerate(choices):

            for pos in ['right', 'left', 'bottom', 'top']:
                axs[ci, a_i].spines[pos].set_visible(False)

            if not (a_i, ci) == (0, 1):
                print(c)

                iiidf = iidf[iidf['{}_name'.format(choice)].str.strip() == c]
                loss_curves = {k: [] for k in possible_pseudod}

                for _, row in iiidf.iterrows():
                    ptype = str2val(row['comments'], 'pseudod', str, equality_symbol='', remove_flag=False)
                    if ptype in possible_pseudod:
                        curve = histories[row['d_name']]['sparse_categorical_crossentropy']
                        if len(curve) > min_loss_len:
                            loss_curves[ptype].append(curve)
                            # axs[a_i].plot(curve, label='train xe', color=pseudod_color(ptype), linestyle=(0, (5, 3)),
                            #               linewidth=.5)

                min_len = 200
                for ptype in possible_pseudod:
                    if len(loss_curves[ptype]) > 0:
                        min_len = np.min([len(c) for c in loss_curves[ptype]] + [min_len])

                for ptype in possible_pseudod:
                    # for ptype in ['originalpseudod']:
                    print(ptype, ' has ', len(loss_curves[ptype]), ' runs ')
                    if len(loss_curves[ptype]) > 0:
                        # min_len = np.min([len(c) for c in loss_curves[ptype]])
                        equalized_loss_curves = np.array([c[:min_len] for c in loss_curves[ptype]])
                        mean = np.mean(equalized_loss_curves, axis=0)
                        std = np.std(equalized_loss_curves, axis=0)
                        axs[ci, a_i].plot(mean, color=pseudod_color(ptype))
                        axs[ci, a_i].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5,
                                                  color=pseudod_color(ptype))
                axs[ci, a_i].set_title(c)
    axs[0, 0].set_xlabel('training iteration')

    plt.text(28, 6.5, 'LIF network', rotation=90, fontsize=18, ha='right')
    plt.text(28, 13, 'SHD task', rotation=90, fontsize=18, ha='right')

    legend_elements = [Line2D([0], [0], color=pseudod_color(n), lw=4, label=clean_pseudname(n))
                       for n in possible_pseudod]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(-2.15, .5))
    # plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 1.2))
    axs[-1, 0].axis('off')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plot_filename = r'experiments/figure1_net_task.pdf'.format(choice)
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()



elif args.type == 'conventional2spike':
    pd.options.mode.chained_assignment = None
    print()
    idf = df[df['comments'].str.contains('4_')]

    print(idf)
    # annealings = idf['comments'].str.split('_').str[3]
    # print(annealings)
    print(idf)

    annealing_types = ['ha', 'pea', 'ea']
    fig, axs = plt.subplots(1, len(annealing_types), gridspec_kw={'wspace': 0}, sharey=True, figsize=(20, 7))

    for a_i, annealing_type in enumerate(annealing_types):
        iidf = df[df['comments'].str.contains('annealing:' + annealing_type)]
        loss_curves = {k: [] for k in possible_pseudod}
        for _, row in iidf.iterrows():
            ptype = str2val(row['comments'], 'pseudod', str, equality_symbol='', remove_flag=False)
            if ptype in possible_pseudod:
                curve = histories[row['d_name']]['sparse_categorical_crossentropy']
                if len(curve) == 150:
                    loss_curves[ptype].append(curve)
                    print(len(curve), row['d_name'])
                    # axs[a_i].plot(curve, label='train xe', color=pseudod_color(ptype), linestyle=(0, (5, 3)),
                    #               linewidth=.5)

        for ptype in possible_pseudod:
            if len(loss_curves[ptype]) > 0:
                min_len = np.min([len(c) for c in loss_curves[ptype]])
                equalized_loss_curves = np.array([c[:min_len] for c in loss_curves[ptype]])
                mean = np.mean(equalized_loss_curves, axis=0)
                std = np.std(equalized_loss_curves, axis=0)
                axs[a_i].plot(mean, label='train xe', color=pseudod_color(ptype))
                axs[a_i].fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5, color=pseudod_color(ptype))
                print(ptype, annealing_type, min(mean))

        axs[a_i].set_ylim([.3, 3])
        axs[a_i].set_title(
            annealing_type.replace('ha', 'switch').replace('pea', 'probabilistic').replace('ea', 'weighted'))
        for pos in ['right', 'left', 'bottom', 'top']:
            axs[a_i].spines[pos].set_visible(False)

    axs[0].set_xlabel('training iteration')
    axs[0].set_ylabel('crossentropy')
    plot_filename = r'experiments/figure4_conventional2spike.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

    plt.show()

    idf['annealings'] = idf['comments'].str.split('_').str[2].str.replace('annealing:', '')
    idf['pseudod'] = idf['comments'].str.split('_').str[3]

    counts = idf.groupby(['annealings', 'pseudod']).size().reset_index(name='counts')
    print(counts)

elif args.type == 'move_folders':
    destination = os.path.join(GEXPERIMENTS, '2021-10-04--wordptb-improvements_with_BiGamma_initializer')

    df = df[df['where'].str.contains('gra')]  # cdr gra blg
    ds = df['d_name']
    print(ds)
    for path in ds:
        print('----------------------------')
        _, d = os.path.split(path)
        d = d + '.zip'
        origin = os.path.join(GEXPERIMENTS, d)
        dest = os.path.join(destination, d)
        print(origin)
        print(dest)
        os.rename(origin, dest)

elif args.type == 'receive_many_selected':
    list_ = [
        '2021-07-25--06-50-27--7050-mnl_',
        '2021-07-24--22-19-45--5723-mnl_',
    ]

    start_string = '/home/lucacehe/scratch/work/stochastic_spiking/experiments/'
    end_string = r' "C:\Users\PlasticDiscobolus\work\stochastic_spiking\good_experiments"'

    big_string = '.zip /home/lucacehe/scratch/work/stochastic_spiking/experiments/'.join(list_)
    string = 'scp -T lucacehe@cedar.computecanada.ca:' + '\"/home/lucacehe/scratch/work/stochastic_spiking/experiments/' + big_string + '.zip\"' + end_string
    print(string)

elif args.type == 'continue':
    already_sent = [
        '2021-07-31--16-31-10--4099-mnl_',
        '2021-07-31--15-58-59--4959-mnl_',
        '2021-07-31--16-31-10--7162-mnl_',
        '2021-07-31--15-54-52--7146-mnl_',
        '2021-07-31--15-35-34--5876-mnl_',
    ]
    # select to continue training
    # dnames = df[df["task_name"] == 'ptb']['d_name']
    nets = ['maLSNN', 'LSTM', 'customLSTM', 'spikingLSTM', 'spikingPerformer', 'smallGPT2', 'gravesLSTM']
    for net in nets:
        dnames = df[df["net_name"] == net]['d_name']
        # LSTM customLSTM spikingLSTM spikingPerformer smallGPT2
        for n in dnames:
            filename = n.split('\\')[-1]
            filename = filename.split('/')[-1]
            if filename not in already_sent:
                print("'{}',".format(filename))

elif args.type == 'pseudod':

    df = df[df['comments'].str.contains("pseudod")]
    # df = df[df['task_name'] == 'wordptb']

    type = 'sharpn'  # dampf sharpn
    df = df[df['comments'].str.contains(type)]

    df['comments'] = df['comments'].astype(str) + '_'

    # nets = np.unique(df['net_name'])

    nets = ['SNU', 'LSNN', 'sLSTM']
    tasks = ['wordptb', 'heidelberg', 'sl_mnist']
    comments = np.unique(df['comments'])
    pseudods_names = [[p for p in c.split('_') if 'pseudo' in p][0] for c in comments]
    pseudods_values = [[p.split(':')[1] for p in c.split('_') if type in p][0] for c in comments]

    pseudods_names = np.unique(pseudods_names)
    pseudods_values = [float(i) for i in np.unique(pseudods_values)]

    fig, axs = plt.subplots(len(tasks), len(nets), gridspec_kw={'wspace': .15})
    for j, task in enumerate(tasks):

        idf = df[df['task_name'] == task]
        if task == 'wordptb':
            metric = 'val_perplexity'
            ylims = [0, 500]
        elif task == 'heidelberg':
            metric = 'val_sparse_mode_accuracy'
            ylims = [0, 90]
        elif task == 's_mnist':
            metric = 'val_sparse_mode_accuracy'
            ylims = [0, 90]
        elif task == 'sl_mnist':
            metric = 'val_sparse_mode_accuracy'
            ylims = [0, 100]
        else:
            raise NotImplementedError

        for i, net in enumerate(nets):
            # plt.figure()
            if net == 'LSNN':
                small_df = idf[idf['net_name'] == 'maLSNN']
                small_df = small_df[small_df['comments'].str.contains("embproj_nolearnv0")]
            elif net == 'SNU':
                small_df = idf[idf['net_name'] == 'maLSNN']
                small_df = small_df[small_df['comments'].str.contains("noalif")]
            elif net == 'sLSTM':
                small_df = idf[idf['net_name'] == 'spikingLSTM']
            else:
                raise NotImplementedError

            for ptype in pseudods_names:
                bpcs = []
                for pvalue in pseudods_values:
                    row = small_df[small_df['comments'].str.contains(ptype)]
                    row = row[row['comments'].str.contains(str(pvalue))]
                    # assert row.shape[0] == 1
                    try:
                        metric_value = row.at[0, metric] * (100 if 'acc' in metric else 1)
                        bpcs.append(metric_value)
                    except:
                        bpcs.append(None)

                pseudods_values, bpcs = zip(*sorted(zip(pseudods_values, bpcs)))

                axs[j, i].plot(pseudods_values, bpcs, label=clean_pseudo_name(ptype), color=pseudod_color(ptype))

            study = 'sharpness' if type == 'sharpn' else 'dampening'
            axs[0, i].set_title(net)
            axs[j, i].set_ylim(ylims)

        axs[j, 0].set_ylabel(
            '{}\n{}'.format(task.replace('s_mnist', 'sMNIST'), metric.replace('val_', '').replace('sparse_mode_', '')))
    axs[-1, 0].set_xlabel(study)

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            for pos in ['right', 'left', 'bottom', 'top']:
                axs[j, i].spines[pos].set_visible(False)

            if j < axs.shape[0] - 1:
                axs[j, i].tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)

            if i > 0:
                axs[j, i].tick_params(
                    axis='y',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    right=False,  # ticks along the bottom edge are off
                    left=False,  # ticks along the top edge are off
                    labelleft=False)

    plot_filename = r'experiments/pseudods_{}.pdf'.format(study)
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()


elif args.type == 'interactive_histories':

    from bokeh.plotting import figure, output_notebook, show
    from bokeh.colors import RGB

    hv = [d['bpc'] for d in list(histories.values())]
    hk = list(histories.keys())

    fig = figure(title='Compare the Trends of Different Functions', width=1000, height=600)

    cm = plt.get_cmap('tab20')  # Reds tab20 gist_ncar
    colors = cm(np.linspace(1, 0, len(hk)))
    for i, (n, v) in enumerate(zip(hk, hv)):
        print(n)
        try:
            print('     ', v[18])
        except Exception as e:
            print(e)
        # print(v)
        color = RGB(*colors[i] * 255)
        fig.line(range(len(v)), v, legend_label=n, color=color, line_width=2)

    # Relocate Legend
    fig.legend.location = 'top_right'
    # Click to hide/show lines
    fig.legend.click_policy = 'hide'
    fig.legend.__setattr__('label_text_font_size', "6pt")

    show(fig)

elif args.type == 'histories':
    plot_filename = os.path.join(EXPERIMENTS, 'histories.png')
    histories = {k: v for k, v in histories.items() if '2022-01-02' in k}

    print(histories.keys())
    hv = list(histories.values())
    print(hv)
    hk = list(histories.keys())
    fig, axs = plot_history(
        histories=hv, plot_filename=None, epochs=1,
        method_names=hk, show=True, legend=False,
        metrics_to_show=['bpc']
    )

elif args.type == 'rhohistories':
    import matplotlib.pyplot as plt
    import copy
    import numpy as np

    restricted_history = copy.deepcopy({k: v for k, v in histories.items() if not 'caLSNN' in k})

    possible_nbetas = []
    for k in restricted_history.keys():
        nbeta = [i.replace('noisebeta:', '')
                 for i in k.split('_') if 'noisebeta:' in i][0]
        if not nbeta in ['none', 'learned']:
            possible_nbetas.append(float(nbeta))
            print(nbeta)

    possible_nbetas = sorted(set(possible_nbetas))
    print(possible_nbetas)
    # restricted_history = {k_: {k: v for k, v in v_.history.items() if 'val' in k} for k_, v_ in restricted_history.items()}
    keys_to_skip = ['loss', 'accuracy', 'contrastive_disorder', 'contrastive_random', 'perplexity', ]
    colors = []
    cm = plt.get_cmap('Reds')
    for k_, v_ in restricted_history.items():
        print(k_)
        if 'noisebeta:learned' in k_:
            nbeta = 'learned'
            color = 'green'
        elif 'noisebeta:none' in k_:
            nbeta = None
            color = 'blue'
        else:
            nbeta = float([i.replace('noisebeta:', '')
                           for i in k_.split('_') if 'noisebeta:' in i][0])
            # possible_nbetas = [0, -0.1, -0.5, -1, -1.5, -2]
            color = cm(np.linspace(.8, .3, len(possible_nbetas)))[possible_nbetas.index(nbeta)]

        colors.append(color)
        keys = list(v_.history.keys())
        for k in keys:
            if k in keys_to_skip:
                del restricted_history[k_].history[k]

        if '_ptb_' in k_:
            restricted_history[k_].history['loss'] = restricted_history[k_].history['bpc']
            restricted_history[k_].history['accuracy'] = restricted_history[k_].history['zeros_categorical_accuracy']
        else:
            restricted_history[k_].history['loss'] = restricted_history[k_].history['categorical_crossentropy']
            restricted_history[k_].history['accuracy'] = restricted_history[k_].history['mode_accuracy']

    plot_filename = os.path.join(EXPERIMENTS, 'figure_2.png')
    print(results['final_epochs'])
    column_id = ['ptb', '_s_mnist', '_ps_mnist', 'heidelberg']
    fig, axs = plot_history(
        histories=list(restricted_history.values()), plot_filename=plot_filename, epochs=float(results['final_epochs']),
        method_names=list(restricted_history.keys()), save=False, show=False,
        metrics_to_show=['loss', 'accuracy'],
        column_id=column_id, colors=colors
    )

    for column in range(len(column_id)):
        ax = axs[(0, column) if column_id else 0]
        ax.set_title(standardize_dataset_names(column_id[column]))

    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()


elif args.type == 'activities':

    k_selection = {'input_spikes': 'input', 'encoder_0_0': 'firing', 'encoder_0_0_1': 'v',
                   'encoder_0_0_2': 'thr', 'output_net': 'output_net', 'target_output': 'target_output'}

    all_activities = []
    titles = []
    for d in ds:
        d_path = os.path.join(EXPERIMENTS, d)
        activities_path = os.path.join(d_path, 'images', 'trained', 'png_content.dat')
        config_path = os.path.join(d_path, '1', 'config.json')

        with open(config_path) as f:
            config = json.load(f)

        if conditions_activities(config):
            print('-----------------------------------------')
            print(config['comments'])
            print(config['task_name'])
            print(config['net_name'])
            with open(activities_path, 'rb') as f:
                task = pickle.load(f)

            small_task = {k_selection[k]: task[k] for k in k_selection.keys()}
            # smart_plot(small_task)

            all_activities.append(small_task)
            titles.append(config['task_name'])
            # plt.show()

    ordered_titles = ['ptb', 's_mnist', 'ps_mnist', 'heidelberg']
    all_activities = [all_activities[titles.index(t)] for t in ordered_titles]
    titles = [titles[titles.index(t)] for t in ordered_titles]
    titles = []

    fig, axs = smart_plot(all_activities, clean=False, batch_sample=8)
    for column in range(len(all_activities)):
        ax = axs[(0, column) if titles else 0]
        if not len(titles) == 0:
            ax.set_title(standardize_dataset_names(titles[column]))

        for row in range(len(k_selection)):
            if not row == len(k_selection) - 1:
                axs[row, column].set_xticks([])

            if row % 2 == 0:
                axs[row, column].yaxis.tick_right()

    plot_filename = os.path.join(*['experiments', 'figure_2_activities.png'])

    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()

elif args.type == 'weights':
    task_name = 'heidelberg'  # 's_mnist'
    w_of_interest = ['input_weights', 'recurrent_weights', 'tau', 'tau_adaptation', 'thr', 'beta', 'n_std',
                     'conv1d_0/kernel', 'conv1d_0/bias', 'conv1d_1/kernel', 'conv1d_1/bias', 'decoder/kernel:0',
                     'decoder/bias'
                     ]

    plot_names = {'input_weights': r'$W_{in}$', 'recurrent_weights': r'$W_{rec}$', 'tau': r'$\tau_v$',
                  'tau_adaptation': r'$\tau_b$', 'thr': r'$b_0$', 'beta': r'$\beta$', 'n_std': r'$\sigma$',
                  'conv1d_0/kernel': r'$c_1$-$k$', 'conv1d_0/bias': r'$c_1$-$b$', 'conv1d_1/kernel': r'$c_2$-$k$',
                  'conv1d_1/bias': r'$c_2$-$b$', 'decoder/kernel:0': r'$l$-$k$', 'decoder/bias': r'$l$-$b$'
                  }

    # w_of_interest = ['input_weights', 'recurrent_weights', 'decoder/bias']
    # w_of_interest = ['conv1d_1/bias', 'decoder/kernel:0', 'decoder/bias']
    # w_of_interest = ['decoder/bias']
    # ['encoder_0_0/highdamp_a_lsnn_6/input_weights:0', 'encoder_0_0/highdamp_a_lsnn_6/recurrent_weights:0',
    # 'encoder_0_0/highdamp_a_lsnn_6/tau:0', 'encoder_0_0/highdamp_a_lsnn_6/tau_adaptation:0',
    # 'encoder_0_0/highdamp_a_lsnn_6/thr:0', 'encoder_0_0/highdamp_a_lsnn_6/beta:0',
    # 'encoder_0_0/highdamp_a_lsnn_6/n_std:0', 'conv1d_12/kernel:0', 'conv1d_12/bias:0',
    # 'conv1d_13/kernel:0', 'conv1d_13/bias:0', 'decoder/kernel:0', 'decoder/bias:0', 'total:0', 'count:0']

    filename = os.path.join(EXPERIMENTS, 'woi.pickle')

    if not os.path.isfile(filename):

        woi = {}
        for w_name in w_of_interest:

            ws = {}
            for d in ds:
                # break
                d_path = os.path.join(EXPERIMENTS, d)
                config_path = os.path.join(d_path, '1', 'config.json')

                with open(config_path) as f:
                    config = json.load(f)

                if conditions_weights(config, task_name):
                    print('-----------------------------')
                    print(config['task_name'])
                    print(config['comments'])
                    model_path = os.path.join(d_path, 'trained_models', 'train_model.h5')
                    model_args = ['task_name', 'net_name', 'n_neurons', 'tau', 'input_scaling', 'n_dt_per_step',
                                  'neutral_phase_length', 'reg_cost', 'lr', 'batch_size', 'stack', 'loss_name',
                                  'embedding', 'skip_inout', 'spike_dropout', 'spike_dropin', 'optimizer_name',
                                  'lr_schedule', 'weight_decay', 'clipnorm', 'initializer', 'comments']
                    kwargs = {k: config[k] for k in model_args}

                    # task definition
                    maxlen = 100
                    if 'maxlen:' in config['comments']:
                        maxlen = int(
                            [s for s in config['comments'].split('_') if 'maxlen:' in s][0].replace('maxlen:', ''))
                    gen_val = Task(n_dt_per_step=config['n_dt_per_step'], batch_size=config['batch_size'],
                                   steps_per_epoch=config['steps_per_epoch'], category_coding=config['category_coding'],
                                   name=config['task_name'], train_val_test='val',
                                   neutral_phase_length=config['neutral_phase_length'], maxlen=maxlen)

                    final_epochs = gen_val.epochs
                    final_steps_per_epoch = gen_val.steps_per_epoch
                    tau_adaptation = int(gen_val.in_len / 2)  # 200 800 4000
                    kwargs.update(
                        {'in_len': gen_val.in_len, 'n_in': gen_val.in_dim, 'out_len': gen_val.out_len,
                         'n_out': gen_val.out_dim,
                         'tau_adaptation': tau_adaptation, 'final_epochs': gen_val.epochs,
                         'final_steps_per_epoch': gen_val.steps_per_epoch})

                    train_model = build_model(**kwargs)
                    train_model.load_weights(model_path)
                    w = 0

                    w_names = [copy.deepcopy(w.name) for w in train_model.weights]
                    conv_names = sorted(set([wn.split('/')[0] for wn in w_names if 'conv' in wn]))

                    conv_names = {k: 'conv1d_{}'.format(i) for i, k in enumerate(conv_names)}

                    for weight in train_model.weights:
                        wn = copy.deepcopy(weight.name)

                        if 'conv' in wn:

                            for k in conv_names.keys():
                                if wn.split('/')[0] == k:
                                    wn = wn.replace(k, conv_names[k])

                        if w_name in wn:
                            w = weight.numpy()
                            # w = np.random.rand(30, 30)

                    nbeta = [x for x in config['comments'].split('_') if 'noisebeta' in x][0].replace('noisebeta:', '')
                    nbeta = float(nbeta) if not nbeta in ['learned', 'none'] else nbeta
                    if nbeta == 'learned':
                        nbeta = 111
                    elif nbeta == 'none':
                        nbeta = 222

                    ws[nbeta] = w

            ws = dict(sorted(ws.items()))
            ws = {str(k).replace('111', 'learned').replace('222', 'none'): v for k, v in ws.items()}
            woi[w_name] = ws

        with open(filename, 'wb') as handle:
            pickle.dump(woi, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(filename, 'rb') as handle:
            woi = pickle.load(handle)

    from scipy.stats import gaussian_kde

    fig, axs = plt.subplots(len(woi), len(woi[list(woi.keys())[0]]), figsize=(20, 10),
                            gridspec_kw={'hspace': 0.7, 'wspace': 0.0})

    for row, (w_name, ws) in tqdm(enumerate(woi.items())):

        print(w_name)
        if w_name in w_of_interest:
            ps = []
            for column, k in enumerate(ws.keys()):
                ax = axs[(row, column) if len(woi) > 1 else column]

                fw = ws[k].flatten()
                ax.hist(fw, bins=50, alpha=.4, density=True)

                prob_density = gaussian_kde(fw)
                x_fine = np.linspace(np.min(fw), np.max(fw), 100)
                probs = prob_density(x_fine)
                ax.plot(x_fine, probs, 'b', '.', linewidth=1)

                if row == 0:
                    ax.set_title(r'$\rho={}$'.format(k))

                for j, k2 in enumerate(ws.keys()):
                    if j > column and not k == 'learned' and not k2 == 'learned':
                        _, p = mannwhitneyu(fw, ws[k2].flatten())
                        ps.append(p)

                # plt.colorbar()
                p25, p50, p75 = np.percentile(fw, [5, 50, 95], axis=0)
                ax.set_xticks([p25, p50, p75])
                x_amplitude = p75 - p25
                ax.set_xlim(p25 - 0.2 * x_amplitude, p75 + 0.2 * x_amplitude)
                # ax.set_ylim(0., 0.12)

                if not column == 0:
                    ax.set_yticks([])
                # else:
                #     ax.set_yticks([0., 0.1])

            ax = axs[(row, 0) if len(woi) > 1 else 0]
            ax.set_ylabel('{}\n{}'.format(plot_names[w_name], significance_to_star(np.min(ps))))

    fig.align_ylabels(axs[:])

    time_string = timeStructured()
    plot_filename = os.path.join(*['experiments', '{}_figure_weights_p.png'.format(time_string)])
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()

elif args.type == 'robustness':
    # pass
    # print(ds[:1])
    task_name = 'heidelberg'
    filename = os.path.join(EXPERIMENTS, 'robustness.json')

    if not os.path.isfile(filename):
        all_to_plot = {}
        for d in ds:
            # break
            d_path = os.path.join(EXPERIMENTS, d)
            config_path = os.path.join(d_path, '1', 'config.json')

            with open(config_path) as f:
                config = json.load(f)

            if conditions_weights(config, task_name):
                print('-----------------------------')
                print(config['task_name'])
                print(config['comments'])
                model_path = os.path.join(d_path, 'trained_models', 'train_model.h5')
                model_args = ['task_name', 'net_name', 'n_neurons', 'tau', 'input_scaling', 'n_dt_per_step',
                              'neutral_phase_length', 'reg_cost', 'lr', 'batch_size', 'stack', 'loss_name',
                              'embedding', 'skip_inout', 'spike_dropout', 'spike_dropin', 'optimizer_name',
                              'lr_schedule', 'weight_decay', 'clipnorm', 'initializer', 'comments']
                kwargs = {k: config[k] for k in model_args}

                # task definition
                maxlen = 100
                if 'maxlen:' in config['comments']:
                    maxlen = int([s for s in config['comments'].split('_') if 'maxlen:' in s][0].replace('maxlen:', ''))

                steps_per_epoch = 4  # config['steps_per_epoch']
                gen_val = Task(n_dt_per_step=config['n_dt_per_step'], batch_size=config['batch_size'],
                               steps_per_epoch=steps_per_epoch, category_coding=config['category_coding'],
                               name=config['task_name'], train_val_test='val',
                               neutral_phase_length=config['neutral_phase_length'], maxlen=maxlen)

                final_epochs = gen_val.epochs
                final_steps_per_epoch = gen_val.steps_per_epoch
                tau_adaptation = int(gen_val.in_len / 2)  # 200 800 4000
                kwargs.update(
                    {'in_len': gen_val.in_len, 'n_in': gen_val.in_dim, 'out_len': gen_val.out_len,
                     'n_out': gen_val.out_dim,
                     'tau_adaptation': tau_adaptation, 'final_epochs': gen_val.epochs,
                     'final_steps_per_epoch': gen_val.steps_per_epoch})

                train_model = build_model(**kwargs)
                w_names = [copy.deepcopy(w.name) for w in train_model.weights]

                # evaluation = train_model.evaluate(gen_val, return_dict=True)
                # print(evaluation)
                train_model.load_weights(model_path)
                # evaluation = train_model.evaluate(gen_val, return_dict=True)
                # print(evaluation)

                names = [weight.name for layer in train_model.layers for weight in layer.weights]
                rec_name = [n for n in names if 'recurrent' in n][0]
                weights = train_model.get_weights()

                original_rec_w = weights[names.index(rec_name)]
                evaluations = {}
                for n_mask in [0., .2, .4, .6, .8, 1.]:
                    mask = np.random.choice([0, 1], size=original_rec_w.shape, p=[n_mask, 1 - n_mask])
                    weights[names.index(rec_name)] = original_rec_w * mask
                    train_model.set_weights(weights)

                    evaluation = train_model.evaluate(gen_val, return_dict=True)
                    evaluations[n_mask] = evaluation

                all_to_plot[config['comments']] = evaluations

        json.dump(all_to_plot, open(filename, "w"))
    else:
        with open(filename) as f:
            all_to_plot = json.load(f)

    print(all_to_plot)
    metric = 'mode_accuracy'
    plt.figure()
    for k in all_to_plot.keys():
        evaluations = all_to_plot[k]
        p_mask = evaluations.keys()
        performances = [evaluations[m][metric] for m in p_mask]

        plt.plot(p_mask, performances, label=k)

    plt.legend()

    plot_filename = os.path.join(*['experiments', '{}_figure_robustness.png'.format(timeStructured())])
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()



else:

    raise NotImplementedError

print('DONE')
