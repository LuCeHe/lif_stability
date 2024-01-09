import matplotlib.pyplot as plt
import logging, os, glob, imageio
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from pyaromatics.stay_organized.plot_tricks import large_num_to_reasonable_string
from pyaromatics.keras_tools.esoteric_tasks.ptb_get_statistics import get_probabilities

logger = logging.getLogger('mylogger')

rect = lambda color: plt.Rectangle((0, 0), 1, 1, color=color)



def smart_plot(nts, pathplot=None, batch_sample=0, clean=True, show=False):
    if not isinstance(nts, list): nts = [nts]

    if len(nts) == 1:
        fig, axs = plt.subplots(len(nts[0]), figsize=(6, 15), sharex=True, gridspec_kw={'hspace': 0})
    else:
        fig, axs = plt.subplots(len(nts[0]), len(nts), figsize=(20, 5), gridspec_kw={'hspace': 0})

    for column, nt in enumerate(nts):
        for row, k in enumerate(nt.keys()):
            ax = axs[row] if len(nts) == 1 else axs[row, column]
            ax.clear()
            y = nt[k][batch_sample]
            if 'encoder' in k:
                if k.count('_') == 2:
                    k = 'firing_' + k[-1]
                elif k.count('_') == 1:
                    flag = ['', 'voltage', 'thr', 'v_sc'][int(k[-1])]
                    k = flag + '_' + k[-3]
            try:
                if k == 'a' or 'thr' in k:
                    ax.plot(range(len(y)), y, color='b', lw=.5, alpha=.2)
                    ax.set_xlim(0, len(y))
                elif 'v' == k or 'v_sc' in k or 'voltage' in k:
                    ax.plot(range(len(y)), y, color='g', lw=.5, alpha=.2)
                    ax.set_xlim(0, len(y))
                elif 'grad' in k:
                    ax.plot(range(len(y)), y, color='g', lw=.5, alpha=.2)
                    ax.set_xlim(0, len(y))
                elif 'rate' in k:
                    ax.plot(range(len(y)), y, color='orange', lw=.5, alpha=.2)
                    ax.set_xlim(0, len(y))
                elif k in ['output_net']:
                    c = 'Reds'
                    im = ax.pcolormesh(y.T, cmap=c)
                else:
                    if len(y.shape) == 1:
                        vocab_size = nt['target_output'][batch_sample].shape[-1]
                        y = tf.one_hot(np.squeeze(y), vocab_size, ).numpy()
                    c = 'Greys'
                    im = ax.pcolormesh(y.T, cmap=c)
                    # fig.colorbar(im, ax=ax)
                if column == 0:
                    ax.set_ylabel(k.replace('_', '\n'))
            except Exception as e:
                print(e)
                ax.set_ylabel(k + '\nERROR')

        ax.set_xlabel('t')

    fig.align_ylabels(axs[:])

    if not pathplot is None:
        fig.savefig(pathplot, bbox_inches='tight')

    if show:
        plt.show()

    if clean:
        plt.close('all')
        plt.close(fig)

    return fig, axs


CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(*[CDIR, r'../data/simple-examples/data/']))
probs_filepath = DATAPATH + '/prior_prob_ptb.npy'


def bpc_prior(gen):
    if not os.path.isfile(probs_filepath):
        get_probabilities()

    p = np.load(probs_filepath)

    mean_bpc = 0
    for _ in tqdm(range(gen.steps_per_epoch)):
        batch = gen.data_generation()
        y = batch['target_output']

        batch_size = y.shape[0]
        time_steps = y.shape[1]
        probabilities = p[None, None, ...].astype(float)
        probabilities = np.repeat(np.repeat(probabilities, batch_size, axis=0), time_steps, axis=1)
        xent = tf.keras.losses.CategoricalCrossentropy()(y, probabilities)
        bits_per_character = xent / np.log(2)
        mean_bpc += bits_per_character

    mean_bpc /= gen.steps_per_epoch

    try:
        sess = tf.compat.v1.Session()
        bpc = sess.run(mean_bpc)
        logger.warning('bpc given the prior: {}'.format(bpc))
    except Exception as e:
        logger.warning(e)
        logger.warning('bpc given the prior: {}'.format(mean_bpc))


def conditions_activities(config):
    c1 = 'noisebeta:0' in config['comments']
    c2 = 'highdamp_aLSNN' in config['net_name']
    c3 = config['n_neurons'] == 1000
    c4 = not 'conductance' in config['comments']
    c5 = not 'thrbeta' in config['comments']

    all = np.all([c1, c2, c3, c4, c5])
    return all


def conditions_weights(config, task_name='s_mnist'):
    c1 = task_name == config['task_name']
    c2 = 'highdamp_aLSNN' in config['net_name']
    c3 = config['n_neurons'] == 1000
    all = np.all([c1, c2, c3])
    return all


def standardize_dataset_names(initial_name):
    name = ''
    if initial_name == 'ptb':
        name = 'PTB'
    elif initial_name == 'heidelberg':
        name = 'SHD'
    elif initial_name in ['_s_mnist', 's_mnist']:
        name = 'S-MNIST'
    elif initial_name in ['_ps_mnist', 'ps_mnist']:
        name = 'PS-MNIST'
    return name


def postprocess_results(k, v):
    if k == 'n_params':
        v = large_num_to_reasonable_string(v, 1)
    return v
