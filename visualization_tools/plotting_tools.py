import matplotlib.pyplot as plt
import logging, os, glob, imageio
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from matplotlib.ticker import MaxNLocator

from GenericTools.stay_organized.plot_tricks import large_num_to_reasonable_string
from GenericTools.keras_tools.esoteric_tasks.ptb_get_statistics import get_probabilities

logger = logging.getLogger('mylogger')

rect = lambda color: plt.Rectangle((0, 0), 1, 1, color=color)


def plot_and_write(aux, metrics, n_rec,
                   n_input, fixed_input, target, k_iter,
                   dt, losses, pathplot):
    # plt.ion()
    fig, axes = plt.subplots(8, figsize=(6, 10), sharex=True)

    reinforce_loss_metric, reparam_loss_metric, hard_loss_metric, cond_soft_loss_metric, soft_loss_metric, baseline_loss_metric = losses
    date_str = dt.datetime.now().strftime('%H:%M:%S %d-%m-%y')
    logger.info(f'summary {k_iter} @ {date_str}')
    logger.info(f'  - reinforce       {reinforce_loss_metric.result():.4f}')
    logger.info(f'  - reparam         {reparam_loss_metric.result():.4f}')
    logger.info(f'  - baseline        {baseline_loss_metric.result():.4f}')
    logger.info('')
    logger.info(f'  - hard      {hard_loss_metric.result():.4f}')
    logger.info(f'  - cond soft {cond_soft_loss_metric.result():.4f}')
    logger.info(f'  - soft      {soft_loss_metric.result():.4f}')
    logger.info('')
    mean_rate_hard = aux['mean_hard_rate']
    mean_rate_soft = aux['mean_soft_rate']
    reg_loss = aux['reg_loss'].numpy().mean()
    logger.info(f'  - reg loss     {reg_loss:.3f}')
    logger.info(f'  - rate (hard)  {mean_rate_hard:.1f}')
    logger.info(f'  - rate (soft)  {mean_rate_soft:.1f}')
    [m.reset_states() for m in metrics]
    pred = aux['pred']
    out = aux['out']
    hard_pred, soft_cond_pred, soft_pred = pred
    hard_b, soft_cond_b, soft_b = out
    [ax.clear() for ax in axes]
    ax = axes[0]
    ax.pcolormesh(fixed_input[0].T, cmap='Greys', vmin=0, vmax=1)
    ax.set_ylabel('input')
    ax.set_yticks([0, n_input])

    ax = axes[1]
    ax.pcolormesh(hard_b[0].numpy().T, cmap='Greys', vmin=0, vmax=1)
    ax.set_ylabel('hard')
    ax.set_yticks([0, n_rec])

    ax = axes[2]
    ax.pcolormesh(soft_cond_b[0].numpy().T, cmap='Greys', vmin=0, vmax=1)
    ax.set_ylabel('conditioned\nsoft')
    ax.set_yticks([0, n_rec])

    ax = axes[3]
    ax.pcolormesh(soft_b[0].numpy().T, cmap='Greys', vmin=0, vmax=1)
    ax.set_ylabel('soft')
    ax.set_yticks([0, n_rec])

    ax = axes[4]
    ax.plot(aux['v'][0].numpy(), color='b', lw=.5, alpha=.2)
    ax.set_ylabel('membrane\nvoltage')

    ax = axes[5]
    ax.plot(aux['p'][0].numpy(), color='b', lw=.5, alpha=.2)
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 1])
    ax.set_ylabel('spike\nprobabilities')

    ax = axes[6]
    ax.plot(target[0], 'k--', alpha=.7, label='target')
    ax.plot(hard_pred[0, :, 0].numpy(), 'b', alpha=.7, label='hard')
    ax.plot(soft_cond_pred[0, :, 0].numpy(), 'orange', alpha=.7, label='soft conditioned')
    ax.plot(soft_pred[0, :, 0].numpy(), 'r', alpha=.7, label='soft')
    ax.legend(frameon=False)
    ax.set_ylabel('output')

    ax = axes[7]
    ax.plot(aux['baseline_target'][0, :].numpy(), 'k--', alpha=.7, lw=2, label='target')
    ax.plot(aux['baseline'][0, :].numpy(), 'b', alpha=.7, lw=2, label='prediction')
    ax.legend(frameon=False)
    ax.set_ylabel('value')

    [ax.yaxis.set_label_coords(-.09, .5) for ax in axes]

    if k_iter == 0:
        fig.tight_layout()

    plt.draw()
    plt.pause(.1)

    plt.savefig(pathplot)


def small_plot(nt, batch_x, batch_y, probs_x=None, probs_y=None, pathplot=''):
    b, v, p, a, v_sc, output_net, exp_conv = nt

    N = batch_y.shape[-1]
    reds = plt.cm.Reds(np.linspace(.5, .8, N))
    blues = plt.cm.Blues(np.linspace(.2, .7, N))

    fig, axes = plt.subplots(9, figsize=(6, 10), sharex='all',
                             gridspec_kw={'hspace': 0})
    [ax.clear() for ax in axes]

    ax = axes[0]
    ax.pcolormesh(batch_x[0].T, cmap='Greys', vmin=0, vmax=1, label='input')
    if not probs_x is None:
        width = 1.1
        x = np.arange(len(probs_x[0])) + width / 2 - .1
        ax.bar(x, np.squeeze(probs_x[0]), width, color='orange', alpha=.9, label='input prob')

        ax.legend([rect("black"), rect("orange")],
                  ["input", "input prob"],
                  loc='upper left', prop={'size': 6})
    else:
        ax.legend([rect("black")],
                  ["input"],
                  loc='upper left', prop={'size': 6})

    ax.set_ylabel('input')
    ax.set_yticks([0, 1])

    ax = axes[1]
    ax.pcolormesh(batch_y[0].T, cmap='Greys', vmin=0, vmax=1, label="target output")
    ax.legend([rect("black")], ["target output"],
              loc='upper left', prop={'size': 6})
    ax.set_ylabel('target output')

    ax = axes[2]
    ax.pcolormesh(output_net[0].T, cmap='Reds', vmin=0, vmax=1, label='output_net')
    ax.legend([rect("r")], ["softmax(exp_conv)"],
              loc='upper left', prop={'size': 6})
    ax.set_ylabel('network output')

    if not probs_y is None:
        width = 1.1
        x = np.arange(len(probs_y[0])) + width / 2 - .1
        ax.bar(x, np.squeeze(probs_y[0]), width, color='orange', alpha=.9, label='input prob')
        ax.legend([rect("black"), rect("orange"), rect("r")],
                  ["output", "output prob", "output net"],
                  loc='upper left', prop={'size': 6})
    else:
        pass

    ax.set_ylabel('output')
    ax.set_yticks([0, 1])

    ax = axes[3]
    ax.pcolormesh(b[0].T, cmap='Greys', vmin=0, vmax=1)
    ax.set_ylabel('b')
    # ax.set_yticks([0, n_rec])

    ax = axes[4]
    ax.pcolormesh(p[0].T, cmap='Greys', vmin=0, vmax=1)
    ax.set_ylabel('p')
    # ax.set_yticks([0, n_rec])

    ax = axes[5]
    ax.plot(v[0], color='g', lw=.5, alpha=.2)
    ax.set_ylabel('v')
    # ax.set_yticks([0, n_rec])

    ax = axes[6]
    ax.plot(a[0], color='b', lw=.5, alpha=.2)
    ax.set_ylabel('a')

    ax = axes[7]
    ax.plot(v_sc[0], color='g', lw=.5, alpha=.2)
    ax.set_ylabel('v_sc')

    ax = axes[8]
    for i in range(N):
        ax.plot(output_net[0, :, i], color=reds[i], lw=1.5, alpha=1.)
        ax.plot(exp_conv[0, :, i], color=blues[i], lw=1.5, alpha=1.)
    ax.set_ylim([0, 1])
    # ax.set_yticks([0, 1])
    ax.set_ylabel('output_net')

    ax.legend([rect("blue"), rect("r")],
              ["exp_conv", "softmax(exp_conv)"],
              loc='upper left', prop={'size': 6})

    fig.tight_layout()

    ax.set_xlabel('time')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(pathplot, bbox_inches='tight')

    plt.close('all')
    plt.close(fig)


def plot_batch_and_prediction(np_arrays, plot_filename):
    fig, axs = plt.subplots(len(np_arrays), figsize=(8, 8))

    fig.suptitle(plot_filename)

    for array, ax in zip(np_arrays, axs):
        # plot training and validation losses
        ax.plot(array)
        # ax.plot(history.history['val_' + k], label='val ' + k)
        # ax.set_ylabel(k)
        ax.set_xlabel('epoch')
        ax.legend()

    fig.savefig(plot_filename, bbox_inches='tight')


def generate_and_save_images(model, epoch, test_input, destination_path=''):
    if not model == None:
        predictions = model.sample(test_input)
        plotname = 'gif_image_generation_at_epoch_{:04d}.png'.format(epoch)
    else:
        predictions = test_input
        plotname = 'generation_input.png'.format(epoch)

    fig, axs = plt.subplots(4, 4, figsize=(8, 8), sharex='all', sharey='all',
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle('epoch {}'.format(epoch), fontsize=12)
    for i, ax in enumerate(axs.ravel()):
        ax.pcolormesh(tf.transpose(predictions[i], [1, 0]), cmap='gray')
        # ax.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plotpath = os.path.join(destination_path, plotname)
    fig.savefig(plotpath)
    plt.close('all')


def prediction_and_save_images(model, epoch, test_input, test_output, destination_path=''):
    if model == 'input':
        predictions = test_input
        plotname = 'prediction_input.png'.format(epoch)
    elif model == 'output':
        predictions = test_output
        plotname = 'prediction_output.png'.format(epoch)
    else:
        predictions = model.predict(test_input)
        plotname = 'gif_image_prediction_at_epoch_{:04d}.png'.format(epoch)

    fig, axs = plt.subplots(4, 4, figsize=(8, 8), sharex='all', sharey='all',
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle('epoch {}'.format(epoch), fontsize=12)
    for i, ax in enumerate(axs.ravel()):
        ax.pcolormesh(tf.transpose(predictions[i], [1, 0]), cmap='gray')
        # ax.pcolormesh(tf.transpose(test_output[i], [1, 0]), cmap='Oranges')
        # ax.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plotpath = os.path.join(destination_path, plotname)
    fig.savefig(plotpath)


def img2gif(destination_path, name_clue=''):
    anim_file = os.path.join(destination_path, 'movie_{}.gif'.format(name_clue))

    with imageio.get_writer(anim_file, mode='I', duration=.3) as writer:
        filenames = glob.glob(os.path.join(destination_path, 'gif_image_{}*.png'.format(name_clue)))
        filenames = sorted(filenames)

        min_imgs = min(len(filenames), 30)
        idx = np.round(np.linspace(0, len(filenames) - 1, min_imgs)).astype(int)
        filenames = np.array(filenames)[idx].tolist()
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


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
