import os
import tensorflow as tf
from pyaromatics.keras_tools.esoteric_layers.surrogated_step import possible_pseudod, clean_pseudname, \
    clean_pseudo_name, pseudod_color, ChoosePseudoHeaviside

# create convenient folders for all experiments
CDIR = os.path.dirname(os.path.realpath(__file__))
EXPERIMENTS = os.path.join(CDIR, 'experiments')

def draw_pseudods():
    import numpy as np
    import matplotlib as mpl
    from pyaromatics.stay_organized.mpl_tools import load_plot_settings

    # mpl.rcParams['font.family'] = 'serif'
    label_fotsize = 18
    linewidth = 3
    mpl = load_plot_settings(mpl)

    import matplotlib.pyplot as plt
    # figsize=(10, 5)
    fig, axs = plt.subplots(1, 2, gridspec_kw={'wspace': .1}, sharey=False, figsize=(8, 5))

    for k in possible_pseudod:
        x = tf.cast(tf.constant(np.linspace(0, 1.5, 1000)), tf.float32)
        # x = tf.cast(tf.constant(np.linspace(-1.5, 1.5, 1000)), tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = ChoosePseudoHeaviside(x, k + '_sharpn:1')
        grad = tape.gradient(y, x)
        print(k)
        print(np.mean(grad) * 4)

        c = pseudod_color(k)
        print(c)
        cint = (int(255 * i) for i in c)
        print(cint)
        print(k, '#{:02x}{:02x}{:02x}'.format(*cint))
        axs[0].plot(x, grad, color=c, label=clean_pseudo_name(k), linewidth=linewidth)

    n_exps = 7
    exponents = 10 ** np.linspace(-2, 1.2, n_exps) + 1

    cm = plt.get_cmap('Oranges')
    for i, k in enumerate(exponents):
        c = cm(.4 + i / (len(exponents) - 1) * .4)
        x = tf.cast(tf.constant(np.linspace(0, 1.5, 1000)), tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = ChoosePseudoHeaviside(x, 'ntailpseudod_tailvalue:' + str(k))
        grad = tape.gradient(y, x)

        print(c)
        cint = (int(255 * i) for i in c)
        print(cint)
        print(k, '#{:02x}{:02x}{:02x}'.format(*cint))
        axs[1].plot(x, grad, color=c, linewidth=linewidth)

    n_grad = 100
    gradient = np.linspace(.4, .8, n_grad)[::-1]
    gradient = np.vstack((gradient, gradient))

    ax = fig.add_axes([.90, 0.2, .015, .6])
    ax.imshow(gradient.T, aspect='auto', cmap=cm)
    # ax.text(-0.01, 0.5, '$q$-PseudoSpike \t', va='center', ha='right', fontsize=16, transform=ax.transAxes)
    ax.text(12, 0.5, '$q$ \t', va='center', ha='right', fontsize=label_fotsize, transform=ax.transAxes)
    ax.set_yticks([])
    ax.set_xticks([])

    loc = [-.5 + i / (n_exps - 1) * n_grad for i in range(n_exps)]
    exponents = [round(e, 2) for e in 10 ** np.linspace(-2, 1.2, n_exps) + 1][::-1]

    ax.set_yticks(loc)
    ax.set_yticklabels(exponents, fontsize=label_fotsize * .85)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    for pos in ['right', 'left', 'bottom', 'top']:
        ax.spines[pos].set_visible(False)

    # axs[0].set_xlabel('centered voltage')
    axs[0].set_ylabel('Surrogate gradient\namplitude', fontsize=label_fotsize)
    axs[1].set_xlabel('Centered voltage', fontsize=label_fotsize)
    # axs[1].set_ylabel('surrogate gradient\namplitude')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=pseudod_color(n), lw=4, label=clean_pseudname(n))
                       for n in [possible_pseudod[-1]] + possible_pseudod[:-1]]
    # axs[0].legend(handles=legend_elements, loc='best', bbox_to_anchor=(0.4, 0.5, 0.4, 0.5))
    for ax in axs:
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

        ax.tick_params(axis='both', which='major', labelsize=label_fotsize * .75)

    axs[0].set_xticks([0, 1])
    axs[1].set_xticks([0, 1])
    axs[0].set_yticks([0, 1])
    axs[1].set_yticks([])

    plot_filename = os.path.join(EXPERIMENTS, 'pseudods.pdf')
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()


def clean_pseudname(name):
    name = name.replace('pseudod', '').replace('original', 'triangular')
    name = name.replace('fastsigmoid', '$\partial$ fast sigmoid')
    name = name.replace('sigmoidal', '$\partial$ sigmoid')
    name = name.replace('cappedskip', 'rectangular')
    name = name.replace('ntail', '$q$-PseudoSpike')
    return name


def draw_legend():
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from pyaromatics.stay_organized.mpl_tools import load_plot_settings
    mpl = load_plot_settings(mpl=mpl)

    ncol = 3

    pseudods = [possible_pseudod[-1]] + possible_pseudod[:-1] + ['ntailpseudod']
    pseudods = [
        'cappedskippseudod',
        'originalpseudod',
        'ntailpseudod',
        'exponentialpseudod',
        'gaussianpseudod',
        'sigmoidalpseudod',
        'fastsigmoidpseudod',
    ]
    legend_elements = [Line2D([0], [0], color=pseudod_color(n), lw=4, label=clean_pseudname(n))
                       for n in pseudods]

    # Create the figure
    fig, ax = plt.subplots(figsize=(3.5, .5))
    for pos in ['right', 'left', 'bottom', 'top']:
        ax.spines[pos].set_visible(False)
    # ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
    #                 labelleft='off')

    ax.legend(ncol=ncol, handles=legend_elements, loc='center')

    ax.axis('off')
    ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                   labelright='off', labelbottom='off')
    # plt.tight_layout(pad=0)
    plt.tight_layout(pad=0., w_pad=0., h_pad=0.)

    plot_filename = rf'legend_cols{ncol}.pdf'
    # fig.tight_layout(pad=0)
    fig.savefig(plot_filename, bbox_inches='tight', pad_inches=0)
    # fig.savefig(plot_filename)
    plt.show()


def draw_legend_mini():
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from pyaromatics.stay_organized.mpl_tools import load_plot_settings
    mpl = load_plot_settings(mpl=mpl)

    legend_elements = [
        Line2D([0], [0], color='w', lw=4, label='4 seeds'),
        Line2D([0], [0], color=pseudod_color('originalpseudod'), lw=4, label='mean'),
        Line2D([0], [0], color=pseudod_color('originalpseudod'), lw=16, label='std', alpha=0.5),
    ]

    # Create the figure
    fig, ax = plt.subplots(figsize=(3, 3))
    for pos in ['right', 'left', 'bottom', 'top']:
        ax.spines[pos].set_visible(False)
    # ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
    #                 labelleft='off')

    ax.legend(handles=legend_elements, loc='center', frameon=False)

    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                    labelright='off', labelbottom='off')

    plot_filename = r'meanstd.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_pseudods()
