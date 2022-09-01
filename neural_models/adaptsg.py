import os
import tensorflow as tf
import numpy as np
from scipy import special as sp
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from GenericTools.keras_tools.esoteric_layers.surrogated_step import ChoosePseudoHeaviside
from GenericTools.keras_tools.esoteric_tasks.time_task_redirection import Task

from sg_design_lif.neural_models.full_model import build_model
from sg_design_lif.visualization_tools.training_tests import get_test_model

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.abspath(os.path.join(CDIR, '..', 'experiments'))


def doubleexp(x, a, b, c, d, e, f, g, h, i, l, m):
    a, b, c, d, e, f, g, h, i, l, m = abs(a), abs(b), abs(c), abs(d), abs(e), abs(f), abs(g), abs(h), abs(i), abs(l), m
    x = x + m
    return \
        (a * tf.exp(b * x) + e * 1 / (1 + f * tf.abs(x) ** (1 + i))) * np.heaviside(-x, .5) \
        + (c * tf.exp(-d * x) + h * 1 / (1 + g * tf.abs(x) ** (1 + l))) * np.heaviside(x, .5)  # one of best so far

def movedgauss(x, a, b, c, d, e, f, g, h, i, l, m):
    a, b, c, d, e, f, g, h, i, l, m = abs(a), abs(b), abs(c), abs(d), abs(e), abs(f), abs(g), abs(h), abs(i), abs(l), m
    x = x + m
    return a * tf.exp(-b * x**2)

def get_shape(comments):
    if 'doubleexp' in comments:
        asgshape = doubleexp
    elif 'movedgauss':
        asgshape = movedgauss
    else:
        asgshape = movedgauss

    return asgshape

def adapt_sg_shape(data_generator, model, comments, test=False):
    (tin, tout), = data_generator.__getitem__()

    test_model = get_test_model(model)
    trt = test_model.predict([tin, tout], batch_size=tin.shape[0])
    trt = {name: pred for name, pred in zip(test_model.output_names, trt)}

    activity_names = [k for k in trt.keys() if k.endswith('_3') and k.startswith('encoder')]
    all_bins = []
    all_ns = []
    popts = []
    for i, k in enumerate(activity_names):
        cv = trt[k].flatten()

        n, bins, patches = plt.hist(cv, 1000, density=True, facecolor='g', alpha=0.5)

        all_bins.append(bins[:-1])
        all_ns.append(n)
        try:
            # moved_gauss
            shape_func = get_shape(comments)
            popt, _ = curve_fit(shape_func, bins[:-1], n, p0=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, .5, .5, .5, .5, .0))
            print('\n\n            Adapted SG!\n\n')
        except Exception as e:
            print('\n\n            Non Adapted SG error!\n\n')
            print(e)
            popt = [
                1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 0
            ]

        popts.append(popt)
        comments += '_eppseudod'
        for l, v in zip(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm'], popt):
            comments += f'_{l}{i}:{v}'
    if not test:
        return comments
    else:
        return comments, all_bins, all_ns, popts


def test_adaptsg():
    params = [-0.20201285957912918, 2.5118566166639584, -0.6673356614771232, 1.2510880363762846,
              -0.049519468893990996,
              0.004120122160092256, 2.0072643043174256, 0.8743848390304889, 1.5061652880205298, 1.1392061683504318,
              0.30571907616976585]

    config = 'eppseudod'
    for l, v in zip(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm'], params):
        config += f'_{l}:{v}'
    print(config)
    v_sc = tf.random.uniform((2, 3))
    ChoosePseudoHeaviside(v_sc, config=config, sharpness=1, dampening=1)


def test_adapt_sg_shape():
    import time
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    EXPERIMENT = os.path.join(EXPERIMENTS, time_string + random_string + '_adapt')
    MODL = os.path.join(EXPERIMENT, 'trained_models')

    for d in [EXPERIMENT, MODL]:
        os.makedirs(d, exist_ok=True)
    os.makedirs(EXPERIMENT, exist_ok=True)

    task_name = 'sl_mnist'
    timerepeat = 2
    epochs = 1
    comments = '8_noalif_nogradreset_dropout:.3_timerepeat:2_'
    n_neurons = 256
    stack = 4
    batch_size = 32
    loss_name = 'sparse_categorical_crossentropy'
    embedding = None  if not 'ptb' in task_name else f'learned:None:None:{int(n_neurons/3)}'
    comments += '_**folder:' + EXPERIMENT + '**_'
    comments += '_batchsize:' + str(batch_size)
    gen_train = Task(timerepeat=timerepeat, epochs=epochs, batch_size=batch_size, steps_per_epoch=10,
                     name=task_name, train_val_test='train', maxlen=100, comments=comments)

    model_args = dict(task_name=task_name, net_name='maLSNN', n_neurons=n_neurons, lr=0.01, stack=stack,
                      loss_name=loss_name, embedding=embedding, optimizer_name='SGD', lr_schedule='',
                      weight_decay=0.1, clipnorm=1., initializer='glorot_uniform', comments=comments,
                      in_len=gen_train.in_len, n_in=gen_train.in_dim, out_len=gen_train.out_len,
                      n_out=gen_train.out_dim, final_epochs=gen_train.epochs)
    train_model = build_model(**model_args)

    adapt_comments, all_bins, all_ns, popts = adapt_sg_shape(gen_train, train_model, comments, test=True)

    fig, axs = plt.subplots(len(all_bins), 1, gridspec_kw={'wspace': .3, 'hspace': .1}, figsize=(10, 5))

    shape_func = get_shape(comments)
    for i, (bin, n, popt) in enumerate(zip(all_bins, all_ns, popts)):
        axs[i].plot(bin, n, '-', color='r')
        axs[i].plot(bin, shape_func(bin, *popt), '-', color='b')

    plt.show()


if __name__ == '__main__':
    test_adapt_sg_shape()
