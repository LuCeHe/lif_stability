import os, argparse, time
import numpy as np
import tensorflow as tf

from GenericTools.keras_tools.esoteric_layers import AddLossLayer
from GenericTools.keras_tools.expose_latent import expose_latent_model
from sg_design_lif.generate_data.task_redirection import checkTaskMeanVariance
from sg_design_lif.neural_models.full_model import build_model
from sg_design_lif.neural_models.sparsity_gp import sparsity_gp


def reduce_model_firing_activity(
        model, target_firing_rate, generator, epochs=100, layer_identifier='encoder', output_index=0,
        trainable_param_identifier='internal_current'
):
    new_model = expose_latent_model(model, include_layers=[layer_identifier], idx=output_index)

    initial_trainable = None
    initial_non_trainable = None
    for layer in new_model.layers:
        for i, w in enumerate(layer.trainable_weights):
            if not 'internal_current' in w.name:
                w._trainable = False
                if initial_non_trainable is None:
                    initial_non_trainable = w
            else:
                if initial_trainable is None:
                    initial_trainable = w

    outs = []
    for i, out in enumerate(new_model.outputs):
        loss = lambda t, p: tf.square(tf.reduce_mean(t) - target_firing_rate)
        loss.name = f'fire_adjustment_{i}'
        output_net = AddLossLayer(loss=loss, name=loss.name)([out, out])
        outs.append(output_net)

    train_model = tf.keras.models.Model(new_model.inputs, outs, name='sparsifier')
    train_model.compile(optimizer='AdaM', loss=lambda x, y: 0)

    train_model.fit(generator, epochs=epochs)

    final_trainable = None
    final_non_trainable = None
    for layer in train_model.layers:
        for i, w in enumerate(layer.trainable_weights):
            if not trainable_param_identifier in w.name:
                if final_non_trainable is None:
                    final_non_trainable = w
            else:
                if final_trainable is None:
                    final_trainable = w


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default='wordptb', type=str, help="Task to test")
    parser.add_argument("--comments", default='', type=str, help="Comments that influence the code")
    parser.add_argument("--opt_type", default='dl', type=str, help="Comments that influence the code",
                        choices=['gp', 'dl'])
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--n_calls", default=110, type=int, help="Gaussian Processes iterations")

    args = parser.parse_args()

    FILENAME = os.path.realpath(__file__)
    CDIR = os.path.dirname(FILENAME)
    DATA = os.path.join(CDIR, 'data', )
    EXPERIMENTS = os.path.join(CDIR, 'experiments')
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    EXPERIMENT = os.path.join(EXPERIMENTS, time_string + random_string + '_initconds')
    MODL = os.path.join(EXPERIMENT, 'trained_models')
    GENDATA = os.path.join(DATA, 'initconds')

    for d in [EXPERIMENT, GENDATA, MODL]:
        os.makedirs(d, exist_ok=True)

    np.random.seed(42)
    tf.random.set_seed(42)

    input_dim = 2
    time_steps = 100
    batch_size = 32
    units = 100
    stack = 2
    n_classes = 10
    epochs = 600
    batch = tf.random.uniform((batch_size, time_steps, input_dim))
    words = tf.random.uniform(shape=(batch_size, time_steps), minval=0, maxval=n_classes, dtype=tf.int64)

    comments = args.comments
    full_mean, full_var = checkTaskMeanVariance(args.task_name)
    comments = comments + '_taskmean:{}_taskvar:{}'.format(full_mean, full_var)

    comments += '_**folder:' + EXPERIMENT + '**_'
    comments += '_batchsize:' + str(batch_size)

    # gp to find the means of wrec and bias to have a desired firing rate

    # parameter to whose initialization to optimize: wrecm, v0m
    ptoopt = 'v0m'
    target_firing_rate = .1

    net_hyp = [
        args.task_name, 'maLSNN', units, .1, .1, stack, 'sparse_categorical_crossentropy', False,
        'SWAAdaBelief', .1, '', .1, .1, 'glorot_uniform', comments, 1, input_dim, 1, n_classes, 1
    ]

    if args.opt_type == 'gp':
        sparsity_gp(comments, ptoopt, target_firing_rate, net_hyp, args, batch, words, stack)

    elif args.opt_type == 'dl':
        comments += '_v0m'

        net_hyp = [
            args.task_name, 'maLSNN', units, .1, .1, stack, 'sparse_categorical_crossentropy', False,
            'SWAAdaBelief', .1, '', .1, .1, 'glorot_uniform', comments, 1, input_dim, 1, n_classes, 1
        ]

        model = build_model(*net_hyp)

        # test_model = get_test_model(model)
        new_model = expose_latent_model(model, include_layers=['encoder'], idx=0)
        new_model.summary()
        # prediction = new_model.predict((batch, words))
        # print(len(prediction))
        # print(prediction)
        names = [weight.name for layer in new_model.layers for weight in layer.weights]
        print(names)

        initial_trainable = None
        initial_non_trainable = None
        for layer in new_model.layers:
            print(layer.name, len(layer.non_trainable_weights), len(layer.trainable_weights))
            for i, w in enumerate(layer.trainable_weights):
                # print(w.name)
                if not 'internal_current' in w.name:
                    w._trainable = False
                    if initial_non_trainable is None:
                        initial_non_trainable = w
                else:
                    if initial_trainable is None:
                        initial_trainable = w

        outs = []
        for i, out in enumerate(new_model.outputs):
            loss = lambda t, p: tf.square(tf.reduce_mean(t) - target_firing_rate)
            loss.name = f'fire_adjustment_{i}'
            output_net = AddLossLayer(loss=loss, name=loss.name)([out, out])
            outs.append(output_net)

        print('initial_means', np.mean(initial_trainable), np.mean(initial_non_trainable))

        train_model = tf.keras.models.Model(new_model.inputs, outs, name='sparsifier')
        train_model.compile(optimizer='AdaM', loss=lambda x, y: 0)

        train_model.fit((batch, words), None, epochs=epochs)

        final_trainable = None
        final_non_trainable = None
        for layer in train_model.layers:
            for i, w in enumerate(layer.trainable_weights):
                if not 'internal_current' in w.name:
                    if final_non_trainable is None:
                        final_non_trainable = w
                else:
                    if final_trainable is None:
                        final_trainable = w

        print('trainable', np.mean(initial_trainable), np.mean(final_trainable))
        print(initial_trainable[:10])
        print(final_trainable[:10])
        print('non-trainable', np.mean(initial_non_trainable), np.mean(final_non_trainable))
        print(initial_non_trainable[:10])
        print(final_non_trainable[:10])
        model.evaluate((batch, words))
    else:
        raise NotImplementedError
    # save in a pandas: desired f, value wrecm/v0m, obtained f, task
