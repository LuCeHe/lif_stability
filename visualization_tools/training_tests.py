import os, pickle, logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from lif_stability.visualization_tools.plotting_tools import smart_plot, bpc_prior
from lif_stability.visualization_tools.save_text import save_sentences
from lif_stability.neural_models.full_model import build_model

logger = logging.getLogger('mylogger')
exclude_layers = ['Squeeze', 'multiply', 'add_loss_layer', 'add_metrics_layer', 'softmax', 'target_words', 'dropout',
                  'input_spikes']


def Tests_tf1(x, y, mask, gen, train_tensors, train_tensors_grads, sess, images_dir, rnn):
    task_name = gen.name
    val_batch = gen.data_generation()

    batch_size = val_batch['input_spikes'].shape[0]
    val_dict = {x: val_batch['input_spikes'], y: val_batch['target_output'], mask: val_batch['mask']}
    trt, gs = sess.run([train_tensors, train_tensors_grads], feed_dict=val_dict)

    task = {'input': val_batch['input_spikes'], 'target_output': val_batch['target_output']}
    task.update(trt)
    trained_images_dir = os.path.join(*[images_dir, 'trained'])
    os.mkdir(trained_images_dir)
    for batch_sample in tqdm(range(min(batch_size, 8))):
        pathplot = os.path.join(*[trained_images_dir, 'plot_s{}.png'.format(batch_sample)])
        smart_plot(task, pathplot, batch_sample)

    with open(trained_images_dir + '/png_content.dat', 'wb') as outfile:
        pickle.dump(task, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    text_path = os.path.join(*[trained_images_dir, 'text.png'])
    save_sentences(task, text_path.replace('.png', ''), gen)

    # task specific tests
    if task_name == 'ptb':
        bpc_prior(gen)

    if task_name == 'time_ae_merge':

        for batch_flag in gen.batch_types:
            try:
                length = len(rnn.cell.switch_off)
                logger.warning(rnn.cell.switch_off)
                new_switch_off = np.ones(length).astype(float)
                rnn.cell.switch_off = new_switch_off
                logger.warning(rnn.cell.switch_off)
            except Exception as e:
                logger.warning(e)
                logger.warning('Neurons were not switched off successfully!!')
            logger.warning(batch_flag)
            gen.batch_flag = batch_flag
            mixed_batch = gen.data_generation()
            val_dict = {x: mixed_batch['input_spikes'],
                        y: mixed_batch['target_output'],
                        mask: mixed_batch['mask']}

            # test behavior of the full neural network
            trt, gs = sess.run([train_tensors, train_tensors_grads], feed_dict=val_dict)

            task = {'input': mixed_batch['input_spikes'], 'target_output': mixed_batch['target_output']}
            task.update(trt)
            trained_images_dir = os.path.join(*[images_dir, '{}_trained_on'.format(batch_flag)])
            os.mkdir(trained_images_dir)
            for batch_sample in tqdm(range(min(batch_size, 8))):
                pathplot = os.path.join(*[trained_images_dir, 'plot_s{}.png'.format(batch_sample)])
                smart_plot(task, pathplot, batch_sample)

            text_path = os.path.join(*[trained_images_dir, 'text.png'])
            save_sentences(task, text_path.replace('.png', ''), gen)

            # test behavior of the full neural network when neurons are switched off
            for i in range(1):
                gen.batch_flag = batch_flag
                try:
                    length = len(rnn.cell.switch_off)
                    logger.warning(rnn.cell.switch_off)
                    new_switch_off = np.random.choice(2, length).astype(float)
                    rnn.cell.switch_off = new_switch_off
                    logger.warning(rnn.cell.switch_off)
                except Exception as e:
                    logger.warning(e)
                    logger.warning('Neurons were not switched off successfully!!')

                trt, gs = sess.run([train_tensors, train_tensors_grads], feed_dict=val_dict)

                task = {'input': mixed_batch['input_spikes'], 'target_output': mixed_batch['target_output']}
                task.update(trt)
                trained_images_dir = os.path.join(*[images_dir, '{}_trained_off_{}'.format(batch_flag, i)])
                os.mkdir(trained_images_dir)
                for batch_sample in tqdm(range(min(batch_size, 8))):
                    pathplot = os.path.join(*[trained_images_dir, 'plot_s{}.png'.format(batch_sample)])
                    smart_plot(task, pathplot, batch_sample)

                text_path = os.path.join(*[trained_images_dir, 'text.png'])
                save_sentences(task, text_path.replace('.png', ''), gen)


def get_test_model(original_model):
    intermediate_layers = []
    for l in original_model.layers:
        if (not l.name in exclude_layers) and all([not el in l.name for el in exclude_layers]):
            try:
                # if len(l.output_shape) > 2:
                if isinstance(l.output_shape[0], int):
                    intermediate_layers.append(original_model.get_layer(l.name).output)
                else:
                    layer = original_model.get_layer(l.name).output
                    if l.output_shape[0] == None:
                        layer = (layer,)
                    intermediate_layers.extend(layer)
                # elif len(l.output_shape) == 1:
            except Exception as e:
                print(e)

    test_model = tf.keras.models.Model(original_model.input, intermediate_layers, name='test_model')
    return test_model


def do_grad_tests_old(train_model, batch, task):
    batch = [tf.convert_to_tensor(tf.cast(b, tf.float32), dtype=tf.float32) for b in batch[0]],
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(batch)
        cargo = train_model.layers[0](batch[0][0])
        for layer in train_model.layers[1:-4]:
            new_cargo = layer(cargo)
            if 'encoder' in layer.name:
                try:
                    new_cargo = new_cargo[0]
                    grad = tape.gradient(new_cargo, cargo)
                    task.update({layer.name + '_grad': grad})
                except Exception as e:
                    print(e)

            cargo = new_cargo
    return task


def do_grad_tests(model_args, batch, task, batch_size, seed=None):
    if isinstance(model_args['stack'], str):
        stack = [int(s) for s in model_args['stack'].split(':')]
    elif isinstance(model_args['stack'], int):
        stack = [model_args['n_neurons'] for _ in range(model_args['stack'])]

    batch = [tf.convert_to_tensor(tf.cast(b, tf.float32), dtype=tf.float32) for b in batch[0]],

    states = []
    for width in stack:
        for _ in range(4):
            states.append(tf.zeros((batch_size, width)))
    all_maxs = [[] for _ in stack]
    all_vars = [[] for _ in stack]
    time_steps = batch[0][0].shape[1]
    timei = int(time_steps/2)
    print(model_args['comments'])
    if 'test' in model_args['comments']:
        timef = int(time_steps/2) + 2
    else:
        timef = int(time_steps/2) + 2

    for t in tqdm(range(timei, timef)):
        # print(t, '-' * 30)
        tf.random.set_seed(seed)
        bt = batch[0][0][:, t, :][:, None]
        wt = batch[0][1][:, t][:, None]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(bt)
            tape.watch(wt)
            tape.watch(states)

            model = build_model(**model_args, initial_state='')
            ws = model.get_weights()

            outputs = model([bt, wt, *states])
            states_p1 = outputs[1:]

            hs_t = []
            hs_tp1 = []
            for i in range(len(stack)):
                hs_tp1.append(states_p1[i * 4 + 1])
                hs_t.append(states[i * 4 + 1])

        grads = []

        for htp1, ht in zip(hs_tp1, hs_t):
            grad = tape.batch_jacobian(htp1, ht)
            grad = np.reshape(grad, (grad.shape[0], -1))
            grads.append(grad)

        tf.keras.backend.clear_session()
        del model, tape, grad

        states = states_p1
        for i, g in enumerate(grads):
            print('-'*30)
            print(g)
            all_maxs[i].append(np.max(g, axis=-1)[..., None])
            all_vars[i].append(np.var(g, axis=-1)[..., None])

    all_maxs = [np.concatenate(gs, axis=1) for i, gs in enumerate(all_maxs)]
    all_vars = [np.concatenate(gs, axis=1) for i, gs in enumerate(all_vars)]

    print(all_maxs)
    all_maxs_means = {f'encoder_{i}_1_grad_III_maxs': np.mean(gs) for i, gs in enumerate(all_maxs)}
    all_maxs_diff = {f'encoder_{i}_1_grad_III_maxs_diff': np.mean(np.diff(gs)) for i, gs in enumerate(all_maxs)}
    all_vars_means = {f'encoder_{i}_1_grad_vars': np.mean(gs, axis=1) for i, gs in enumerate(all_vars)}
    all_vars_diff = {f'encoder_{i}_1_vars_diff': np.mean(np.diff(gs)) for i, gs in enumerate(all_vars)}
    results = {}

    for d in [all_maxs_means, all_maxs_diff, all_vars_means, all_vars_diff]:
        results.update(d)
    return results


def Tests(task_name, gen, train_model, images_dir, max_pics=3, subdir_name='trained', png_suffix='', save_pickle=True,
          save_text=True, disable_tqdm=False, plot_weights=False, grad_tests=True, save_plots=True, model_args={}, seed=None):
    test_results = {}
    test_model = get_test_model(train_model)
    rnns_cells = [l.cell for l in test_model.layers if 'encoder_' in l.name]

    batch = gen.__getitem__()
    task = {'input_spikes': batch[0][0], 'target_output': batch[0][1]}
    trt = test_model.predict(batch, batch_size=gen.batch_size)
    trt = {name: pred for name, pred in zip(test_model.output_names, trt)}
    task.update(trt)
    cresults = check_assumptions(task)
    test_results.update(cresults)

    if grad_tests:
        gresults = do_grad_tests(model_args, batch, task, batch_size=gen.batch_size, seed=seed)
        test_results.update(gresults)

    trained_images_dir = os.path.join(*[images_dir, subdir_name])
    if not os.path.isdir(trained_images_dir):
        os.mkdir(trained_images_dir)

    if save_text:
        text_path = os.path.join(*[trained_images_dir, 'text'])
        save_sentences(task, text_path, gen)

    if save_plots:
        for batch_sample in tqdm(range(min(gen.batch_size, max_pics)), disable=disable_tqdm):
            pathplot = os.path.join(*[trained_images_dir, 'plot_s{}{}.png'.format(batch_sample, png_suffix)])
            smart_plot(task, pathplot, batch_sample)

    if save_pickle:
        with open(trained_images_dir + '/png_content.dat', 'wb') as outfile:
            pickle.dump(task, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    # task specific tests
    # if task_name == 'ptb':
    #     bpc_prior(gen)
    if plot_weights:
        tests_on_weights(test_model, images_dir)

    if task_name == 'time_ae_merge':

        for batch_flag in gen.batch_types:

            try:
                logger.warning([rc.switch_off for rc in rnns_cells])
                for rc in rnns_cells: rc.switch_off = 1
                logger.warning([rc.switch_off for rc in rnns_cells])

            except Exception as e:
                logger.warning(e)
                logger.warning('Neurons were not switched off successfully!!')

            gen.batch_flag = batch_flag

            input = gen.data_generation()
            task = input
            trt = test_model.predict((task['input_spikes'], task['mask']))
            trt = {name: pred for name, pred in zip(test_model.output_names, trt)}
            task.update(trt)

            trained_images_dir = os.path.join(*[images_dir, '{}_trained_on'.format(batch_flag)])
            os.mkdir(trained_images_dir)
            for batch_sample in tqdm(range(min(gen.batch_size, 8))):
                pathplot = os.path.join(*[trained_images_dir, 'plot_s{}.png'.format(batch_sample)])
                smart_plot(task, pathplot, batch_sample)

            text_path = os.path.join(*[trained_images_dir, 'text.png'])
            save_sentences(task, text_path.replace('.png', ''), gen)

            # test behavior of the full neural network when neurons are switched off
            for i in range(1):
                gen.batch_flag = batch_flag
                try:
                    logger.warning([rc.switch_off for rc in rnns_cells])
                    for rc in rnns_cells: rc.switch_off = np.random.choice(2, rc.num_neurons).astype(float)
                    logger.warning([rc.switch_off for rc in rnns_cells])
                except Exception as e:
                    logger.warning(e)
                    logger.warning('Neurons were not switched off successfully!!')

                task = input
                trt = test_model.predict((task['input_spikes'], task['mask']))
                trt = {name: pred for name, pred in zip(test_model.output_names, trt)}
                task.update(trt)
                trained_images_dir = os.path.join(*[images_dir, '{}_trained_off_{}'.format(batch_flag, i)])
                os.mkdir(trained_images_dir)
                for batch_sample in tqdm(range(min(gen.batch_size, 8))):
                    pathplot = os.path.join(*[trained_images_dir, 'plot_s{}.png'.format(batch_sample)])
                    smart_plot(task, pathplot, batch_sample)

                text_path = os.path.join(*[trained_images_dir, 'text.png'])
                save_sentences(task, text_path.replace('.png', ''), gen)

    test_results = {k + '_' + subdir_name: v for k, v in test_results.items()}
    return test_results


def checkWeightDistribution(model, gen_test, do_smart_plot):
    batch = gen_test.data_generation()
    # prediction = model.predict(batch[0], batch_size=gen_test.batch_size)
    # print(prediction)
    if do_smart_plot:
        task = {k: v for k, v in batch.copy().items() if not k in ['mask', 'sentences']}
        trt = model.predict((batch['input_spikes'], batch['mask']),
                            batch_size=gen_test.batch_size)
        trt = {name: pred for name, pred in zip(model.output_names, trt)}

        task.update(trt)
        pathplot = os.path.join(*[GOOD_EXPS, 'plot.png'])
        smart_plot(task, pathplot)

    for layer in model.layers:
        weights = layer.get_weights()
        print(layer.name)

    l = model.get_layer('encoder_0')
    weights = l.get_weights()
    names = [w.name for w in l.weights]

    n_features = len(weights)
    fig, axs = plt.subplots(n_features, 1)
    for n, k, ax in zip(names, weights, axs.tolist()):
        x = k if len(k.shape) == 1 else k.flatten()
        x = x[~np.isnan(x)]

        # the histogram of the data
        ax.hist(x, 50, density=True)
        ax.set_ylabel(n)

    plt.show()


class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, gen_val, test_model, grads_model=None, print_every=1,
                 images_dir='', text_dir='', task_name=''):
        self.gen_val = gen_val
        self.val_batch = gen_val.data_generation()
        # self.val_batch = (self.val_batch[0][0], self.val_batch[1])
        self.test_model, self.grads_model = test_model, grads_model
        self.print_every = print_every
        self.images_dir, self.text_dir, self.task_name = images_dir, text_dir, task_name

    def plot(self, epoch):
        task = {k: v for k, v in self.val_batch.copy().items() if not k in ['mask', 'sentences']}
        trt = self.test_model.predict((self.val_batch['input_spikes'], self.val_batch['mask']),
                                      batch_size=self.gen_val.batch_size)
        trt = {name: pred for name, pred in zip(self.test_model.output_names, trt)}

        task.update(trt)
        pathplot = os.path.join(*[self.images_dir, 'plot_iter_{}.png'.format(epoch)])
        smart_plot(task, pathplot)

        text_path = os.path.join(*[self.text_dir, 'sentences_iter_{}'.format(epoch)])
        save_sentences(task, text_path, self.gen_val)

        if not self.grads_model is None:
            task = self.val_batch
            gs = self.grads_model.predict(task)
            task.update(gs)
            pathplot = os.path.join(*[self.images_dir, 'grads_plot_iter_{}.png'.format(epoch)])
            smart_plot(task, pathplot)
            del gs

        del task, trt

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            self.plot(-1)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_every == 0:
            self.plot(epoch)


def tests_on_weights(keras_model, save_dir=None):
    import pandas as pd
    from scipy.stats import skew

    df = pd.DataFrame()

    for w in keras_model.trainable_weights:
        if len(w.shape) == 2:
            plt.figure()
            # w = np.random.rand(20, 30)
            plt.imshow(w)
            plt.colorbar()
            _, sv, _ = np.linalg.svd(w)
            sk = skew(w.numpy().flatten())
            row_df = pd.DataFrame([[w.name, np.mean(w), np.std(w), np.min(w), np.max(w), sk, np.min(sv), np.max(sv)]])
            df = pd.concat([row_df, df], ignore_index=True)
            if not save_dir is None:
                pathplot = os.path.join(save_dir, w.name.replace('/', '_').replace(':', '_') + '.png')
                plt.savefig(pathplot, bbox_inches='tight')

    df.rename(columns={0: 'name', 1: 'mean', 2: 'variance', 3: 'min', 4: 'max', 5: 'skewness', 6: 'min_sing_value',
                       7: 'max_sing_value'}, inplace=True)
    if not save_dir is None:
        pathplot = os.path.join(save_dir, 'weights_statistics.txt')
        with open(pathplot, "w") as text_file:
            text_file.write(df.to_string())


def check_assumptions(task):
    cresults = {}
    # print(task.keys())
    # print('Is median~mean a good assumption?')
    for k in task.keys():
        if k.endswith("_3") and k.startswith("encoder"):
            mean = np.mean(task[k])
            median = np.median(task[k])
            std = np.std(task[k])
            # print(k)
            # print(
            #     f'Mean {mean}, median {median}, mean-median {mean - median}, std {std}, |mean-median|^2/std**2 {(mean - median) ** 2 / std ** 2}, ')
            cresults.update({
                k + '_mm_mean': mean,
                k + '_mm_median': median,
                k + '_mm_std': std,
                k + '_mm_mm/s': (mean - median) ** 2 / std ** 2,
            })

    # print('What is the firing rate at initialization?')
    for k in task.keys():
        if k.endswith("_0") and k.startswith("encoder"):
            mean = np.mean(task[k])
            std = np.std(task[k])
            # print(k)
            # print(f'Mean firing rate {mean}, std firing rate {std}')
            cresults.update({
                k + '_fr_mean': mean,
                k + '_fr_std': std,
            })

    # print('Condition III?')
    for k in task.keys():
        if k.endswith("grad") and k.startswith("encoder"):
            maxs = np.max(task[k], axis=1)
            diff_maxs = maxs[..., :-1] - maxs[..., -1:]
            sim = np.mean(np.exp(-diff_maxs ** 2))
            # print(k)
            # print(f'Mean max grad sim {sim}')
            cresults.update({
                k + '_III': np.mean(np.abs(diff_maxs)),
                k + '_III_mean': np.mean(maxs),
                k + '_III_sim': sim,
            })

    # print('Condition IV?')
    for k in task.keys():
        if k.endswith("grad") and k.startswith("encoder"):
            vars = np.var(task[k], axis=1)
            diff_vars = vars[..., :-1] - vars[..., -1:]
            sim = np.mean(np.exp(-diff_vars ** 2))
            # print(k)
            # print(f'Mean var grad sim {sim}')
            cresults.update({
                k + '_IV': np.mean(diff_vars),
                k + '_IV_mean': np.mean(vars),
                k + '_IV_sim': sim,
            })

    return cresults


if __name__ == '__main__':
    i = tf.keras.layers.Input((30,))
    o = tf.keras.layers.Dense(300)(i)
    o = tf.keras.layers.Dense(200)(o)
    model = tf.keras.models.Model(i, o)

    tests_on_weights(model)
