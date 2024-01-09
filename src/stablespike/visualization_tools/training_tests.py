import os, pickle, logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from stablespike.visualization_tools.plotting_tools import smart_plot, bpc_prior
from stablespike.visualization_tools.save_text import save_sentences
from stablespike.neural_models.full_model import build_model

logger = logging.getLogger('mylogger')
exclude_layers = ['Squeeze', 'multiply', 'add_loss_layer', 'add_metrics_layer', 'softmax', 'target_words', 'dropout',
                  'input_spikes']



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

