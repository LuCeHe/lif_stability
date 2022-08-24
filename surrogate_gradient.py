import os, shutil, logging, json, sys

import numpy as np
import pandas as pd

import tensorflow as tf

from sg_design_lif.neural_models.find_sparsities import reduce_model_firing_activity

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from GenericTools.keras_tools.esoteric_callbacks.several_validations import MultipleValidationSets
from sg_design_lif.neural_models.config import default_config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_VMODULE"] = "gpu_process_state=10,gpu_cudamallocasync_allocator=10"

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.callbacks import ReduceLROnPlateau

from GenericTools.keras_tools.convergence_metric import convergence_estimation
from GenericTools.keras_tools.esoteric_callbacks.annealing_callback import *
from GenericTools.keras_tools.esoteric_callbacks.gradient_tensorboard import ExtendedTensorBoard
from GenericTools.keras_tools.esoteric_initializers import esoteric_initializers_list, get_initializer
from GenericTools.keras_tools.esoteric_callbacks import *
from GenericTools.keras_tools.plot_tools import plot_history
from GenericTools.stay_organized.VeryCustomSacred import CustomExperiment, ChooseGPU
from GenericTools.stay_organized.utils import timeStructured, setReproducible, str2val, NumpyEncoder

from sg_design_lif.generate_data.task_redirection import Task, checkTaskMeanVariance, language_tasks
from sg_design_lif.visualization_tools.training_tests import Tests, check_assumptions
from sg_design_lif.neural_models.full_model import build_model

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

ex = CustomExperiment('-mnl', base_dir=CDIR, seed=11)
logger = logging.getLogger('mylogger')


@ex.config
def config():
    # environment properties
    GPU = None
    seed = 41

    # task and net
    # ptb time_ae simplest_random time_ae_merge ps_mnist heidelberg wiki103 wmt14 s_mnist xor small_s_mnist
    # wordptb sl_mnist
    task_name = 'wordptb'

    # test configuration
    epochs = 3
    steps_per_epoch = 2
    batch_size = 2
    stack = None
    n_neurons = 10

    # net
    # mn_aLSNN_2 mn_aLSNN_2_sig LSNN maLSNN spikingPerformer smallGPT2 aLSNN_noIC spikingLSTM
    net_name = 'maLSNN'
    # zero_mean_isotropic zero_mean learned positional normal onehot zero_mean_normal
    sLSTM_factor = 2 / 3 if task_name == 'wordptb' else 1 / 3
    n_neurons = n_neurons if not net_name == 'spikingLSTM' else int(n_neurons * sLSTM_factor)
    embedding = 'learned:None:None:{}'.format(n_neurons) if task_name in language_tasks else False

    comments = '6_embproj_noalif_nogradreset_dropout:.3_timerepeat:2_adjfi:.1_v0m'

    # optimizer properties
    lr = None  # 7e-4
    optimizer_name = 'SWAAdaBelief'  # AdaBelief AdamW SWAAdaBelief
    lr_schedule = ''  # 'warmup_cosine_restarts'
    weight_decay_prop_lr = None
    weight_decay = .01 if not 'mnist' in task_name else 0.  # weight_decay_prop_lr * lr
    clipnorm = 1.  # not 1., to avoid NaN in the embedding, only ptb though

    loss_name = 'sparse_categorical_crossentropy'  # categorical_crossentropy categorical_focal_loss contrastive_loss
    initializer = 'glorot_uniform'  # uniform glorot_uniform orthogonal glorot_normal NoZeroGlorot

    continue_training = ''
    save_model = False

    # 22h=79200 s, 21h=75600 s, 20h=72000 s, 12h = 43200 s, 6h = 21600 s, 72h = 259200
    stop_time = 21600


@ex.capture
@ex.automain
def main(epochs, steps_per_epoch, batch_size, GPU, task_name, comments,
         continue_training, save_model, seed, net_name, n_neurons, lr, stack, loss_name, embedding, optimizer_name,
         lr_schedule, weight_decay, clipnorm, initializer, stop_time, _log):
    stack, batch_size, embedding, n_neurons, lr = default_config(stack, batch_size, embedding, n_neurons, lr, task_name)

    exp_dir = os.path.join(*[CDIR, ex.observers[0].basedir])
    comments += '_**folder:' + exp_dir + '**_'

    config_dir = os.path.join(*[exp_dir, '1'])
    images_dir = os.path.join(*[exp_dir, 'images'])
    other_dir = os.path.join(*[exp_dir, 'other_outputs'])
    models_dir = os.path.join(*[exp_dir, 'trained_models'])
    # tracker = OfflineEmissionsTracker(country_iso_code="CAN", output_dir=other_dir)
    # tracker.start()
    full_mean, full_var = checkTaskMeanVariance(task_name)
    print(comments)
    comments = comments + '_taskmean:{}_taskvar:{}'.format(full_mean, full_var)
    print(comments)

    ChooseGPU(GPU)
    setReproducible(seed)

    shutil.copytree(os.path.join(CDIR, 'neural_models'), other_dir + '/neural_models')
    # shutil.copyfile(os.path.join(CDIR, 'run_tf2.sh'), other_dir + '/run_tf2.sh')
    shutil.copyfile(FILENAME, other_dir + '/' + os.path.split(FILENAME)[-1])

    timerepeat = str2val(comments, 'timerepeat', int, default=1)
    maxlen = str2val(comments, 'maxlen', int, default=100)
    comments = str2val(comments, 'maxlen', int, default=maxlen, replace=maxlen)
    comments += '_batchsize:' + str(batch_size)

    # task definition
    gen_train = Task(timerepeat=timerepeat, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                     name=task_name, train_val_test='train', maxlen=maxlen, comments=comments)
    gen_val = Task(timerepeat=timerepeat, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                   name=task_name, train_val_test='val', maxlen=maxlen, comments=comments)
    gen_test = Task(timerepeat=timerepeat, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                    name=task_name, train_val_test='test', maxlen=maxlen, comments=comments)

    final_epochs = gen_train.epochs
    final_steps_per_epoch = gen_train.steps_per_epoch
    tau_adaptation = str2val(comments, 'taub', float, default=int(gen_train.in_len / 2))
    tau = str2val(comments, 'tauv', float, default=.1)
    # tau_adaptation = int(gen_train.in_len / 2)  # 200 800 4000

    if initializer in esoteric_initializers_list:
        initializer = get_initializer(initializer_name=initializer)

    model_args = dict(task_name=task_name, net_name=net_name, n_neurons=n_neurons, tau=tau,
                      lr=lr, stack=stack, loss_name=loss_name,
                      embedding=embedding, optimizer_name=optimizer_name, lr_schedule=lr_schedule,
                      weight_decay=weight_decay, clipnorm=clipnorm, initializer=initializer, comments=comments,
                      in_len=gen_train.in_len, n_in=gen_train.in_dim, out_len=gen_train.out_len,
                      n_out=gen_train.out_dim, tau_adaptation=tau_adaptation,
                      final_epochs=gen_train.epochs)
    train_model = build_model(**model_args)

    results = {}
    # this block is only necessary when I'm continuing training a previous model
    if 'continue_202' in continue_training:
        print(continue_training)
        path_exp = os.path.join(CDIR, 'experiments', continue_training.replace('continue_', ''))
        path_model = os.path.join(path_exp, 'trained_models', 'train_model.h5')
        train_model.load_weights(path_model)

        old_results = os.path.join(path_exp, 'other_outputs', 'results.json')

        with open(old_results) as f:
            old_data = json.load(f)

        results['accumulated_epochs'] = old_data['accumulated_epochs']  # + final_epochs
    else:
        results['accumulated_epochs'] = 0  # final_epochs

    train_model.summary()

    history_path = other_dir + '/log.csv'
    val_data = gen_val.__getitem__()

    checkpoint_filepath = os.path.join(models_dir, 'checkpoint')
    callbacks = [
        LearningRateLogger(),
        # VariablesLogger(variables_to_log=['hard_heaviside']),
        TimeStopping(stop_time, 1),  # 22h=79200 s, 21h=75600 s, 20h=72000 s, 12h = 43200 s, 6h = 21600 s, 72h = 259200
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True
        ),
        MultipleValidationSets({'v': gen_val, 't': gen_test}, verbose=0),
        tf.keras.callbacks.CSVLogger(history_path),
    ]

    if 'annealing' in comments:
        annealing_schedule = str2val(comments, 'annealing', str, default='ha')
        callbacks.append(
            AnnealingCallback(
                epochs=final_epochs, variables_to_anneal=['hard_heaviside'], annealing_schedule=annealing_schedule,
            )
        )

    if 'tenb' in comments:
        callbacks.append(
            ExtendedTensorBoard(validation_data=val_data, log_dir=other_dir, histogram_freq=2),
        )

    if 'roplateau' in comments:
        callbacks.append(
            ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=15, min_lr=lr / 1000),
        )

    # plots before training
    # Tests(task_name, gen_val, train_model, images_dir, save_pickle=False, subdir_name='nontrained')
    # sys.exit("Error message")

    # evaluation = train_model.evaluate(gen_val, return_dict=True, verbose=True)

    if 'adjfi' in comments:
        target_firing_rate = str2val(comments, 'adjfi', float, default=.1)
        reduce_model_firing_activity(
            train_model, target_firing_rate, gen_train, epochs=5
        )

    train_model.fit(
        gen_train, batch_size=batch_size, validation_data=gen_val, epochs=final_epochs, steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )

    actual_epochs = 0
    if final_epochs > 0:
        train_model.load_weights(checkpoint_filepath)
        history_df = pd.read_csv(history_path)

        actual_epochs = history_df['epoch'].iloc[-1] + 1
        results['accumulated_epochs'] = str(int(results['accumulated_epochs']) + int(actual_epochs))
        history_dict = {k: history_df[k].tolist() for k in history_df.columns.tolist()}

        plot_filename = os.path.join(*[images_dir, 'history.png'])
        plot_history(histories=history_dict, plot_filename=plot_filename, epochs=final_epochs)
        json_filename = os.path.join(*[other_dir, 'history.json'])
        history_jsonable = {k: np.array(v).astype(float).tolist() for k, v in history_dict.items()}
        json.dump(history_jsonable, open(json_filename, "w"))

        # plot only validation curves
        history_dict = {k: history_df[k].tolist() if 'val' in k else [] for k in history_df.columns.tolist()}
        plot_filename = os.path.join(images_dir, 'history_val.png')
        plot_history(histories=history_dict, plot_filename=plot_filename, epochs=final_epochs)

        removable_checkpoints = sorted([d for d in os.listdir(models_dir) if 'checkpoint' in d])
        for d in removable_checkpoints: os.remove(os.path.join(models_dir, d))

        results['convergence'] = convergence_estimation(history_dict['val_loss'])

    print('Fitting done!')
    # if not task_name == 'ptb':
    if save_model:
        train_model_path = os.path.join(models_dir, 'train_model.h5')
        train_model.save(train_model_path)
        print('Model saved!')

    # plots after training
    test_results = Tests(task_name, gen_test, train_model, images_dir, save_pickle=False, model_args=model_args,
                         grad_tests=False)
    results.update(test_results)

    evaluation = train_model.evaluate(gen_test, return_dict=True, verbose=True)
    for k in evaluation.keys():
        results[k + '_test_{}'.format(timerepeat)] = evaluation[k]

    results['n_params'] = train_model.count_params()
    results['final_epochs'] = str(actual_epochs)
    results['final_steps_per_epoch'] = final_steps_per_epoch
    results['batch_size'] = batch_size
    results['lr'] = lr
    results['n_neurons'] = n_neurons
    results['stack'] = stack
    results['embedding'] = embedding

    results_filename = os.path.join(other_dir, 'results.json')
    json.dump(results, open(results_filename, "w"), cls=NumpyEncoder)

    string_result = json.dumps(results, indent=4, cls=NumpyEncoder)
    print(string_result)
    path = os.path.join(other_dir, 'results.txt')
    with open(path, "w") as f:
        f.write(string_result)

    print('DONE')
