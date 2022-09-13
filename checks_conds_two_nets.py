import os, time, argparse, json, shutil, logging
import matplotlib.pyplot as plt

from sg_design_lif.visualization_tools.plotting_tools import smart_plot

logging.getLogger("tensorflow").setLevel(logging.ERROR)

from tqdm import tqdm

import numpy as np
from prettytable import PrettyTable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from GenericTools.stay_organized.utils import str2val, NumpyEncoder, setReproducible
from GenericTools.keras_tools.esoteric_tasks.time_task_redirection import Task, checkTaskMeanVariance
from sg_design_lif.neural_models.config import default_config
from sg_design_lif.neural_models.full_model import build_model
from sg_design_lif.visualization_tools.training_tests import Tests, get_test_model

# import warnings
# warnings.filterwarnings('ignore')

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

all_conditions = list(reversed(['', '_conditionI_', '_conditionII_', '_conditionIII_', '_conditionIV_']))

print(all_conditions)
parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--condition", default='_conditionIII_', type=str,
                    help="Condition to test: " + ", ".join(all_conditions))
parser.add_argument("--task_name", default='sl_mnist', type=str, help="Task to test")
parser.add_argument("--steps_per_epoch", default=2, type=int, help="Steps per Epoch")
parser.add_argument("--seed", default=2, type=int, help="Random seed")
parser.add_argument("--tests", default=0, type=int, help="Test on smaller architectures for speed")
args = parser.parse_args()

setReproducible(args.seed)
print(json.dumps(vars(args), sort_keys=True, indent=4))

steps_per_epoch = None if args.steps_per_epoch == -1 else args.steps_per_epoch

base_comments = '6_exponentialpseudod_embproj_noalif_nogradreset_dropout:.3_timerepeat:2_test'

timerepeat = str2val(base_comments, 'timerepeat', int, default=1)
maxlen = str2val(base_comments, 'maxlen', int, default=100)

hyp = [3, 2, 'learned:None:None:2', 2, 0.] if args.tests == 1 else [None] * 5
stack, batch_size, embedding, n_neurons, lr = default_config(*hyp, args.task_name)
stack = '500:300' if args.task_name == 'wordptb' and not args.tests else stack

full_mean, full_var = checkTaskMeanVariance(args.task_name)
base_comments = base_comments + '_taskmean:{}_taskvar:{}'.format(full_mean, full_var)

base_comments += '_**folder:' + EXPERIMENT + '**_'
base_comments += f'_batchsize:{batch_size}_'

c = args.condition
s = args.seed

comment = base_comments + c

evaluations = []
print('\nLoading Models...\n')

for comment in tqdm([comment, comment.replace('condition', '')]):
    gen = Task(timerepeat=timerepeat, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
               name=args.task_name, train_val_test='val', maxlen=maxlen, comments=base_comments)

    model_args = dict(
        task_name=args.task_name, net_name='maLSNN', n_neurons=n_neurons, lr=0., stack=stack,
        loss_name='sparse_categorical_crossentropy', embedding=embedding, optimizer_name='SWAAdaBelief',
        lr_schedule='', weight_decay=.01 if not 'mnist' in args.task_name else 0., clipnorm=1.,
        initializer='he_uniform', comments=comment,
        in_len=gen.in_len, n_in=gen.in_dim, out_len=gen.out_len, n_out=gen.out_dim, final_epochs=gen.epochs,
    )
    model = build_model(**model_args)

    results = {}
    if args.condition == '_conditionI_':
        test_model = get_test_model(model)
        png_suffix = 'c' if 'condition' in comment else 'u'

        batch = gen.__getitem__()
        task = {'input_spikes': batch[0][0], 'target_output': batch[0][1]}
        trt = test_model.predict(batch, batch_size=gen.batch_size)
        trt = {name: pred[:, 50:] for name, pred in zip(test_model.output_names, trt) if
               'encoder' in name and name.endswith('_0')}
        evaluation = {k.replace('encoder', 'fr'): np.mean(v).round(3) for k, v in trt.items()}

    elif args.condition == '_conditionII_':
        grad_tests = True if 'III' in c or 'IV' in c or c == '' else False
        test_results = Tests(args.task_name, gen, model, EXPERIMENT, save_pickle=False,
                             subdir_name='init', save_plots=False, model_args=model_args,
                             grad_tests=grad_tests, seed=s)
        evaluation = model.evaluate(gen, return_dict=True, verbose=False)
        print(evaluation.keys())
        evaluation = {k: v for k, v in evaluation.items() if 'curr_dis_ma_lsnn' in k}
        # evaluation = {
        #     k: 2 * evaluation['curr_dis_ma_lsnn' + k] / (
        #                 evaluation['var_in_ma_lsnn' + k] + evaluation['var_rec_ma_lsnn' + k])
        #     for k in ['', '_1']
        # }

    else:

        grad_tests = True if 'III' in c or 'IV' in c or c == '' else False
        test_results = Tests(args.task_name, gen, model, EXPERIMENT, save_pickle=False,
                             subdir_name='init', save_plots=False, model_args=model_args,
                             grad_tests=grad_tests, seed=s)


        print(test_results.keys())
        # evaluation


    tf.keras.backend.clear_session()
    del model

    evaluations.append(evaluation)
    print(evaluation.keys())

x = PrettyTable()
x.field_names = ["metric", args.condition, "unconditioned"]
for k in evaluations[0].keys():
    x.add_row([k, evaluations[0][k], evaluations[1][k]])

print(x)
