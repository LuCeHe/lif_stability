import os, argparse, time
import numpy as np
import tensorflow as tf

from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real

from GenericTools.stay_organized.skopt_tools import tqdm_skopt
from sg_design_lif.neural_models.full_model import build_model
from sg_design_lif.generate_data.task_redirection import checkTaskMeanVariance

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default='wordptb', type=str, help="Task to test")
parser.add_argument("--comments", default='', type=str, help="Comments that influence the code")
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
batch_size = 10
units = 100
stack = 1
n_classes = 10

batch = tf.random.uniform((batch_size, time_steps, input_dim))
words = tf.random.uniform(shape=(batch_size, time_steps), minval=0, maxval=n_classes, dtype=tf.int64)

comments = args.comments
full_mean, full_var = checkTaskMeanVariance(args.task_name)
comments = comments + '_taskmean:{}_taskvar:{}'.format(full_mean, full_var)

comments += '_**folder:' + EXPERIMENT + '**_'
comments += '_batchsize:' + str(batch_size)

# gp to find the means of wrec and bias to have a desired firing rate

# parameter to whose initialization to optimize: wrecm, v0m
ptoopt = 'wrecm'
target_firing_rate = .1


def gp_objective(space, comments, ptoopt, target_firing_rate):
    comments = comments

    @use_named_args(space)
    def objective(**params):
        layer = [v for k, v in params.items() if 'layer' in k]
        print('layer', layer)
        for i, l in enumerate(layer):
            comments_i = comments + f'_{ptoopt}{i}:{l}'  # '_wrecm0:-.1_v0m0:.2_v0m1:.3'

        model = build_model(
            args.task_name, 'maLSNN', units, .1, .1, stack, 'sparse_categorical_crossentropy', False,
            'SWAAdaBelief', .1, '', .1, .1, 'glorot_uniform', comments_i, 1, input_dim, 1, n_classes, 1
        )

        # evaluate so you evaluate on the whole training set, or a few steps_per_epoch
        evaluation = model.evaluate(((batch, words),), return_dict=True, verbose=False)

        tf.keras.backend.clear_session()
        del model

        fs = [v for k, v in evaluation.items() if 'firing_rate' in k]
        loss = np.mean([np.abs(f - target_firing_rate) for f in fs])
        print(fs)
        # print(loss)

        return loss

    return objective


space = [Real(-10, 10, name='layer_{}'.format(i)) for i in range(stack)]

res_gp = gp_minimize(
    gp_objective(space, comments, ptoopt, target_firing_rate), space, n_calls=args.n_calls,
    random_state=args.seed,
    callback=[tqdm_skopt(total=args.n_calls, desc="Gaussian Process")]
)
print("Best parameters: ", res_gp.x)
print("Best loss:       ", res_gp.fun)

# save in a pandas: desired f, value wrecm/v0m, obtained f, task
