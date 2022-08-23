import os, argparse, time
import numpy as np
import tensorflow as tf
from sg_design_lif.neural_models.full_model import build_model
from sg_design_lif.generate_data.task_redirection import checkTaskMeanVariance

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default='wordptb', type=str, help="Task to test")
parser.add_argument("--comments", default='', type=str, help="Comments that influence the code")
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
time_steps = 4
batch_size = 1
units = 3
stack = 2
n_classes = 10

batch = tf.random.normal((batch_size, time_steps, input_dim))
words = tf.random.uniform(shape=(batch_size, time_steps), minval=0, maxval=n_classes, dtype=tf.int64)

comments = args.comments
full_mean, full_var = checkTaskMeanVariance(args.task_name)
comments = comments + '_taskmean:{}_taskvar:{}'.format(full_mean, full_var)

comments += '_**folder:' + EXPERIMENT + '**_'
comments += '_batchsize:' + str(batch_size)

model = build_model(args.task_name, 'maLSNN', units, .1, .1, stack, 'sparse_categorical_crossentropy', False,
                    'SWAAdaBelief', .1, '', .1, .1, 'glorot_uniform', comments, 1, input_dim, 1, n_classes, 1)
