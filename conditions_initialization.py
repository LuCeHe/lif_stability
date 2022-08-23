import os, time, argparse, json, shutil, logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from GenericTools.stay_organized.utils import str2val, NumpyEncoder, setReproducible
from stochastic_spiking.generate_data.task_redirection import Task, checkTaskMeanVariance
from stochastic_spiking.neural_models.config import default_config
from stochastic_spiking.neural_models.full_model import build_model
from stochastic_spiking.visualization_tools.training_tests import Tests

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
parser.add_argument(
    "--conditions", default='', type=str, help="Conditions to test: " + ", ".join(all_conditions) + " and 'all'",
)
parser.add_argument("--task_name", default='sl_mnist', type=str, help="Task to test")
parser.add_argument("--steps_per_epoch", default=2, type=int, help="Steps per Epoch")
parser.add_argument("--n_seeds", default=1, type=int, help="Steps per Epoch")
parser.add_argument("--init_seed", default=1, type=int, help="Steps per Epoch")
parser.add_argument("--plot", default=0, type=int, help="Plot")
parser.add_argument("--plot_existing", default=0, type=int, help="Plot existing seeds, or create new seeds")
parser.add_argument("--histogram", default=1, type=int, help="Plot histogram or scatter")
parser.add_argument("--redoseeds", default=1, type=int, help="Redo seeds that were already computed before")
parser.add_argument("--tests", default=1, type=int, help="Test on smaller architectures for speed")
args = parser.parse_args()
print(json.dumps(vars(args), sort_keys=True, indent=4))

steps_per_epoch = None if args.steps_per_epoch == -1 else args.steps_per_epoch
conditions = all_conditions if args.conditions == 'all' else [args.conditions]
seeds = range(args.init_seed, args.init_seed + args.n_seeds)

base_comments = '6_exponentialpseudod_embproj_noalif_nogradreset_dropout:.3_timerepeat:2_'

timerepeat = str2val(base_comments, 'timerepeat', int, default=1)
maxlen = str2val(base_comments, 'maxlen', int, default=100)

hyp = [1, 2, 'learned:None:None:2', 2, 0.] if args.tests == 1 else [None] * 5
stack, batch_size, embedding, n_neurons, lr = default_config(*hyp, args.task_name)
stack = '500:300' if args.task_name == 'wordptb' and not args.tests else stack

gen_val = Task(timerepeat=timerepeat, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
               name=args.task_name, train_val_test='val', maxlen=maxlen, comments=base_comments)

tau_adaptation = str2val(base_comments, 'taub', float, default=int(gen_val.in_len / 2))
tau = str2val(base_comments, 'tauv', float, default=.1)

full_mean, full_var = checkTaskMeanVariance(args.task_name)
base_comments = base_comments + '_taskmean:{}_taskvar:{}'.format(full_mean, full_var)

base_comments += '_**folder:' + EXPERIMENT + '**_'
base_comments += '_batchsize:' + str(batch_size)

if not args.plot_existing:
    for s in tqdm(seeds):
        setReproducible(s)

        for c in conditions:
            # print(c)
            tf.keras.backend.clear_session()
            # try:
            cfilename = os.path.join(GENDATA, f't{args.task_name}_c{c}_s{s}.txt')
            if not os.path.exists(cfilename) or args.redoseeds:
                try:
                    model_args = dict(
                        task_name=args.task_name, net_name='maLSNN', n_neurons=n_neurons, tau=tau, lr=0., stack=stack,
                        loss_name='sparse_categorical_crossentropy', embedding=embedding, optimizer_name='SWAAdaBelief',
                        lr_schedule='', weight_decay=.01 if not 'mnist' in args.task_name else 0., clipnorm=1.,
                        initializer='glorot_uniform', comments=base_comments + c,
                        in_len=gen_val.in_len, n_in=gen_val.in_dim, out_len=gen_val.out_len,
                        n_out=gen_val.out_dim, tau_adaptation=tau_adaptation, final_epochs=gen_val.epochs,
                    )
                    model = build_model(**model_args)

                    results = {}

                    grad_tests = True if 'III' in c or 'IV' in c or c == '' else False
                    test_results = Tests(args.task_name, gen_val, model, EXPERIMENT, save_pickle=False,
                                         subdir_name='init', save_plots=False, model_args=model_args,
                                         grad_tests=grad_tests, seed=s)
                    evaluation = model.evaluate(gen_val, return_dict=True, verbose=False)

                    tf.keras.backend.clear_session()
                    del model

                    results.update(test_results)
                    results.update(evaluation)

                    string_result = json.dumps(results, indent=4, cls=NumpyEncoder)
                    # print(string_result)
                    with open(cfilename, "w") as f:
                        f.write(string_result)
                except Exception as e:
                    print(e)
                    raise e
else:
    fs = [f for f in os.listdir(GENDATA) if args.task_name in f and '.txt' in f]

    print(fs)
    seeds = np.unique([int(f[f.index('_s') + 2:].replace('.txt', '')) for f in fs])
    print(seeds)

unconditioned = {'cI': [], 'cII': [], 'cIII': [], 'cIV': [], }
conditioned = {'cI': [], 'cII': [], 'cIII': [], 'cIV': [], }


def condition_operation(c, config):
    unique_layers_2 = [k.replace('var_in_ma_', '') for k in config.keys() if 'var_in_ma_' in k]
    unique_layers_1 = [k.replace('_grad_III_mean_init', '') for k in config.keys() if '_grad_III_mean_init' in k]

    if c == 'cI':
        result = np.mean([abs(config[l + '_fr_mean_init'] - .5) for l in unique_layers_1])
    elif c == 'cII':
        result = np.mean(
            [2 * config['curr_dis_ma_' + l] / (config['var_in_ma_' + l] + config['var_rec_ma_' + l])
             for l in unique_layers_2]
        )

    elif c == 'cIII':
        result = np.mean(
            [config[l + '_grad_III_init'] / config[l + '_grad_III_mean_init']
             for l in unique_layers_1]
        )
    elif c == 'cIV':
        result = np.mean(
            [config[l + '_grad_IV_init'] / config[l + '_grad_IV_mean_init']
             for l in unique_layers_1]
        )
    else:
        raise NotImplementedError
    return result


if args.plot:
    tag_conditions = ['cI', 'cII', 'cIII', 'cIV']

    for s in tqdm(seeds):
        for c in conditions:
            cfilename = os.path.join(GENDATA, f't{args.task_name}_c{c}_s{s}.txt')
            if os.path.exists(cfilename):
                with open(cfilename) as f:
                    config = json.load(f)
                unique_layers_2 = [k.replace('var_in_ma_', '') for k in config.keys() if 'var_in_ma_' in k]
                unique_layers_1 = ['encoder_0_0', 'encoder_1_0']

                if c == '':
                    for ci in tag_conditions:
                        unconditioned[ci].append(condition_operation(ci, config))
                else:
                    ci = c.replace('_', '').replace('ondition', '')
                    conditioned[ci].append(condition_operation(ci, config))

    print('unconditioned', unconditioned)
    print('conditioned  ', conditioned)

    # make sure the two lists have the same length for plotting conveniently after
    for k in unconditioned.keys():
        min_len = min(len(unconditioned[k]), len(conditioned[k]))
        unconditioned[k] = unconditioned[k][:min_len]
        conditioned[k] = conditioned[k][:min_len]

    n_axis = len(tag_conditions)
    fig, axs = plt.subplots(n_axis, 1, gridspec_kw={'wspace': .0, 'hspace': 0.}, figsize=(6, 6), sharex=True)
    for i in range(n_axis):
        names = ['', tag_conditions[i]]
        if not args.histogram:
            for j in range(len(unconditioned[tag_conditions[i]])):
                axs[i].scatter(names, [unconditioned[tag_conditions[i]][j], conditioned[tag_conditions[i]][j]],
                               color='b')
        else:
            x = np.array(unconditioned[tag_conditions[i]])
            q25, q75 = np.percentile(x, [25, 75])
            bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
            bins = round((x.max() - x.min()) / bin_width)
            print("Freedman–Diaconis number of bins:", bins)
            axs[i].hist(x, bins=bins, color="skyblue", lw=0, density=True, label='no c')

            x = np.array(conditioned[tag_conditions[i]])
            q25, q75 = np.percentile(x, [25, 75])
            bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
            bins = round((x.max() - x.min()) / bin_width)
            print("Freedman–Diaconis number of bins:", bins)
            axs[i].hist(x, bins=bins, color="red", lw=0, density=True, label=tag_conditions[i])

            axs[i].legend()

    axs[0].set_title(args.task_name)

    plt.show()
    plot_filename = os.path.join(GENDATA, f't{args.task_name}_initcond.pdf')
    fig.savefig(plot_filename, bbox_inches='tight')

print('DONE')
shutil.make_archive(EXPERIMENT, 'zip', EXPERIMENT)
