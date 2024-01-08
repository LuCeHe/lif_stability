import tensorflow as tf
import numpy as np

from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real

from pyaromatics.stay_organized.skopt_tools import tqdm_skopt
from lif_stability.neural_models.full_model import build_model


def gp_objective(space, comments, ptoopt, target_firing_rate, net_hyp, batch, words):
    comments = comments
    (task_name, net_name, n_neurons, tau, lr, stack, loss_name, embedding, optimizer_name, tau_adaptation, lr_schedule,
     weight_decay, clipnorm, initializer, comments, in_len, n_in, out_len, n_out, final_epochs) = net_hyp

    @use_named_args(space)
    def objective(**params):
        layer = [v for k, v in params.items() if 'layer' in k]
        print('layer', layer)
        comments_i = comments
        for i, l in enumerate(layer):
            comments_i += f'_{ptoopt}{i}:{l}'  # '_wrecm0:-.1_v0m0:.2_v0m1:.3'

        model = build_model(
            task_name, net_name, n_neurons, tau, lr, stack,
            loss_name, embedding, optimizer_name, tau_adaptation, lr_schedule, weight_decay, clipnorm,
            initializer, comments_i, in_len, n_in, out_len, n_out, final_epochs,
        )

        # evaluate so you evaluate on the whole training set, or a few steps_per_epoch
        evaluation = model.evaluate(((batch, words),), return_dict=True, verbose=False)

        tf.keras.backend.clear_session()
        del model

        fs = [v for k, v in evaluation.items() if 'firing_rate' in k]
        loss = np.mean([(np.abs(f - target_firing_rate) * 10) ** 2 for f in fs])
        print(fs)
        # print(loss)

        return loss

    return objective


def sparsity_gp(comments, ptoopt, target_firing_rate, net_hyp, args, batch, words, stack):
    space = [Real(-10, 10, name='layer_{}'.format(i)) for i in range(stack)]

    res_gp = gp_minimize(
        gp_objective(space, comments, ptoopt, target_firing_rate, net_hyp, batch, words), space, n_calls=args.n_calls,
        random_state=args.seed,
        callback=[tqdm_skopt(total=args.n_calls, desc="Gaussian Process")]
    )
    print("Best parameters: ", res_gp.x)
    print("Best loss:       ", res_gp.fun)
