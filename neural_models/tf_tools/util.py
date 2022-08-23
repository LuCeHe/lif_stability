import numpy as np
import tensorflow as tf


def switch_time_and_batch_dimension(_tensor):
    rank = len(_tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[1] = 1, 0
    if _tensor.dtype == tf.bool:
        _tensor = tf.cast(_tensor, tf.int64)
    res = tf.transpose(_tensor, perm, name='switch_time_and_batch_dimension')
    if _tensor.dtype == tf.bool:
        return tf.cast(res, tf.bool)
    return res


def safe_log_prob(_x, eps=1e-8):
    return tf.math.log(tf.clip_by_value(_x, eps, 1.))


def coupling(param, b, u):
    uprime = tf.nn.sigmoid(-param)
    v = ((1. - b) * (u / tf.clip_by_value(uprime, 1e-8, 1.)) +
         b * ((u - uprime) / tf.clip_by_value(1. - uprime, 1e-8, 1.)))
    return tf.clip_by_value(v, 0., 1.)


def exp_convolve(tensor, decay=.8, tau=None):
    if not tau is None:
        decay = tf.cast(tf.math.exp(-1 / tau), tf.float32)

    def scan_fun(_acc, _t):
        return _acc * decay + (1 - decay) * _t

    filtered = tf.scan(scan_fun, tf.transpose(tensor, (1, 0, 2)))
    return tf.transpose(filtered, (1, 0, 2))




def target_loss(target=-120, denominator=40):
    def tloss(rate):
        loss = tf.square(rate - target / denominator)
        return loss

    return tloss


def gate(z):
    return (1. + tf.sign(z)) / 2.


def softgate(z, t):
    return tf.nn.sigmoid(z / t)


def binary_forward(param, u):
    z = param + safe_log_prob(u) - safe_log_prob(1. - u)
    return z


@tf.custom_gradient
def binary_forward_cg(param):
    def grad(dy):
        dz_dv_scaled = tf.maximum(1 - tf.abs(param), 0)
        return dz_dv_scaled * dy

    noise = tf.random.uniform(tf.shape(param))
    z = param + safe_log_prob(noise) - safe_log_prob(1. - noise)
    return z, grad


def binary_forward_old(param):
    noise = tf.random.uniform(tf.shape(param), dtype=tf.float32)
    z = param + safe_log_prob(noise) - safe_log_prob(1. - noise)
    return z


def cond_noise(param, b, noise=None):
    """draw reparameterization z of binary variable b from p(z|b)."""
    if noise is not None:
        v = noise
    else:
        v = tf.random.uniform(param.shape.as_list(), dtype=param.dtype)
    uprime = tf.nn.sigmoid(-param)
    ub = b * (uprime + (1. - uprime) * v) + (1. - b) * uprime * v
    ub = tf.clip_by_value(ub, 0., 1.)
    return ub



def string_to_contrastive_coef(comments):
    coef_disorder, coef_random = 0, 0
    if 'contrastive10' in comments:
        coef_disorder, coef_random = .1, 0.
    elif 'contrastive01' in comments:
        coef_disorder, coef_random = 0., .1
    elif 'contrastive11' in comments:
        coef_disorder, coef_random = .1, .1

    if '10contrastive' in comments:
        coef_disorder *= 10
        coef_random *= 10
    elif '100contrastive' in comments:
        coef_disorder *= 100
        coef_random *= 100

    return coef_disorder, coef_random