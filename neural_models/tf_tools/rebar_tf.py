import numpy as np
import tensorflow as tf


def Q_func(z):
    h1 = tf.layers.dense(2. * z - 1., 20, tf.nn.relu, name="q_1", use_bias=True)
    out = tf.layers.dense(h1, 1, name="q_out", use_bias=True)
    scale = tf.get_variable(
        "q_scale", shape=[1], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=True
    )
    return scale[0] * out


def safe_log_prob(x, eps=1e-8):
    return tf.math.log(tf.clip_by_value(x, eps, 1.0))


def safe_clip(x, eps=1e-8):
    return tf.clip_by_value(x, eps, 1.0)


def gs(x):
    return x.get_shape().as_list()


def softplus(x):
    '''
    Let m = max(0, x), then,

    sofplus(x) = log(1 + e(x)) = log(e(0) + e(x)) = log(e(m)(e(-m) + e(x-m)))
                         = m + log(e(-m) + e(x - m))

    The term inside of the log is guaranteed to be between 1 and 2.
    '''
    m = tf.maximum(tf.zeros_like(x), x)
    return m + tf.math.log(tf.exp(-m) + tf.math.exp(x - m) + 1e-8)


def bernoulli_loglikelihood(b, log_alpha):
    return b * (-softplus(-log_alpha)) + (1 - b) * (-log_alpha - softplus(-log_alpha))


def create_reparam_variables(log_alpha, u, eps=1e-8):
    # logistic reparameterization z = g(u, log_alpha)
    z = log_alpha + safe_log_prob(u) - safe_log_prob(1 - u)

    # b = H(z)
    b = tf.cast(tf.stop_gradient(z > 0), tf.float32)

    # g(u', log_alpha) = 0
    a = tf.exp(log_alpha)
    theta = safe_clip(a / (1 + a))
    v = u
    v_prime = (1 - b) * v * (1 - theta) + b * (v * theta + 1 - theta)
    z_tilde = log_alpha + safe_log_prob(v_prime) - safe_log_prob(1 - v_prime)

    return z, z_tilde, b


def create_reparam_variables_old(log_alpha, u, eps=1e-8):
    original_shape = log_alpha.shape
    u = tf.reshape(u, [-1])
    log_alpha = tf.reshape(log_alpha, [-1])

    # logistic reparameterization z = g(u, log_alpha)
    z = log_alpha + safe_log_prob(u) - safe_log_prob(1 - u)

    # b = H(z)
    b = tf.to_float(tf.stop_gradient(z > 0))

    # g(u', log_alpha) = 0
    u_prime = tf.nn.sigmoid(-log_alpha)
    v_1 = (u - u_prime) / tf.clip_by_value(1 - u_prime, eps, 1)
    v_1 = tf.clip_by_value(v_1, 0, 1)
    v_1 = tf.stop_gradient(v_1)
    v_1 = v_1 * (1 - u_prime) + u_prime
    v_0 = u / tf.clip_by_value(u_prime, eps, 1)
    v_0 = tf.clip_by_value(v_0, 0, 1)
    v_0 = tf.stop_gradient(v_0)
    v_0 = v_0 * u_prime

    v = tf.where(u > u_prime, v_1, v_0)
    v = tf.check_numerics(v, 'v sampling is not numerically stable.')
    v = v + tf.stop_gradient(-v + u)  # v and u are the same up to numerical errors

    # tf.summary.histogram("u-v", u - v)

    z_tilde = log_alpha + safe_log_prob(v) - safe_log_prob(1 - v)

    z = tf.reshape(z, original_shape)
    z_tilde = tf.reshape(z_tilde, original_shape)
    b = tf.reshape(b, original_shape)

    return z, z_tilde, b


def bernoulli_loglikelihood_derivative(b, log_alpha):
    assert gs(b) == gs(log_alpha)
    sna = tf.sigmoid(-log_alpha)
    return b * sna - (1 - b) * (1 - sna)


def PseudogradOptimizer(*args, **kwargs):
    if kwargs['name'] == 'REBAR':
        return REBAROptimizer(*args, **kwargs)
    elif kwargs['name'] == 'relaxedREBAR':
        return RelaxedREBAROptimizer(*args, **kwargs)
    else:
        raise NotImplementedError


class REBAROptimizer(object):
    def __init__(self, loss, log_alpha=None, dim=None, name="REBAR", learning_rate=.01, n_samples=1, z=None,
                 z_tilde=None, b=None, noise=None, log_temperature=None, variance_optimizer=None):
        self.__dict__.update(loss=loss, log_alpha=log_alpha, dim=dim, name=name, learning_rate=learning_rate,
                             n_samples=n_samples, z=z, z_tilde=z_tilde, b=b, noise=noise,
                             log_temperature=log_temperature
                             )

        """ extra steps for children classes """
        self.initial_extra_steps()
        """ model parameters """
        self._create_model_parameters()
        """ reparameterization noise """
        self._create_reparam_variables()
        """ relaxed loss evaluations """
        self._create_loss_evaluations()
        """ gradvars for optimizers """
        self._create_gradvars()
        """ variance reduction optimization operation """
        if variance_optimizer == None:
            variance_optimizer = tf.train.AdamOptimizer(.001 * learning_rate)
        self.variance_optimizer = variance_optimizer
        self.variance_reduction_op = self.variance_optimizer.apply_gradients(self.variance_gradvars)
        """ extra steps for children classes """
        self.final_extra_steps()
        """ plots """
        self.create_tf_summaries()

    def create_tf_summaries(self):

        tf.summary.scalar("f_b", tf.reduce_mean(self.f_b))
        tf.summary.scalar("f_z_tilde", tf.reduce_mean(self.f_z_tilde))
        tf.summary.scalar("f_z", tf.reduce_mean(self.f_z))
        tf.summary.histogram("temperature", self.temperature)
        tf.summary.histogram("theta", self.theta)
        tf.summary.histogram("rebar_gradient", self.rebar)
        tf.summary.histogram("reinforce_gradient", self.reinforce)

    def _create_model_parameters(self):
        self.original_shape = gs(self.log_alpha)
        self.batch_size = self.original_shape[0]
        time_steps = self.original_shape[1]
        n_vars = self.original_shape[2]
        self.flat_log_alpha = tf.reshape(self.log_alpha, [-1])

        a = tf.exp(self.log_alpha)
        self.theta = a / (1 + a)

        # expanded version for internal purposes
        # self._log_alpha = tf.expand_dims(self.log_alpha, 0)
        # n_vars = int(self.dim / self.batch_size)
        self.n_vars = n_vars

        if self.log_temperature is None:
            self.batch_log_temperature = tf.Variable(
                [np.log(.5) for _ in range(n_vars)],
                trainable=False,
                name='log_temperature',
                dtype=tf.float32)
        else:
            self.batch_log_temperature = tf.reshape(
                tf.tile(tf.expand_dims(self.log_temperature, 0), [time_steps, 1]), [-1])

        log_temperature = tf.reshape(tf.tile(tf.expand_dims(self.batch_log_temperature, 0),
                                             [self.batch_size, 1]),
                                     [-1])
        tiled_log_temperature = tf.tile([log_temperature], [self.n_samples, 1])
        self.temperature = tf.exp(tiled_log_temperature)

        self.batch_eta = tf.Variable(
            [1.0 for _ in range(n_vars)],
            trainable=False,
            name='eta',
            dtype=tf.float32)
        self.eta = tf.reshape(tf.tile(tf.expand_dims(self.batch_eta, 0), [self.batch_size, 1]), [-1])

    def _create_reparam_variables(self, eps=1e-8):
        reparam_variables = [self.b, self.z, self.z_tilde]
        if any([tensor == None for tensor in reparam_variables]):
            # noise for generating z
            u = self.noise if not self.noise == None else tf.random_uniform([self.n_samples, self.dim],
                                                                            dtype=tf.float32)
            self.z, self.z_tilde, self.b = create_reparam_variables(self.log_alpha, u, eps)
        else:
            self.b, self.z, self.z_tilde = [tf.reshape(tensor, [-1]) for tensor in reparam_variables]

    def _create_loss_evaluations(self):
        """
        produces f(b), f(sig(z)), f(sig(z_tilde))
        """
        # relaxed inputs
        sig_z = tf.nn.sigmoid(self.z / self.temperature)
        sig_z_tilde = tf.nn.sigmoid(self.z_tilde / self.temperature)

        # evaluate loss
        self.f_b = tf.reshape(self.loss(tf.reshape(self.b, self.original_shape)), [-1])
        self.f_z = tf.reshape(self.loss(tf.reshape(sig_z, self.original_shape)), [-1])
        self.f_z_tilde = tf.reshape(self.loss(tf.reshape(sig_z_tilde, self.original_shape)), [-1])

    def _create_gradvars(self):
        """
        produces d[log p(b)]/d[log_alpha], d[f(sigma_theta(z))]/d[log_alpha], d[f(sigma_theta(z_tilde))]/d[log_alpha]
        """
        # log_alpha = self.log_alpha
        d_log_p_d_log_alpha = bernoulli_loglikelihood_derivative(self.b, self.flat_log_alpha)

        term1 = ((self.f_b - self.eta * self.f_z_tilde) * d_log_p_d_log_alpha)[0]

        # d[f(sigma_theta(z))]/d[log_alpha] - eta * d[f(sigma_theta(z_tilde))]/d[log_alpha]
        term2 = tf.gradients(tf.reduce_mean(self.f_z - self.f_z_tilde),
                             self.log_alpha)[0]
        term2 = tf.reshape(term2, [-1])

        # rebar gradient estimator
        rebar = term1 + self.eta * term2
        reinforce = (self.f_b * d_log_p_d_log_alpha)

        # now compute gradients of the variance of this wrt other parameters, and regularize
        variance_loss = (tf.square(1e-4 * rebar) / self.batch_size
                         )
        variance_loss = tf.reduce_sum(variance_loss)
        self.variance_loss = variance_loss

        # eta
        d_var_d_eta = tf.gradients(variance_loss,
                                   self.batch_eta)[0]
        # temperature
        d_var_d_temperature = tf.gradients(variance_loss,
                                           self.log_temperature)[0]
        self._rebar = rebar
        self.rebar = tf.reshape(rebar, self.original_shape)
        self.reinforce = tf.reshape(reinforce, self.original_shape)

        self.rebar_gradvars = [(rebar, self.log_alpha)]
        self.variance_gradvars = [(d_var_d_eta, self.batch_eta),
                                  (d_var_d_temperature, self.log_temperature)]
        self.variance_vars = [self.batch_eta, self.log_temperature]

    def final_extra_steps(self):
        pass

    def initial_extra_steps(self):
        pass


class RelaxedREBAROptimizer(REBAROptimizer):
    def initial_extra_steps(self):
        self.q_func = Q_func

    def _create_relaxed_gradvars(self):
        self.Q_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.Q_vars = [v for v in tf.trainable_variables() if "Q_func" in v.name]
        self._Q_gradvars()
        self.Q_opt_op = self.Q_optimizer.apply_gradients(self.Q_gradvars)
        old_var_op = self.variance_reduction_op

        with tf.control_dependencies([self.Q_opt_op, old_var_op]):
            self.variance_reduction_op = tf.no_op()

    def _create_loss_evaluations(self):
        """
        produces f(b), f(sig(z)), f(sig(z_tilde))
        """

        # relaxed inputs
        sig_z = tf.nn.sigmoid(self.z / self.temperature)
        sig_z_tilde = tf.nn.sigmoid(self.z_tilde / self.temperature)

        # evaluate loss
        f_b = tf.reshape(self.loss(tf.reshape(self.b, self.original_shape)), [-1])
        z_inp = tf.reshape(sig_z, self.original_shape)
        z_tilde_inp = tf.reshape(sig_z_tilde, self.original_shape)
        l_z = self.loss(z_inp)
        l_z_tilde = self.loss(z_tilde_inp)

        with tf.variable_scope("Q_func"):
            f_z = tf.reshape(self.q_func(z_inp) + l_z, [-1])

        with tf.variable_scope("Q_func", reuse=True):
            f_z_tilde = tf.reshape(self.q_func(z_tilde_inp) + l_z_tilde, [-1])

        self.f_b = f_b
        self.f_z = tf.reduce_mean(f_z)
        self.f_z_tilde = tf.reduce_mean(f_z_tilde)

    def _Q_gradvars(self):
        """
        produces d[log p(b)]/d[log_alpha], d[f(sigma_theta(z))]/d[log_alpha], d[f(sigma_theta(z_tilde))]/d[log_alpha]
        """
        self.Q_gradvars = []
        for var in self.Q_vars:
            d_var_d_v = tf.gradients(
                tf.reduce_sum(tf.square(self._rebar)) / self.batch_size,
                var)[0]

            self.Q_gradvars.append((d_var_d_v, var))

    def final_extra_steps(self):
        self._create_relaxed_gradvars()
