"""
WTA inspired by
Z. Jonke, R. Legenstein, S. Habenschuss, and W. Maass.
Feedback inhibition shapes emergent computational properties of cortical
microcircuit motifs. Journal of Neuroscience, 37(35):8511-8523, 2017

"""

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

from pyaromatics.keras_tools.esoteric_layers.surrogated_step import SpikeFunction


class WTA(tf.keras.layers.Layer):
    """
    Winner Takes All
    """

    def get_config(self):
        return self.init_args

    def __init__(self, num_neurons, tau=20., tau_adaptation=20, spike_dropout=.3, spike_dropin=.3,
                 in_neurons=None, out_neurons=None, dampening_factor=.3, ref_period_i=2., ref_period_e=5.):
        super().__init__()

        self.n_exc = num_neurons  # int(4 * num_neurons / 5)
        self.n_inh = int(1 * num_neurons / 4)

        self.state_size = (self.n_inh, self.n_exc,) * 3
        self.decay = np.exp(-1 / tau)
        self.thr = .03  # .03

        self.init_args = dict(
            num_neurons=num_neurons, tau=tau, tau_adaptation=tau_adaptation,
            ref_period_e=ref_period_e, ref_period_i=ref_period_i,
            dampening_factor=dampening_factor,
            spike_dropout=spike_dropout, spike_dropin=spike_dropin,
            in_neurons=in_neurons, out_neurons=out_neurons)
        self.__dict__.update(self.init_args)

    def build(self, input_shape):
        n_input = input_shape[-1]

        self.w_input2exc = self.add_weight(
            shape=(n_input, self.n_exc),
            initializer=tf.keras.initializers.RandomNormal(stddev=1. / np.sqrt(n_input)),
            name='w_input2exc')

        self.w_i2i = self.add_disconnected_weights(p=.55, shape=(self.n_inh, self.n_inh), diagonal=True, name='w_i2i')
        self.w_i2e = self.add_disconnected_weights(p=.60, shape=(self.n_inh, self.n_exc), diagonal=False, name='w_i2e')
        self.w_e2i = self.add_disconnected_weights(p=.575, shape=(self.n_exc, self.n_inh), diagonal=False, name='w_e2i')

        super().build(input_shape)

    def add_disconnected_weights(self, p=.6, shape=(3, 3), diagonal=True, name=''):
        w = self.add_weight(
            shape=shape,
            initializer=tf.keras.initializers.Orthogonal(),
            # initializer=tf.keras.initializers.RandomNormal(stddev=1. / np.sqrt(shape[0])),
            name=name)

        if diagonal:
            assert shape[0] == shape[1]
            disconnect_mask = tf.cast(np.diag(np.ones(shape[0], dtype=np.bool)), tf.bool,
                                      name='disconnect_mask')
            w = tf.where(disconnect_mask, tf.zeros_like(w), w)

        disconnect_mask = tf.cast(np.random.choice(2, size=shape, p=[p, 1 - p]), tf.bool,
                                  name='disconnect_mask')
        w = tf.where(disconnect_mask, tf.zeros_like(w), w)
        w = .01 * tf.exp(w)
        return w

    def call(self, inputs, states):
        old_spike_i = states[0]
        old_spike_e = states[1]
        old_v_i = states[2]
        old_v_e = states[3]
        last_spike_distance_i = states[4]
        last_spike_distance_e = states[5]

        i_in_i = old_spike_e @ self.w_e2i - old_spike_i @ self.w_i2i
        i_in_e = inputs @ self.w_input2exc - old_spike_i @ self.w_i2e

        i_reset_i = - self.thr * old_spike_i
        new_v_i = self.decay * old_v_i + (1 - self.decay) * i_in_i + i_reset_i
        i_reset_e = - self.thr * old_spike_e
        new_v_e = self.decay * old_v_e + (1 - self.decay) * i_in_e + i_reset_e

        v_sc_i = (new_v_i - self.thr) / self.thr
        z_i = SpikeFunction(v_sc_i, self.dampening_factor)
        z_i.set_shape(v_sc_i.get_shape())

        v_sc_e = (new_v_e - self.thr) / self.thr
        z_e = SpikeFunction(v_sc_e, self.dampening_factor)
        z_e.set_shape(v_sc_e.get_shape())

        # refractoriness
        non_refractory_neurons_i = tf.cast(last_spike_distance_i >= self.ref_period_i, tf.float32)
        z_i = non_refractory_neurons_i * z_i
        new_last_spike_distance_i = (last_spike_distance_i + 1) * (1 - z_i)

        non_refractory_neurons_e = tf.cast(last_spike_distance_e >= self.ref_period_e, tf.float32)
        z_e = non_refractory_neurons_e * z_e
        new_last_spike_distance_e = (last_spike_distance_e + 1) * (1 - z_e)

        output = (z_e, z_i, v_sc_e, v_sc_i)
        new_state = (z_i, z_e, new_v_i, new_v_e, new_last_spike_distance_i, new_last_spike_distance_e)
        return output, new_state
