import tensorflow as tf
from lif_stability.neural_models.lsnn import baseLSNN


class Izhikevich(baseLSNN):
    """
    LSNN
    """

    def __init__(self, num_neurons, tau=20., beta=1.8, tau_adaptation=20,
                 dampening_factor=.3, ref_period=2., thr=.03, inh_exc=1., spike_dropout=.3,
                 in_neurons=None, out_neurons=None):
        super().__init__(num_neurons, tau, beta, tau_adaptation,
                         dampening_factor, ref_period, thr, inh_exc, spike_dropout,
                         in_neurons, out_neurons)

        self.a = 0.1 * tf.random.uniform([1, num_neurons]) + 0.01
        self.b = 0.1 * tf.random.uniform([1, num_neurons]) + 0.18
        self.c = .020 * tf.random.uniform([1, num_neurons]) - .067
        self.d = .009 * tf.random.uniform([1, num_neurons]) + 0.0002

    def call(self, inputs, states):
        old_spike = states[0]
        old_v = states[1]
        old_u = states[2]
        last_spike_distance = states[3]

        batch_size = tf.shape(inputs)[0]
        external_current = tf.concat([inputs @ self.input_weights,
                                      tf.zeros((batch_size, self.num_neurons - self.in_neurons))], 1)
        i_in = external_current + (self.switch_off * self.inh_exc * old_spike) @ self.recurrent_transform()

        # i_reset = - thr * old_spike

        decay = tf.exp(-1 / self.tau)
        v_eq = .04 * tf.math.square(old_v) + 5 * old_v + 140 - old_u
        new_v = decay * old_v + (1 - decay) * (i_in + v_eq)
        new_u = self.a * (self.b * old_v - old_u)

        v_sc = (new_v - self.thr) / self.thr

        z = SpikeFunction(v_sc, self.dampening_factor)
        z.set_shape(v_sc.get_shape())

        # refractoriness
        non_refractory_neurons = tf.cast(last_spike_distance >= self.ref_period, tf.float32)
        z = non_refractory_neurons * z
        new_last_spike_distance = (last_spike_distance + 1) * (1 - z)

        new_v = (1 - z) * new_v + z * self.c
        new_u = (1 - z) * new_u + z * (new_u + self.d)

        output = (z[:, -self.out_neurons:], new_v, new_u, v_sc)
        new_state = (z, new_v, new_u, new_last_spike_distance)
        return output, new_state
