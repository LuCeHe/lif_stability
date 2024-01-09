import numpy as np
import tensorflow as tf
import tensorflow.keras as k


def safe_log_prob(x, eps=1e-8):
    return tf.math.log(tf.clip_by_value(x, eps, 1.0))


def reparametrize(log_alpha, noise):
    return log_alpha + safe_log_prob(noise) - safe_log_prob(1 - noise)


def concrete_relaxation(z, temp):
    return tf.nn.sigmoid(z / temp)


class StochasticRSNN(k.layers.Layer):
    def __init__(self, num_neurons, tau=20, base_rate=5, hard=True, n_refractory=5, temperature=.5, alpha=1.):
        super().__init__()
        self.num_neurons = num_neurons
        #                  membrane,    spikes       refrac
        self.state_size = (num_neurons, num_neurons, num_neurons)
        self.decay = np.exp(-1 / tau)

        self.input_weights = None
        self.recurrent_weights = None
        self.disconnect_mask = None
        self.thr = 1.
        dt = .001
        if base_rate is None:
            base_rate = 5
        _tmp = base_rate * dt * np.exp(-base_rate * dt)
        self.theta_0 = -np.log((1 - _tmp) / _tmp)

        self.hard = hard
        self.n_refractory = n_refractory
        self.temperature = temperature
        self.alpha = alpha

    def build(self, input_shape):
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.num_neurons),
                                             initializer=k.initializers.RandomNormal(
                                                 stddev=1. / np.sqrt(input_shape[-1])),
                                             name='input_weights')
        self.disconnect_mask = tf.cast(np.diag(np.ones(self.num_neurons, dtype=np.bool)), tf.bool)
        self.recurrent_weights = self.add_weight(
            shape=(self.num_neurons, self.num_neurons),
            initializer=k.initializers.Orthogonal(),
            name='recurrent_weights')
        super().build(input_shape)

    def spike_mechanism(self, shifted_v, old_r):
        u = tf.random.uniform(shape=tf.shape(shifted_v), dtype=shifted_v.dtype)
        z = reparametrize(shifted_v, u)

        is_refractory = tf.greater(old_r, .99)

        b = concrete_relaxation(z, self.temperature)
        hard_b = tf.cast(z > 0, tf.float32)
        b = tf.where(is_refractory, tf.zeros_like(b), b)
        hard_b = tf.where(is_refractory, tf.zeros_like(hard_b), hard_b)

        r = tf.clip_by_value(old_r + self.n_refractory * b - 1, 0., float(self.n_refractory - 1))

        if self.hard:
            output = tf.stop_gradient(hard_b - b) + b
        else:
            output = b

        return output, r

    def call(self, inputs, states):
        old_v = states[0]
        old_spike = states[1]
        old_r = states[2]

        recurrent_weights = tf.where(
            self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)
        i_in = inputs @ self.input_weights + old_spike @ recurrent_weights
        i_reset = -self.decay * old_v * old_spike
        new_v = self.decay * old_v + i_in + i_reset
        v_sc = tf.clip_by_value(new_v / self.thr, 0, np.inf)
        l_p_z = (tf.pow(v_sc, self.alpha) - 1.) * (-self.theta_0)
        p_z = tf.nn.sigmoid(l_p_z)

        output, r = self.spike_mechanism(l_p_z, old_r)

        new_state = (new_v, output, r)
        return (output, new_v, p_z), new_state


class StochasticLSNN(StochasticRSNN):
    def __init__(self, num_neurons, tau=20, base_rate=5, hard=True, n_refractory=5, temperature=.5, alpha=2., beta=0.,
                 tau_adaptation=200.):
        super().__init__(num_neurons, tau, base_rate, hard, n_refractory, temperature, alpha)

        self.rho = np.exp(-1 / tau_adaptation)
        self.beta = beta
        self.state_size = tuple([num_neurons] * 4)

    def call(self, inputs, states):
        old_v = states[0]
        old_spike = states[1]
        old_r = states[2]
        old_a = states[3]

        recurrent_weights = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights),
                                     self.recurrent_weights)
        i_in = inputs @ self.input_weights + old_spike @ recurrent_weights
        # i_reset = -self.decay * old_v * old_spike
        i_reset = -self.decay * self.thr * old_spike
        new_v = self.decay * old_v + i_in + i_reset
        new_a = self.rho * old_a + (1 - self.rho) * old_spike

        v_sc = tf.clip_by_value((new_v - self.beta * new_a) / self.thr, 0, np.inf)
        l_p_z = (tf.pow(v_sc, self.alpha) - 1.) * (-self.theta_0)
        p_z = tf.nn.sigmoid(l_p_z)

        output, r = self.spike_mechanism(l_p_z, old_r)

        new_state = (new_v, output, r, new_a)
        return (output, new_v, new_a, p_z), new_state


class Readout(k.layers.Layer):
    def __init__(self, num_neurons, tau=20):
        super().__init__()
        self.state_size = num_neurons
        self.readout = k.layers.Dense(num_neurons)
        self.decay = np.exp(-1 / tau)

    def call(self, inputs, state):
        state = self.decay * state[0] + (1 - self.decay) * self.readout(inputs)
        return state, (state,)


if __name__ == '__main__':
    n_time = 1000
    n_input = 50
    poisson_input = tf.cast(tf.random.uniform(shape=(1, n_time, n_input)) < .05, tf.float32)
    cell = StochasticLSNN(50, base_rate=1, n_refractory=0, hard=False, alpha=1., beta=1.6)
    rnn = k.layers.RNN(cell, return_sequences=True)
    spike, voltage, threshold, p_spike = rnn(poisson_input)

    mean_rate = tf.reduce_mean(spike) * 1000

    print(f'rate {mean_rate.numpy()}')

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, figsize=(9, 6), sharex=True)
    axes[0].pcolormesh(poisson_input[0].numpy().T, cmap='Greys')
    axes[1].pcolormesh(spike[0].numpy().T, cmap='Greys')
    abs_max = np.max(np.abs(voltage[0]))
    # axes[1].pcolormesh(voltage[0].numpy().T, vmin=-abs_max, vmax=abs_max, cmap='bwr')
    axes[2].plot(voltage[0].numpy(), alpha=.2, color='b', lw=.5)
    axes[2].set_ylim([-1 * cell.thr, 2 * cell.thr])
    axes[3].plot(p_spike[0].numpy(), alpha=.2, color='b', lw=.5)
    axes[4].plot(threshold[0].numpy(), alpha=.2, color='b', lw=.5)
    # axes[3].set_ylim([0, 1])
    plt.show()