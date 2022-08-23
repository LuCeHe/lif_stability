import numpy as np
import tensorflow as tf
import tensorflow.keras as k

from stochastic_spiking.tf_tools.rebar_tf import create_reparam_variables


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


class StochasticLSNN_Franz(StochasticRSNN):
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


class StochasticLSNN(tf.keras.layers.Layer):
    def __init__(self, num_neurons,
                 tau=10., beta=1.7, tau_adaptation=20, ref_period=2.,
                 thr=.03, inh_exc=1.):
        super().__init__()
        self.__dict__.update(num_neurons=num_neurons, tau=tau, tau_adaptation=tau_adaptation,
                             ref_period=ref_period, thr=thr, inh_exc=inh_exc)

        self.state_size = (num_neurons, num_neurons, num_neurons, num_neurons)

        n_regular = int(num_neurons / 2)
        self.beta = np.concatenate([np.zeros(n_regular), np.ones(num_neurons - n_regular) * beta])

    def build(self, input_shape):
        n_input = input_shape[-1]  # input_shape[-1] - self.num_neurons

        self.input_weights = self.add_weight(
            shape=(n_input, self.num_neurons),
            initializer=tf.keras.initializers.RandomNormal(
                stddev=1. / tf.sqrt(np.array(n_input).astype(np.float32))),  # K.sqrt(tf.cast(n_input, tf.float32)))
            name='input_weights')

        self.mask = tf.ones((self.num_neurons, self.num_neurons)) - tf.eye(self.num_neurons)
        self.recurrent_weights = self.add_weight(
            shape=(self.num_neurons, self.num_neurons),
            initializer=tf.keras.initializers.Orthogonal(),
            name='recurrent_weights')

        self.baseline_current = self.add_weight(
            shape=(self.num_neurons,),
            initializer=tf.keras.initializers.RandomNormal(
                stddev=1. / np.sqrt(n_input)),  # K.sqrt(tf.cast(n_input, tf.float32)))
            name='baseline_current')

        """
        parameter2trainable = {k: v for k, v in self.__dict__.items()
                               if k in ['tau', 'tau_adaptation', 'thr', 'inh_exc']}
        for k, p in parameter2trainable.items():
            if k in ['tau', 'tau_adaptation', 'thr']:
                # initializer = tf.keras.initializers.Constant(value=p)
                initializer = tf.keras.initializers.TruncatedNormal(mean=p, stddev=3 * p / 7)
            else:
                initializer = tf.keras.initializers.RandomNormal(stddev=1. / np.sqrt(n_input))

            p = self.add_weight(
                shape=(self.num_neurons,),
                initializer=initializer,
                name=k)

            self.__dict__.update({k: p})
        """
        self.inh_exc = 2*np.random.choice(2, (self.num_neurons,))-1
        self.built = True

    def call(self, inputs, states, constants):
        # inputs, noise = inputs[..., :-self.num_neurons], inputs[..., -self.num_neurons:]
        old_spike = states[0]
        old_v = states[1]
        old_a = states[2]
        last_spike_distance = states[3]

        rw = tf.math.softplus(self.mask * self.recurrent_weights)
        i_in = inputs @ self.input_weights + (self.inh_exc * old_spike) @ rw + self.baseline_current
        # i_in = inputs @ self.input_weights + old_spike @ (self.mask*self.recurrent_weights) + self.baseline_current
        decay = tf.cast(tf.exp(-1 / self.tau), tf.float32)
        rho = tf.cast(tf.exp(-1 / self.tau_adaptation), tf.float32)

        new_a = rho * old_a + (1 - rho) * old_spike
        thr = self.thr + new_a * self.beta

        i_reset = - thr * old_spike
        new_v = decay * old_v + (1 - decay) * i_in + i_reset

        v_sc = (new_v - thr) / thr

        log_alpha = v_sc
        noise = tf.random.uniform(tf.shape(old_spike), dtype=tf.float32)
        z, z_tilde, hard_b = create_reparam_variables(log_alpha, noise)

        # refractoriness
        non_refractory_neurons = tf.cast(last_spike_distance > self.ref_period, tf.float32)
        hard_b = non_refractory_neurons * hard_b
        new_last_spike_distance = (last_spike_distance + 1) * (1 - hard_b)

        # train on continuous approx, test on non continuous
        bzz_idx, log_temperature = constants
        temperature = tf.exp(log_temperature)
        output = select_output(bzz_idx, hard_b, z, z_tilde, temperature)

        complete_output = (output, new_v, thr, v_sc)
        new_state = (output, new_v, new_a, new_last_spike_distance)
        return complete_output, new_state


tf_is_identical = lambda x, y: tf.reduce_all(tf.equal(x, y))


@tf.function
def select_output(bzz_idx, b, z, z_tilde, temperature):
    cond_0 = bzz_idx == 0
    cond_1 = bzz_idx == 1
    cond_2 = bzz_idx == 2

    pred_fn_pairs = [(cond_0, lambda: tf.identity(b, name='b')),
                     (cond_1, lambda: tf.nn.sigmoid(z / temperature, name='sig_z')),
                     (cond_2, lambda: tf.nn.sigmoid(z_tilde / temperature, name='sig_z_tilde')),
                     ]
    output = tf.case(pred_fn_pairs)
    return output


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
