import numpy as np
import tensorflow as tf


def pseudo_derivative(v_scaled, dampening_factor):
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


@tf.custom_gradient
def spike_function16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


class ZeroDiagonalConstraint(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return w - tf.linalg.diag(tf.linalg.diag_part(w))


class FastALIFCell(tf.keras.layers.Layer):
    def __init__(self, units, n_in=100, tau=20., tau_adaptation=200, beta=.18, adaptive_fraction=-1., thr=1., dt=1,
                 n_delay=5, n_refractory=5, dampening_factor=.3,
                 sparse_weights=True, rec_conn_density=0.1, **kwargs):
        super().__init__()
        assert n_refractory == n_delay or 'Different values for refractory period and synapse delay are not supported'
        self.units = units

        self.sparse_weights = sparse_weights
        self.rec_conn_density = rec_conn_density

        self._dt = float(dt)
        self._n_substeps = int(n_delay / dt)
        self._decay = tf.cast(tf.exp(-dt / tau), self._dtype_policy.compute_dtype)
        self._n_refractory = n_refractory
        self._decay_kernel = self._decay**tf.cast(
            tf.range(0, self._n_refractory), self._dtype_policy.compute_dtype)[::-1, None, None, None]
        self._decay_adaptation = tf.cast(tf.exp(-dt / tau_adaptation), self._dtype_policy.compute_dtype)
        self._decay_kernel_adaptation = self._decay_adaptation**tf.cast(
            tf.range(-1, self._n_refractory - 1), self._dtype_policy.compute_dtype)[::-1, None, None, None]
        if adaptive_fraction > 0.:
            n_adaptive = int(adaptive_fraction * units)
            self.beta = tf.concat(
                (tf.ones((1, n_adaptive), dtype=self._dtype_policy.compute_dtype),
                 tf.zeros((1, units - n_adaptive), dtype=self._dtype_policy.compute_dtype)), -1) * beta
        else:
            self.beta = tf.ones((1, units), dtype=self._dtype_policy.compute_dtype) * beta

        self.input_layer = tf.keras.layers.Dense(
            units, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1. / np.sqrt(n_in + self.units)))
        
        if not self.sparse_weights:
            self.recurrent_weights = None

        self.threshold = tf.cast(thr, self._dtype_policy.compute_dtype)
        self._dampening_factor = tf.cast(dampening_factor, self._dtype_policy.compute_dtype)

        self.state_size = (units, units, units * self._n_substeps)

    def zero_state(self, batch_size, dtype=tf.float32):
        v0 = tf.zeros((batch_size, self.units), dtype)
        a0 = tf.zeros((batch_size, self.units), dtype)
        z_buf0 = tf.zeros((batch_size, self.units * self._n_substeps), dtype)
        return v0, a0, z_buf0

    def build(self, input_shape):

        if not self.sparse_weights:
            self.recurrent_weights = self.add_weight(
                shape=(self.units, self.units),
                initializer=tf.keras.initializers.Orthogonal(gain=1.),
                name='recurrent_weights',
                constraint=ZeroDiagonalConstraint())
        else:
            self._add_sparse_recurrent_weights()

        super().build(input_shape)

    def _add_sparse_recurrent_weights(self):

        self.w_rec_dense_shape = (self.units, self.units)
        n_inputs_per_neuron = int(self.units * self.rec_conn_density)
        n_sparse_values = self.units * n_inputs_per_neuron

        w_sparse_values_init = np.random.normal(0.0, 1.0/np.sqrt(n_inputs_per_neuron), (n_sparse_values))
        self.w_rec_sparse_values = self.add_weight(name="RecurrentWeightsSparseValues", 
            shape=(n_sparse_values,), trainable=True,
            dtype=self._dtype_policy.variable_dtype, 
            initializer=tf.constant_initializer(w_sparse_values_init))

        self.w_rec_sparse_idcs = tf.constant(self._create_sparse_recurrent_idcs(n_inputs_per_neuron),
            name="RecurrentWeightsSparseIdcs")

    def _create_sparse_recurrent_idcs(self, n_inputs_per_neuron, zero_diagonal=True):
        '''
        create sparse indices which select n_inputs_per_neuron inputs for each neuron
        supposed to index into a [neurons x inputs] matrix
        '''

        # every neuron gets n_inputs_per_neuron inputs
        sparse_idcs = np.zeros((self.units*n_inputs_per_neuron, 2), dtype=np.int64)
        for neuron in range(self.units):
            input_idcs = np.arange(self.units, dtype=np.int32)
            if zero_diagonal:
                np.delete(input_idcs, neuron)

            input_choices = np.random.choice(input_idcs, n_inputs_per_neuron, replace=False)
            input_choices.sort()

            neuron_sparse_idcs = np.stack((neuron*np.ones(n_inputs_per_neuron), input_choices), axis=1)
            sparse_idcs[neuron*n_inputs_per_neuron:(neuron+1)*n_inputs_per_neuron] = neuron_sparse_idcs

        return sparse_idcs

    def compute_input_current(self, inp):
        tf_shp = tf.unstack(tf.shape(inp))
        shp = inp.shape.as_list()
        for i, a in enumerate(shp):
            if a is None:
                shp[i] = tf_shp[i]
        input_current = self.input_layer(inp)
        input_current = tf.reshape(
            input_current, (shp[0], shp[1] // self._n_refractory, self._n_refractory * self.units))
        return input_current

    def call(self, inputs, state, constants=None):
        old_v = state[0]
        old_a = state[1]

        inputs = tf.reshape(inputs, (-1, self._n_refractory, self.units))
        old_z_buf = tf.reshape(state[2], (-1, self._n_substeps, self.units))

        i_in = inputs
        if not self.sparse_weights:
            i_rec = tf.einsum('bti,ij->btj', old_z_buf, self.recurrent_weights)
        else:
            w_rec_sparse = tf.SparseTensor(self.w_rec_sparse_idcs, self.w_rec_sparse_values, self.w_rec_dense_shape)
            reshaped_old_z_buf = tf.reshape(old_z_buf, (-1, self.units))
            t_i_rec = tf.transpose(tf.sparse.sparse_dense_matmul(w_rec_sparse, reshaped_old_z_buf, adjoint_b=True))
            i_rec = tf.reshape(t_i_rec, (-1, self._n_substeps, self.units))
        i_total = i_in + i_rec

        padded_i_total = tf.pad(i_total, ((0, 0), (self._n_refractory - 1, 0), (0, 0)))[..., None]
        filtered_i = tf.nn.conv2d(padded_i_total, self._decay_kernel, strides=[1, 1], padding='VALID')[..., 0]

        voltage_decay = self._decay**tf.cast(tf.range(1, self._n_refractory + 1), self._dtype_policy.compute_dtype)
        tilde_v_buf = voltage_decay[None, :, None] * old_v[:, None, :] + filtered_i

        adaptation_decay = self._decay_adaptation**tf.cast(
            tf.range(1, self._n_refractory + 1), self._dtype_policy.compute_dtype)
        tilde_a_buf = adaptation_decay[None, :, None] * old_a[:, None, :]

        v_sc = (tilde_v_buf - self.beta[:, None, :] * tilde_a_buf - self.threshold) / self.threshold
        if v_sc.dtype == tf.float32:
            z_buf = spike_function(v_sc, self._dampening_factor)
        else:
            z_buf = spike_function16(v_sc, self._dampening_factor)

        ref_mask_old = (tf.cumsum(old_z_buf, axis=1, reverse=True) - old_z_buf) > .5
        ref_mask_new = (tf.cumsum(z_buf, axis=1) - z_buf) > .5
        z_buf = tf.where(tf.logical_or(ref_mask_new, ref_mask_old), tf.zeros_like(z_buf), z_buf)

        padded_i_reset = tf.pad(z_buf * self.threshold, ((0, 0), (self._n_refractory - 1, 0), (0, 0)))[..., None]
        filtered_i_reset = tf.nn.conv2d(padded_i_reset, self._decay_kernel, strides=[1, 1], padding='VALID')[..., 0]

        padded_a_increase = tf.pad(z_buf, ((0, 0), (self._n_refractory - 1, 0), (0, 0)))[..., None]
        filtered_a_increase = tf.nn.conv2d(
            padded_a_increase, self._decay_kernel_adaptation, strides=[1, 1], padding='VALID')[..., 0]

        v_buf = tilde_v_buf - filtered_i_reset
        a_buf = tilde_a_buf + filtered_a_increase

        flat_z_buf = tf.reshape(z_buf, (-1, self._n_refractory * self.units))
        new_state = (v_buf[:, -1], a_buf[:, -1], flat_z_buf)
        output = (z_buf, v_buf, a_buf, v_sc)

        return output, new_state


def main():
    n_inputs = 100
    n_units = 128
    n_time = 1000
    n_batch = 1
    cell = FastALIFCell(n_units, n_in=n_inputs, sparse_weights=False)
    initial_state = cell.zero_state(n_batch)
    random_inputs = tf.random.uniform((n_batch, n_time, n_inputs), 0., .1)

    input_currents = cell.compute_input_current(random_inputs)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
    out = rnn(input_currents, initial_state=initial_state)
    z, v, aux = out[0]
    z = tf.reshape(z, (n_batch, n_time, n_units))
    v = tf.reshape(z, (n_batch, n_time, n_units))
    updated_state = out[1:]
    print('Spikes shape:', z.shape)
    print('Voltage shape:', v.shape)


if __name__ == '__main__':
    main()

