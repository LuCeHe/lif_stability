import numpy as np
import tensorflow as tf

iem = lambda v, m, q: - 1 / (2 * m) * tf.math.exp(-2 * m * v) + 1 / (2 * m)
iqtail2 = lambda v, m, q: -(q + 2 * v - 1) / (2 * (2 * q - 1)) / (1 + 2 / (q - 1) * v) ** (2 * q) + (q - 1) / (
            4 * q - 2)

area_m = lambda v_p, v_m, m, q, func: tf.sign(v_p) * func(tf.math.abs(v_p), m, q) - tf.sign(v_m) * func(
    tf.math.abs(v_m), m, q)

E_sm = lambda s, m, dampening, thr, y_max, y_min: dampening ** m / (s * (y_max - y_min)) * \
                                                  area_m(s * (y_max - thr), s * (y_min - thr), m, 1, iem)

E_t2 = lambda s, q, dampening, thr, y_max, y_min: dampening ** 2 / (s * (y_max - y_min)) * \
                                               area_m(s * (y_max - thr), s * (y_min - thr), 2, q, iqtail2)


def equation_IV_sub_reset(s, dampening, decay, thr, w_rec, w_in, s_in, d_in):
    n_rec = w_rec.shape[0]
    n_in = tf.cast(tf.shape(w_in)[0], tf.float32)
    sum_rec = tf.reduce_sum(tf.nn.relu(w_rec), axis=0)
    sum_in = tf.reduce_sum(tf.nn.relu(w_in), axis=0)

    _sum_rec = -tf.reduce_sum(tf.nn.relu(-w_rec), axis=0)
    _sum_in = -tf.reduce_sum(tf.nn.relu(-w_in), axis=0)

    y_max = 1 / (1 - decay) * (sum_rec + sum_in)
    y_min = 1 / (1 - decay) * (_sum_rec + _sum_in - thr)

    if not isinstance(s_in, np.ndarray) and s_in == 0:
        eq = -1 + tf.reduce_mean(decay ** 2) + ((n_rec - 1) * tf.reduce_mean(w_rec ** 2) + tf.reduce_mean(thr ** 2)) \
             * E_sm(s, 2, dampening, thr, y_max, y_min)

    else:
        eq = -1 + tf.reduce_mean(decay ** 2) + ((n_rec - 1) * tf.reduce_mean(w_rec ** 2) + tf.reduce_mean(thr ** 2)) \
             * E_sm(s, 2, dampening, thr, y_max, y_min) \
             + n_in * tf.reduce_mean(w_in ** 2) * E_sm(s_in, 1, d_in, thr, y_max, y_min)

    return eq


def equation_IV_mult_reset1(s, dampening, decay, thr, w_rec, w_in, s_in, d_in):
    n_rec = w_rec.shape[0]
    n_in = tf.cast(tf.shape(w_in)[0], tf.float32)
    sum_rec = tf.reduce_sum(tf.nn.relu(w_rec), axis=0)
    sum_in = tf.reduce_sum(tf.nn.relu(w_in), axis=0)

    _sum_rec = -tf.reduce_sum(tf.nn.relu(-w_rec), axis=0)
    _sum_in = -tf.reduce_sum(tf.nn.relu(-w_in), axis=0)

    y_max = 1 / (1 - decay) * (sum_rec + sum_in)
    y_min = 1 / (1 - decay) * (_sum_rec + _sum_in)

    if not isinstance(s_in, np.ndarray) and s_in == 0:
        eq = -2 + tf.reduce_mean(decay ** 2) + ((n_rec - 1) * tf.reduce_mean(w_rec ** 2)) \
             * E_sm(s, 2, dampening, thr, y_max, y_min)

    else:
        eq = -2 + tf.reduce_mean(decay ** 2) + ((n_rec - 1) * tf.reduce_mean(w_rec ** 2)) \
             * E_sm(s, 2, dampening, thr, y_max, y_min) \
             + n_in * tf.reduce_mean(w_in ** 2) * E_sm(s_in, 1, d_in, thr, y_max, y_min)

    return eq


def equation_IV_mult_reset2(s, dampening, decay, thr, w_rec, w_in, s_in, d_in):
    n_rec = w_rec.shape[0]
    n_in = tf.cast(tf.shape(w_in)[0], tf.float32)
    # sum_rec = tf.reduce_max(w_rec)
    # sum_in = tf.reduce_max(w_in)
    # _sum_rec = tf.reduce_min(w_rec)
    # _sum_in = tf.reduce_min(w_in)

    sum_rec = tf.reduce_sum(tf.nn.relu(w_rec), axis=0)
    sum_in = tf.reduce_sum(tf.nn.relu(w_in), axis=0)
    _sum_rec = -tf.reduce_sum(tf.nn.relu(-w_rec), axis=0)
    _sum_in = -tf.reduce_sum(tf.nn.relu(-w_in), axis=0)

    y_max = 1 / (1 - decay) * (sum_rec + sum_in)
    y_min = 1 / (1 - decay) * (_sum_rec + _sum_in)

    if not isinstance(s_in, np.ndarray) and s_in == 0:
        eq = -1 + tf.reduce_mean(decay ** 2) / 2 + ((n_rec - 1) * tf.reduce_mean(w_rec ** 2)) \
             * E_sm(s, 2, dampening, thr, y_max, y_min)

    else:
        eq = -1 + tf.reduce_mean(decay ** 2) / 2 + ((n_rec - 1) * tf.reduce_mean(w_rec ** 2)) \
             * E_sm(s, 2, dampening, thr, y_max, y_min) \
             + n_in * tf.reduce_mean(w_in ** 2) * E_sm(s_in, 1, d_in, thr, y_max, y_min)

    return eq



def equation_IV_mult_reset2_tail(s, dampening, decay, thr, w_rec, w_in, s_in, d_in):
    n_rec = w_rec.shape[0]
    n_in = tf.cast(tf.shape(w_in)[0], tf.float32)
    sum_rec = tf.reduce_max(w_rec)
    sum_in = tf.reduce_max(w_in)
    _sum_rec = tf.reduce_min(w_rec)
    _sum_in = tf.reduce_min(w_in)

    y_max = 1 / (1 - decay) * (sum_rec + sum_in)
    y_min = 1 / (1 - decay) * (_sum_rec + _sum_in)

    if not isinstance(s_in, np.ndarray) and s_in == 0:
        eq = -1 + tf.reduce_mean(decay ** 2) / 2 + ((n_rec - 1) * tf.reduce_mean(w_rec ** 2)) \
             * E_t2(1, s, dampening, thr, y_max, y_min)

    else:
        eq = -1 + tf.reduce_mean(decay ** 2) / 2 + ((n_rec - 1) * tf.reduce_mean(w_rec ** 2)) \
             * E_t2(1, s, dampening, thr, y_max, y_min) \
             + n_in * tf.reduce_mean(w_in ** 2) * E_sm(s_in, 1, d_in, thr, y_max, y_min)

    return eq

class OptimizeVariance(tf.keras.layers.Layer):

    def __init__(self, w_rec, w_in, decay, thr, dampening, s_in, d_in, equation_0, minmax =[.2, 2.],**kwargs):
        super().__init__(**kwargs)
        self.w_rec = tf.cast(w_rec, tf.float32)
        self.w_in = tf.cast(w_in, tf.float32)
        self.decay = tf.cast(decay, tf.float32)
        self.thr = tf.cast(thr, tf.float32)
        self.dampening = tf.cast(dampening, tf.float32)
        self.s_in = tf.cast(s_in, tf.float32).numpy()
        self.d_in = tf.cast(d_in, tf.float32)
        self.equation_0 = equation_0
        self.minmax = minmax

        self.n_rec = tf.cast(self.w_rec.shape[0], tf.float32)
        self.n_in = tf.cast(tf.shape(w_in)[0], tf.float32)

    def build(self, input_shape):
        self.s = self.add_weight(
            name='sharpness', shape=(int(self.n_rec.numpy()),),
            # initializer=tf.keras.initializers.RandomUniform(minval=.2, maxval=2.),# for s
            # for tail optimization, to avoid nans when q<1.0:
            # initializer=tf.keras.initializers.RandomUniform(minval=1.01, maxval=10.),
            initializer=tf.keras.initializers.RandomUniform(minval=self.minmax[0], maxval=self.minmax[1]),
            trainable=True
        )
        self.built = True
        print('lets see')

        print(np.mean(self.s), np.std(self.s))

    def call(self, inputs, **kwargs):
        w_rec = self.w_rec + tf.random.normal(tf.shape(self.w_rec), mean=0.0, stddev=tf.math.reduce_std(self.w_rec))
        w_in = self.w_in + tf.random.normal(tf.shape(self.w_in), mean=0.0, stddev=tf.math.reduce_std(self.w_rec))
        decay = self.decay
        dampening = self.dampening
        thr = self.thr  # + 2.6*tf.random.normal(tf.shape(self.thr), mean=0.0, stddev=tf.math.reduce_std(self.thr))

        loss = self.equation_0(tf.abs(self.s), dampening, decay, thr, w_rec, w_in, self.s_in, self.d_in) ** 2
        self.add_loss(loss)
        self.add_metric(loss, name='sharpness_cost', aggregation='mean')

        # print(self.s)

        return inputs

    def get_config(self):
        config = {
            'w_rec': self.w_rec, 'w_in': self.w_in, 'decay': self.decay, 'thr': self.thr, 'dampening': self.dampening,
            's_in': self.s_in, 'd_in': self.d_in
        }
        return dict(list(super().get_config().items()) + list(config.items()))


def optimize_sharpness_a(w_rec, w_in, decay, thr, dampening, s_in, d_in, equation_0, n_attempts=100000,
                         min_sharpness=.3, max_sharpness=1.6):
    n_rec = w_rec.shape[0]
    sharpnesses_seeds = np.random.uniform(min_sharpness, max_sharpness, size=(n_attempts, n_rec))

    cost = equation_0(sharpnesses_seeds, dampening, decay, thr, w_rec, w_in, s_in, d_in) ** 2
    idx = tf.math.argmin(cost, axis=0)

    # print(sharpnesses_seeds)
    optimized_sharpness = np.take(sharpnesses_seeds, idx)
    return optimized_sharpness


def optimize_sharpness_b(w_rec, w_in, decay, thr, dampening, s_in, d_in, equation_0, n_attempts=100000,
                         min_sharpness=.3,
                         max_sharpness=1.6):
    s = optimize_sharpness_a(w_rec, w_in, decay, thr, dampening, s_in, d_in, equation_0)
    return tf.reduce_mean(s)


def optimize_sharpness_c(w_rec, w_in, decay, thr, dampening, s_in, d_in, equation_0, epochs=500):
    sharpness_layer = OptimizeVariance(w_rec, w_in, decay, thr, dampening, s_in, d_in, equation_0)
    input_layer = tf.keras.layers.Input((1,))
    output = sharpness_layer(input_layer)
    model = tf.keras.models.Model(input_layer, output)

    model.compile('Adam', None)
    t = tf.random.uniform((1, 1))
    model.fit(t, t, epochs=epochs, verbose=1)
    return abs(sharpness_layer.s.numpy())


def optimize_sharpness_d(w_rec, w_in, decay, thr, dampening, s_in, d_in, equation_0, epochs=1000):
    s = optimize_sharpness_c(w_rec, w_in, decay, thr, dampening, s_in, d_in, equation_0, epochs=epochs)
    return np.mean(s)


def optimize_dampening_a(w_rec, thr, decay, w_in, dampening_in):
    n_rec = w_rec.shape[0]
    n_in = tf.cast(tf.shape(w_in)[0], tf.float32)
    optimized_dampening = tf.reduce_min(w_rec) / \
                          (tf.reduce_max(w_rec) * ((n_rec - 1) * tf.reduce_min(w_rec) - thr)) \
                          * (1 - decay - n_in * dampening_in * tf.reduce_max(w_in))
    return optimized_dampening


def optimize_dampening_b(w_rec, thr, decay, w_in, dampening_in):
    optimized_dampening = optimize_dampening_a(w_rec, thr, decay, w_in, dampening_in)
    return tf.reduce_mean(optimized_dampening)


def optimize_dampening_c(w_rec, thr, decay, w_in, dampening_in):
    n_in = tf.cast(tf.shape(w_in)[0], tf.float32)
    sum_rec = tf.reduce_sum(tf.nn.relu(w_rec), axis=0)
    sum_in = tf.reduce_sum(tf.nn.relu(w_rec), axis=0)
    _sum_rec = -tf.reduce_sum(tf.nn.relu(-w_rec), axis=0)
    optimized_dampening = _sum_rec / (sum_rec * (_sum_rec - thr)) \
                          * (1 - decay - n_in * dampening_in * sum_in)
    return optimized_dampening


def optimize_dampening_d(w_rec, thr, decay, w_in, dampening_in):
    optimized_dampening = optimize_dampening_c(w_rec, thr, decay, w_in, dampening_in)
    return tf.reduce_mean(optimized_dampening)


def optimize_dampening(w_rec, thr, decay, w_in, dampening_in, od_type='a'):
    if od_type == 'a':
        od = optimize_dampening_a(w_rec, thr, decay, w_in, dampening_in)
    elif od_type == 'b':
        od = optimize_dampening_b(w_rec, thr, decay, w_in, dampening_in)
    elif od_type == 'c':
        od = optimize_dampening_c(w_rec, thr, decay, w_in, dampening_in)
    elif od_type == 'd':
        od = optimize_dampening_d(w_rec, thr, decay, w_in, dampening_in)
    return od


def optimize_sharpness(w_rec, w_in, decay, thr, dampening, dampening_in, sharpness_in, os_type='a', config=''):
    if 'multreset1' in config:
        equation_0 = equation_IV_mult_reset1
    elif 'multreset2' in config:
        equation_0 = equation_IV_mult_reset2
    else:
        equation_0 = equation_IV_sub_reset

    ops = 1
    if os_type == 'a':
        ops = optimize_sharpness_a(w_rec, w_in, decay, thr, dampening, sharpness_in, dampening_in, equation_0)
    elif os_type == 'b':
        ops = optimize_sharpness_b(w_rec, w_in, decay, thr, dampening, sharpness_in, dampening_in, equation_0)
    elif os_type == 'c':
        ops = optimize_sharpness_c(w_rec, w_in, decay, thr, dampening, sharpness_in, dampening_in, equation_0)
    elif os_type == 'd':
        ops = optimize_sharpness_d(w_rec, w_in, decay, thr, dampening, sharpness_in, dampening_in, equation_0)
    return tf.cast(ops, tf.float32)


def optimize_tail(w_rec, w_in, decay, thr, dampening, s_in, d_in, epochs=1000):
    equation_0 = equation_IV_mult_reset2_tail
    sharpness_layer = OptimizeVariance(w_rec, w_in, decay, thr, dampening, s_in, d_in, equation_0, minmax=[1.01, 3.0])
    input_layer = tf.keras.layers.Input((1,))
    output = sharpness_layer(input_layer)
    model = tf.keras.models.Model(input_layer, output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01,    name="Adam")
    model.compile(optimizer, None)
    t = tf.random.uniform((1, 1))
    history = model.fit(t, t, epochs=epochs, verbose=1)
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.show()
    # print(history)
    return abs(sharpness_layer.s.numpy())


def test_1():
    n_in, n_rec = 200, 300
    w_in = np.random.uniform(-1, 1, size=(n_in, n_rec))
    w_rec = np.random.uniform(-1, 1, size=(n_rec, n_rec))
    decay = np.random.uniform(size=(n_rec))
    thr = np.random.uniform(size=(n_rec))

    optimized_dampening = optimize_dampening(w_rec, thr, decay)

    dampening = optimized_dampening  # .1
    optimized_sharpness = optimize_sharpness(w_rec, w_in, decay, thr, dampening)
    # print(optimized_dampening)
    print(np.mean(dampening))
    print(np.mean(optimized_sharpness))

    sum_rec = tf.reduce_sum(tf.nn.relu(w_rec), axis=0)
    sum_in = tf.reduce_sum(tf.nn.relu(w_in), axis=0)

    _sum_rec = -tf.reduce_sum(tf.nn.relu(-w_rec), axis=0)
    _sum_in = -tf.reduce_sum(tf.nn.relu(-w_in), axis=0)

    y_max = 1 / (1 - decay) * (sum_rec + sum_in)
    y_min = 1 / (1 - decay) * (_sum_rec + _sum_in - thr)

    # print('sgd optimization:    ', np.mean(equation_0(optimized_sharpness, dampening, y_max, y_min) ** 2))

    sgs = optimize_sharpness_c(w_rec, w_in, decay, thr, dampening)
    print('\n\n\n')
    print('s=1:                 ', np.mean(equation_0(1, dampening, decay, n_rec, thr, w_rec, y_max, y_min) ** 2))
    print('s=2:                 ', np.mean(equation_0(2, dampening, decay, n_rec, thr, w_rec, y_max, y_min) ** 2))
    print('random optimization: ',
          np.mean(equation_0(optimized_sharpness, dampening, decay, n_rec, thr, w_rec, y_max, y_min) ** 2))
    print('sgd optimization:    ',
          np.mean(equation_0(sgs, dampening, decay, n_rec, thr, w_rec, y_max, y_min) ** 2))
    # print(sharpness_layer.s.numpy())


def test_2():
    pass


if __name__ == '__main__':
    test_1()
