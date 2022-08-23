import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from GenericTools.keras_tools.esoteric_layers import SurrogatedStep


class customLSTM(tf.keras.layers.Layer):

    def get_config(self):
        return self.init_args

    def __init__(self, num_neurons=None, activation_gates='hard_sigmoid', activation_c='tanh', activation_h='tanh',
                 initializer='glorot_uniform', string_config='', **kwargs):
        self.init_args = dict(num_neurons=num_neurons, activation_gates=activation_gates, activation_c=activation_c,
                              activation_h=activation_h, string_config=string_config)
        super().__init__(**kwargs)
        self.__dict__.update(self.init_args)

        self.activation_gates = activation_gates
        self.activation_c = activation_c
        self.activation_h = activation_h

        self.state_size = (num_neurons, num_neurons)

        self.linear_f_input = Dense(num_neurons, kernel_initializer=initializer)
        self.linear_f_h = Dense(num_neurons, use_bias=False, kernel_initializer=initializer)

        self.linear_i_input = Dense(num_neurons, kernel_initializer=initializer)
        self.linear_i_h = Dense(num_neurons, use_bias=False, kernel_initializer=initializer)

        self.linear_o_input = Dense(num_neurons, kernel_initializer=initializer)
        self.linear_o_h = Dense(num_neurons, use_bias=False, kernel_initializer=initializer)

        self.linear_c_input = Dense(num_neurons, kernel_initializer=initializer)
        self.linear_c_h = Dense(num_neurons, use_bias=False, kernel_initializer=initializer)

    def call(self, inputs, states, **kwargs):
        # if not training is None:
        #     tf.keras.backend.set_learning_phase(training)

        old_c, old_h = states

        f = Activation(self.activation_gates)(self.linear_f_input(inputs) + self.linear_f_h(old_h))
        i = Activation(self.activation_gates)(self.linear_i_input(inputs) + self.linear_i_h(old_h))
        o = Activation(self.activation_gates)(self.linear_o_input(inputs) + self.linear_o_h(old_h))
        c_tilde = Activation(self.activation_c)(self.linear_c_input(inputs) + self.linear_c_h(old_h))

        c = f * old_c + i * c_tilde
        h = o * Activation(self.activation_h)(c)

        output = h
        new_state = (c, h)
        return output, new_state


class spikingLSTM(customLSTM):

    def __init__(self, *args, **kwargs):
        print(kwargs['string_config'])
        activation_spikes_gates = lambda x: SurrogatedStep(string_config=self.string_config)(x)
        activation_spikes_c = lambda x: 2 * SurrogatedStep(string_config=self.string_config)(x) - 1
        activation_spikes_h = lambda x: 2 * SurrogatedStep(string_config=self.string_config)(x) - 1
        kwargs['activation_gates'] = activation_spikes_gates
        kwargs['activation_c'] = activation_spikes_c
        kwargs['activation_h'] = activation_spikes_h

        super().__init__(*args, **kwargs)


class gravesLSTM(tf.keras.layers.Layer):
    """

    LSTM version used in the paper.

    'Generating Sequences With Recurrent Neural Networks'

    https://arxiv.org/abs/1308.0850

    """

    def get_config(self):
        return self.init_args

    def __init__(self, num_neurons=None, activation_gates='sigmoid', activation_c='tanh', activation_h='tanh',
                 initializer='glorot_uniform', string_config='', **kwargs):
        self.init_args = dict(num_neurons=num_neurons, activation_gates=activation_gates, activation_c=activation_c,
                              activation_h=activation_h, string_config=string_config)
        super().__init__(**kwargs)
        self.__dict__.update(self.init_args)

        self.activation_gates = activation_gates
        self.activation_c = activation_c
        self.activation_h = activation_h

        self.state_size = (num_neurons, num_neurons)

        self.linear_f_input = Dense(num_neurons, kernel_initializer=initializer)
        self.linear_f_h = Dense(num_neurons, use_bias=False, kernel_initializer=initializer)
        self.linear_f_c = Dense(num_neurons, use_bias=False, kernel_initializer=initializer)

        self.linear_i_input = Dense(num_neurons, kernel_initializer=initializer)
        self.linear_i_h = Dense(num_neurons, use_bias=False, kernel_initializer=initializer)
        self.linear_i_c = Dense(num_neurons, use_bias=False, kernel_initializer=initializer)

        self.linear_o_input = Dense(num_neurons, kernel_initializer=initializer)
        self.linear_o_h = Dense(num_neurons, use_bias=False, kernel_initializer=initializer)
        self.linear_o_c = Dense(num_neurons, use_bias=False, kernel_initializer=initializer)

        self.linear_c_input = Dense(num_neurons, kernel_initializer=initializer)
        self.linear_c_h = Dense(num_neurons, use_bias=False, kernel_initializer=initializer)

    def build(self, input_shape):
        n_input = input_shape[-1]

        initializer = tf.keras.initializers.RandomNormal(stddev=1. / tf.sqrt(float(n_input)))

        self.wcf = self.add_weight(shape=(self.num_neurons,), initializer=initializer, name='wcf')
        self.wci = self.add_weight(shape=(self.num_neurons,), initializer=initializer, name='wci')
        self.wco = self.add_weight(shape=(self.num_neurons,), initializer=initializer, name='wco')

        self.built = True

    def call(self, inputs, states, training=None):
        # if not training is None:
        #     tf.keras.backend.set_learning_phase(training)

        old_c, old_h = states

        f = Activation(self.activation_gates)(
            self.linear_f_input(inputs) + self.wcf * old_c + self.linear_f_h(old_h)
        )
        i = Activation(self.activation_gates)(
            self.linear_i_input(inputs) + self.wci * old_c + self.linear_i_h(old_h)
        )
        c_tilde = Activation(self.activation_c)(
            self.linear_c_input(inputs) + self.linear_c_h(old_h)
        )
        c = f * old_c + i * c_tilde

        o = Activation(self.activation_gates)(
            self.linear_o_input(inputs) + self.wco * c + self.linear_o_h(old_h)
        )

        h = o * Activation(self.activation_h)(c)

        output = h
        new_state = (c, h)
        return output, new_state
