import tensorflow as tf
import numpy as np

from lif_stability.neural_models import baseLSNN


class Reservoir(baseLSNN):

    def build(self, input_shape):
        n_input = input_shape[-1]
        self.input_weights = np.random.randn(n_input, self.in_neurons) / np.sqrt(n_input)

        self.mask = tf.ones((self.num_neurons, self.num_neurons)) - tf.eye(self.num_neurons)
        self.recurrent_weights = np.random.randn(self.num_neurons, self.num_neurons) / np.sqrt(self.num_neurons)

        self.rw = self.mask * self.recurrent_weights
        self.inh_exc = np.ones(self.num_neurons)

        self._beta = np.concatenate([np.zeros(self.n_regular), np.ones(self.num_neurons - self.n_regular) * self.beta])

        self.built = True




np_softplus = lambda x: np.log(1+np.exp(-np.abs(x))) + np.maximum(x,0)




class DaleReservoir(Reservoir):

    def recurrent_transform(self):
        return tf.math.softplus(self.mask * self.recurrent_weights)


    def build(self, input_shape):
        n_input = input_shape[-1]
        self.input_weights = np.random.randn(n_input, self.in_neurons) / np.sqrt(n_input)

        self.mask = tf.ones((self.num_neurons, self.num_neurons)) - tf.eye(self.num_neurons)
        self.recurrent_weights = np.random.randn(self.num_neurons, self.num_neurons) / np.sqrt(self.num_neurons)

        self.rw = tf.math.softplus(self.mask * self.recurrent_weights)
        self.inh_exc = np.random.choice([-1, 1], self.num_neurons)

        self._beta = np.concatenate([np.zeros(self.n_regular), np.ones(self.num_neurons - self.n_regular) * self.beta])

        self.built = True