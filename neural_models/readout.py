import tensorflow.keras as k
import numpy as np


class Readout(k.layers.Layer):
    def __init__(self, num_neurons, tau=20):
        super().__init__()
        self.state_size = num_neurons
        self.readout = k.layers.Dense(num_neurons)
        self.decay = np.exp(-1 / tau)

    def call(self, inputs, state):
        state = self.decay * state[0] + (1 - self.decay) * self.readout(inputs)
        return state, (state,)