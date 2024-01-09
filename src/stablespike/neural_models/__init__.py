import sys

from stablespike.neural_models.custom_lstm import customLSTM, spikingLSTM, gravesLSTM
from stablespike.neural_models.fast_alif import FastALIFCell
from stablespike.neural_models.lsnn import *
from stablespike.neural_models.reservoir import *
from stablespike.neural_models.wta import WTA
from stablespike.neural_models.izhikevich import Izhikevich
from stablespike.neural_models.spiking_attention import SpikingAttention
from stablespike.neural_models.readout import Readout

thismodule = sys.modules[__name__]

def net(net_name='LSNN'):
    net_model = getattr(thismodule, net_name)
    return net_model
