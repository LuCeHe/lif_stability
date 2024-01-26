import sys

from stablespike.neural_models.custom_lstm import customLSTM, spikingLSTM, gravesLSTM
from stablespike.neural_models.lsnn import *
from stablespike.neural_models.readout import Readout

thismodule = sys.modules[__name__]

def net(net_name='LSNN'):
    net_model = getattr(thismodule, net_name)
    return net_model
