import sys

from lif_stability.neural_models.custom_lstm import customLSTM, spikingLSTM, gravesLSTM
from lif_stability.neural_models.fast_alif import FastALIFCell
from lif_stability.neural_models.lsnn import *
from lif_stability.neural_models.reservoir import *
from lif_stability.neural_models.wta import WTA
from lif_stability.neural_models.izhikevich import Izhikevich
from lif_stability.neural_models.spiking_attention import SpikingAttention
from lif_stability.neural_models.readout import Readout

thismodule = sys.modules[__name__]

def net(net_name='LSNN'):
    net_model = getattr(thismodule, net_name)
    return net_model
