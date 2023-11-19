import sys

from sg_design_lif.neural_models.custom_lstm import customLSTM, spikingLSTM, gravesLSTM
from sg_design_lif.neural_models.fast_alif import FastALIFCell
from sg_design_lif.neural_models.lsnn import *
from sg_design_lif.neural_models.reservoir import *
from sg_design_lif.neural_models.wta import WTA
from sg_design_lif.neural_models.izhikevich import Izhikevich
from sg_design_lif.neural_models.spiking_attention import SpikingAttention
from sg_design_lif.neural_models.readout import Readout

thismodule = sys.modules[__name__]

def net(net_name='LSNN'):
    net_model = getattr(thismodule, net_name)
    return net_model
