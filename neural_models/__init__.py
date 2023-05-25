import sys

from pyaromatics.keras_tools.esoteric_layers.rate_voltage_reg import RateVoltageRegularization
from sg_design_lif.neural_models.custom_lstm import customLSTM, spikingLSTM, gravesLSTM
from sg_design_lif.neural_models.fast_alif import FastALIFCell
from sg_design_lif.neural_models.lsnn import *
from sg_design_lif.neural_models.reservoir import *
from sg_design_lif.neural_models.wta import WTA
from sg_design_lif.neural_models.izhikevich import Izhikevich
from sg_design_lif.neural_models.spiking_attention import SpikingAttention
from sg_design_lif.neural_models.readout import Readout

# from sg_design_lif.neural_models.spiking_performer import spikingPerformer
# from sg_design_lif.neural_models.modeling_tf_gpt2 import smallGPT2

thismodule = sys.modules[__name__]

def net(net_name='LSNN'):
    net_model = getattr(thismodule, net_name)
    return net_model
