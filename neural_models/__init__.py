import sys

from GenericTools.keras_tools.esoteric_layers.rate_voltage_reg import RateVoltageRegularization
from stochastic_spiking.neural_models.custom_lstm import customLSTM, spikingLSTM, gravesLSTM
from stochastic_spiking.neural_models.fast_alif import FastALIFCell
from stochastic_spiking.neural_models.lsnn import *
from stochastic_spiking.neural_models.reservoir import *
from stochastic_spiking.neural_models.wta import WTA
from stochastic_spiking.neural_models.izhikevich import Izhikevich
from stochastic_spiking.neural_models.spiking_attention import SpikingAttention
from stochastic_spiking.neural_models.readout import Readout

# from stochastic_spiking.neural_models.spiking_performer import spikingPerformer
# from stochastic_spiking.neural_models.modeling_tf_gpt2 import smallGPT2

thismodule = sys.modules[__name__]

def net(net_name='LSNN'):
    net_model = getattr(thismodule, net_name)
    return net_model
