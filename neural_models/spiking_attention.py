import tensorflow as tf
from stochastic_spiking.neural_models.lsnn import baseLSNN as LSNN
from stochastic_spiking.neural_models.wta import WTA


# TODO
# - [DONE] create basic attention cell
# - make full Transformer architecture
# - KQ is elementwise multiplication, make a variant where KQ only concatenate with their pair,
# or similar, maybe through diagonal-like matrices



class SpikingAttention(tf.keras.layers.Layer):
    """
    Spiking Attention -> one step towards a Spiking Transformer
    """

    def __init__(self, units, output_units = None):
        super().__init__()
        if output_units == None: output_units = units
        self.lsnns = {k: LSNN(units) for k in ['k', 'q', 'v']}
        self.lsnns.update({'attention': LSNN(output_units)})

        self.lsnns = {k: tf.keras.layers.RNN(v, return_sequences=True)
                      for k, v in self.lsnns.items()}
        self.wta = tf.keras.layers.RNN(WTA(units), return_sequences=True)

    def call(self, inputs):
        #z, new_v, thr, v_sc
        k = self.lsnns['k'](inputs)
        q = self.lsnns['q'](inputs)
        kq = tf.concat([k[0], q[0]], axis=-1)

        #z_e, z_i, v_sc_e, v_sc_i
        pseudo_softmax = self.wta(kq)
        v = self.lsnns['v'](inputs)
        vs = tf.concat([v[0], pseudo_softmax[0]], axis=-1)
        attention = self.lsnns['attention'](vs)

        attention_spikes = attention[0]
        all_spikes = tf.concat([k[0], q[0], v[0], pseudo_softmax[0], pseudo_softmax[1], attention[0]], axis=-1)
        all_thr = tf.concat([k[2], q[2], v[2], attention[2]], axis=-1)
        all_v_sc = tf.concat([k[3], q[3], v[3], pseudo_softmax[2], pseudo_softmax[3], attention[3]], axis=-1)
        output = (attention_spikes, all_spikes, all_thr, all_v_sc)
        return output