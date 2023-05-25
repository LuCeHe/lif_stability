import sys
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops, gen_linalg_ops, math_ops

from pyaromatics.keras_tools.esoteric_layers.positional_embedding import EmbeddingLayer, PositionEmbeddingLayer
from pyaromatics.keras_tools.esoteric_layers.stochastic_depth import StochasticDepth
from sg_design_lif.neural_models import SpikeFunction

"""
sources:
https://github.com/akanyaani/gpt-2-tensorflow2.0/blob/master/gpt2_model.py
https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_tf_gpt2.py
"""


# tfgp_t2lm_head_model/transformer/h_._0/ln_1/gamma:0
#      (768,)
# tfgp_t2lm_head_model/transformer/h_._0/ln_1/beta:0
#      (768,)
# tfgp_t2lm_head_model/transformer/h_._0/attn/c_attn/weight:0
#      (768, 2304)
# tfgp_t2lm_head_model/transformer/h_._0/attn/c_attn/bias:0
#      (1, 2304)
# tfgp_t2lm_head_model/transformer/h_._0/attn/c_proj/weight:0
#      (768, 768)
# tfgp_t2lm_head_model/transformer/h_._0/attn/c_proj/bias:0
#      (1, 768)
# tfgp_t2lm_head_model/transformer/h_._0/ln_2/gamma:0
#      (768,)
# tfgp_t2lm_head_model/transformer/h_._0/ln_2/beta:0
#      (768,)
# tfgp_t2lm_head_model/transformer/h_._0/mlp/c_fc/weight:0
#      (768, 3072)
# tfgp_t2lm_head_model/transformer/h_._0/mlp/c_fc/bias:0
#      (1, 3072)
# tfgp_t2lm_head_model/transformer/h_._0/mlp/c_proj/weight:0
#      (3072, 768)
# tfgp_t2lm_head_model/transformer/h_._0/mlp/c_proj/bias:0


def gelu(features, approximate=False, name=None):
    # from https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/ops/nn_ops.py#L3505-L3548
    with ops.name_scope(name, "Gelu", [features]):
        features = ops.convert_to_tensor(features, name="features")
        if approximate:
            coeff = math_ops.cast(0.044715, features.dtype)
            return 0.5 * features * (
                    1.0 + math_ops.tanh(0.7978845608028654 *
                                        (features + coeff * math_ops.pow(features, 3))))
        else:
            return 0.5 * features * (1.0 + math_ops.erf(
                features / math_ops.cast(1.4142135623730951, features.dtype)))


def test_linearGPT2cell():
    import numpy as np

    d = 6
    cell = linearGPT2cell(d=6, n_head=3)
    rnn = RNN(cell, return_sequences=False, stateful=True)

    np_input = np.random.rand(3, 5, d)
    output = rnn(np_input)

    print(output.shape)


def random_orthogonal(shape):
    num_rows = 1
    for dim in shape[:-1]:
        num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (tf.math.maximum(num_cols, num_rows), tf.math.minimum(num_cols, num_rows))

    # Generate a random matrix
    a = tf.random.normal(flat_shape)
    # Compute the qr factorization
    q, r = gen_linalg_ops.qr(a, full_matrices=False)

    # Make Q uniform
    d = array_ops.diag_part(r)

    q *= math_ops.sign(d)
    if num_rows < num_cols:
        q = array_ops.matrix_transpose(q)
    return tf.reshape(q, shape)


def true_phi(n_head, d, b=1):
    # w = tf.random.normal((b, n_head, 1, d // n_head))
    w = random_orthogonal(shape=(b, n_head, 1, d // n_head))

    def phi(x):
        return tf.math.softplus(w * x - tf.square(x) / 2)

    return phi


def new_phi(n_head, d, b=1, r=None, kernel_type='sigmoid'):
    if r is None:
        r = d // n_head * 2

    k, norm = kernel_type.split('_')

    def phi(x):
        # w = random_orthogonal(shape=(d, r))
        w = random_orthogonal(shape=(b, n_head, 1, d // n_head, r))
        wx = tf.matmul(x[..., None], w, transpose_a=True)
        y = wx if not 'norm' in norm else wx - tf.square(tf.norm(x)) / 2
        y = x[..., None, :] if 'northogonal' in norm else y
        p = getattr(tf.math, k)(y)  # - tf.square(tf.norm(x)) / 2) softplus
        p = tf.squeeze(p, axis=3)
        return p

    return phi


class linearGPT2cell(tf.keras.layers.Layer):

    def get_config(self):
        return {
            'd': self.d,
            'num_heads': self.num_heads,
            'kernel_type': self.kernel_type
        }

    def __init__(self, d, n_head, kernel_type, **kwargs):
        self.d = d
        self.n_head = n_head
        self.r = d // self.n_head // 2 if not 'northogonal' in kernel_type else d // self.n_head
        # 2 * d // self.n_head
        self.state_size = (d * self.r, self.r * self.n_head)
        self.kernel_type = kernel_type

        self.ln_1 = LayerNormalization(name='ln_1')

        self.c_attn = Dense(3 * d, name='attn/c_attn')
        self.attn_c_proj = Dense(d, name='attn/c_proj')

        self.ln_2 = LayerNormalization(name='ln_2')
        self.c_fc = Dense(4 * d, name='mlp/c_fc')

        self.gelu = gelu if not 'nogelu' in kernel_type else lambda x: x
        self.c_proj = Dense(d, name='mlp/c_proj')
        self.d_1 = Dropout(0.3)
        self.d_2 = Dropout(0.3)

        kwargs['name'] = ''
        super().__init__(**kwargs)

    def split_heads(self, x):
        return tf.reshape(x, (-1, self.n_head, 1, self.d // self.n_head))

    def merge_heads(self, x):
        return tf.reshape(x, (-1, self.d))

    def multi_head_attention(self, x, s, z):
        # c_attn:     (768, 2304) + (1, 2304)
        # c_proj:     (768, 768) + (1, 768)
        # phi_1 = true_phi(self.n_head, self.d, b=self.b)
        phi_1 = new_phi(self.n_head, self.d, r=self.r, kernel_type=self.kernel_type)
        phi_2 = new_phi(self.n_head, self.d, r=self.r, kernel_type=self.kernel_type)
        phi_3 = new_phi(self.n_head, self.d, r=self.r, kernel_type=self.kernel_type)
        phi_4 = new_phi(self.n_head, self.d, r=self.r, kernel_type=self.kernel_type)

        x = self.c_attn(x)
        query, key, value = tf.split(x, 3, axis=1)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        s = tf.reshape(s, (-1, self.n_head, self.r, self.d // self.n_head))
        z = tf.reshape(z, (-1, self.n_head, 1, self.r))

        pkey_T = tf.transpose(phi_1(key), (0, 1, 3, 2))  # tf.transpose(self.phi(key), (0, 1, 3, 2))
        new_s = s + pkey_T @ value
        new_z = z + phi_2(key)
        num = phi_3(query) @ new_s
        new_z_T = tf.transpose(new_z, (0, 1, 3, 2))

        den = phi_4(query) @ new_z_T  # self.phi(query) @ new_z_T
        head = num / den
        head = tf.squeeze(head, axis=2)
        heads = self.merge_heads(head)
        attention = self.attn_c_proj(heads)

        new_s = tf.reshape(new_s, (-1, self.d * self.r))
        new_z = tf.reshape(new_z, (-1, self.r * self.n_head))
        return attention, new_s, new_z

    def ffn(self, x):
        c_fc = self.c_fc(x)
        gelu = self.gelu(c_fc)
        c_proj = self.c_proj(gelu)
        return c_proj

    def call(self, inputs, states, training=None):
        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        self.b = tf.shape(inputs)[0]
        assert inputs.shape[1] == self.d

        s, z = states
        ln_1 = self.ln_1(inputs)

        attention, new_s, new_z = self.multi_head_attention(ln_1, s, z)

        ln_2 = self.ln_2(self.d_1(attention) + inputs)
        ffn = self.ffn(ln_2)

        output = (ln_2 + self.d_2(ffn))
        new_state = (new_s, new_z)
        return output, new_state


class lsnnGPT2cell(linearGPT2cell):
    def __init__(self, d, n_head, kernel_type):
        super().__init__(d, n_head, kernel_type)
        self.state_size = (d * self.r, self.r * self.n_head, d, d)
        self.thr = .03
        self.beta_lsnn = 1.8
        self.dampening_factor = 0.3
        self.tau_adaptation = 30.

    def spike(self, output, thr):
        v_sc = (output - thr) / thr
        new_spike = SpikeFunction(v_sc, self.dampening_factor)
        new_spike.set_shape(v_sc.get_shape())
        return new_spike

    def call(self, inputs, states, training=None):
        if not training is None:
            tf.keras.backend.set_learning_phase(training)
        self.b = tf.shape(inputs)[0]
        assert inputs.shape[1] == self.d

        s, z, old_spike, old_a = states
        ln_1 = self.ln_1(inputs)

        attention, new_s, new_z = self.multi_head_attention(ln_1, s, z)
        ln_2 = self.ln_2(self.d_1(attention) + inputs)
        ffn = self.ffn(ln_2)
        output = ln_2 + self.d_2(ffn)

        rho = tf.exp(-1 / self.tau_adaptation)
        new_a = rho * old_a + (1 - rho) * old_spike
        thr = self.thr + new_a * self.beta_lsnn

        new_spike = self.spike(output, thr)

        output = (new_spike)
        new_state = (new_s, new_z, new_spike, new_a)
        return output, new_state


class alsnnGPT2cell(lsnnGPT2cell):

    def build(self, input_shape):
        parameter2trainable = {k: v for k, v in self.__dict__.items()
                               if k in ['tau_adaptation', 'thr', 'beta_lsnn']}
        for k, p in parameter2trainable.items():
            initializer = tf.keras.initializers.TruncatedNormal(mean=p, stddev=3 * p / 7)
            p = self.add_weight(shape=(self.d,), initializer=initializer, name=k)
            self.__dict__.update({k: p})

        self.built = True


class alsnnSynthGPT2cell(alsnnGPT2cell):
    def build(self, input_shape):
        super(alsnnSynthGPT2cell, self).build(input_shape)

        self.synth_q = self.add_weight(shape=(self.d,), initializer=tf.keras.initializers.RandomNormal(),
                                       name='synth_q')
        self.synth_k = self.add_weight(shape=(self.d,), initializer=tf.keras.initializers.RandomNormal(),
                                       name='synth_k')

        self.built = True

    def multi_head_attention(self, x, s, z):
        # c_attn:     (768, 2304) + (1, 2304)
        # c_proj:     (768, 768) + (1, 768)
        # phi_1 = true_phi(self.n_head, self.d, b=self.b)
        phi_1 = new_phi(self.n_head, self.d, r=self.r, kernel_type=self.kernel_type)
        phi_2 = new_phi(self.n_head, self.d, r=self.r, kernel_type=self.kernel_type)
        phi_3 = new_phi(self.n_head, self.d, r=self.r, kernel_type=self.kernel_type)
        phi_4 = new_phi(self.n_head, self.d, r=self.r, kernel_type=self.kernel_type)

        x = self.c_attn(x)
        _, _, value = tf.split(x, 3, axis=1)
        sq = self.c_attn(self.synth_q[None])
        query, _, _ = tf.split(sq, 3, axis=1)
        sk = self.c_attn(self.synth_k[None])
        _, key, _ = tf.split(sk, 3, axis=1)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        s = tf.reshape(s, (-1, self.n_head, self.r, self.d // self.n_head))
        z = tf.reshape(z, (-1, self.n_head, 1, self.r))

        pkey_T = tf.transpose(phi_1(key), (0, 1, 3, 2))  # tf.transpose(self.phi(key), (0, 1, 3, 2))
        new_s = s + pkey_T @ value
        new_z = z + phi_2(key)
        num = phi_3(query) @ new_s
        new_z_T = tf.transpose(new_z, (0, 1, 3, 2))

        den = phi_4(query) @ new_z_T  # self.phi(query) @ new_z_T
        head = num / den
        head = tf.squeeze(head, axis=2)
        heads = self.merge_heads(head)
        attention = self.attn_c_proj(heads)

        new_s = tf.reshape(new_s, (-1, self.d * self.r))
        new_z = tf.reshape(new_z, (-1, self.r * self.n_head))
        return attention, new_s, new_z


class mn_alsnnSynthGPT2cell(alsnnSynthGPT2cell):
    def build(self, input_shape):
        super().build(input_shape)
        n_input = input_shape[-1]
        initializer = tf.keras.initializers.RandomNormal(stddev=1. / tf.math.sqrt(float(n_input)))
        self.n_std = self.add_weight(shape=(self.d,), initializer=initializer, name='n_std_lsnn')

    def spike(self, new_v, thr, *args):
        is_train = tf.cast(tf.keras.backend.learning_phase(), tf.float32)
        new_v = new_v * (1 + is_train * self.n_std * tf.random.normal(tf.shape(new_v), mean=0.0))
        v_sc = (new_v - thr) / thr

        z = SpikeFunction(v_sc, self.dampening_factor)
        z.set_shape(v_sc.get_shape())
        return z


thismodule = sys.modules[__name__]


class linearGPT2(tf.keras.layers.Layer):

    def __init__(
            self,
            num_layers,
            d_model,
            num_heads,
            max_seq_len,
            vocab_size,
            net_name='linearGPT2cell',
            kernel_type='exp'):
        super(linearGPT2, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.net_name = net_name
        self.kernel_type = kernel_type

        self.embedding = EmbeddingLayer(self.vocab_size, self.d_model, name='wte')
        self.pos_embedding = PositionEmbeddingLayer(self.max_seq_len, self.d_model, name='wpe')

        assert net_name in ['linearGPT2cell', 'lsnnGPT2cell', 'alsnnGPT2cell', 'alsnnSynthGPT2cell',
                            'mn_alsnnSynthGPT2cell']
        cell = getattr(thismodule, net_name)

        self.decoder_layers = [
            RNN(cell(self.d_model, self.num_heads, self.kernel_type),
                return_sequences=True, name='h_._{}'.format(i))
            for i in range(self.num_layers)]
        self.layer_norm = LayerNormalization(name='ln_f')
        self.stoch_depth = StochasticDepth(.2)

    def get_config(self):
        return {
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'max_seq_len': self.max_seq_len,
            'vocab_size': self.vocab_size,
            'net_name': self.net_name,
            'kernel_type': self.kernel_type
        }

    def call(self, x, training=None, past=None):

        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        x = tf.cast(x, tf.int32)
        # self.batch_size, self.sequence = tf.shape(x)[0], tf.shape(x)[1]
        if past is None:
            pasts = [None] * self.num_layers
        else:
            pasts = past

        assert len(pasts) == self.num_layers

        past_length = 1 if past is None else tf.shape(past)[-2]
        with tf.name_scope("embeddings"):
            hidden_states = self.embedding(x) + self.pos_embedding(x, start=past_length)

        presents = []
        for decoder_layer, past in zip(self.decoder_layers, pasts):
            output = decoder_layer(hidden_states, training=training)  # , att_mask, past=past)
            hidden_states = self.stoch_depth([output, hidden_states])
            # stochastic depth

            # presents.append(present)

        hidden_states = self.layer_norm(hidden_states)
        logits = self.embedding(hidden_states, mode="projection")
        return logits
