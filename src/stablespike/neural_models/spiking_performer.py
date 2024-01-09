import tensorflow as tf
import numpy as np
from pyaromatics.keras_tools.esoteric_layers.positional_embedding import SymbolAndPositionEmbedding
from stablespike.neural_models.configuration_performer_attention_spiking import SpikingPerformerAttentionConfig
from stablespike.neural_models.modeling_tf_performer_attention_spiking import TFSpikingPerformerAttention
from transformers.modeling_tf_utils import TFConv1D

tf.config.run_functions_eagerly(True)


class TFMLP(tf.keras.layers.Layer):
    def __init__(self, n_state, config, **kwargs):
        super(TFMLP, self).__init__(**kwargs)
        nx = config.d_model
        self.c_fc = TFConv1D(n_state, nx, name='c_fc')
        self.c_proj = TFConv1D(nx, n_state, name='c_proj')
        self.act = lambda x: x  # gelu
        self.dropout = tf.keras.layers.Dropout(config.residual_dropout)

    def call(self, x, training=False):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)
        return h2


class SpikingBlock(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        nx = config.d_model
        self.ln_1 = tf.keras.layers.LayerNormalization(name='ln_1') if config.layer_normalizations else lambda x: x
        # self.attn = TFAttention(nx, n_ctx, config, scale, name='attn')
        self.attn = TFSpikingPerformerAttention(config)
        self.ln_2 = tf.keras.layers.LayerNormalization(name='ln_2') if config.layer_normalizations else lambda x: x
        self.mlp = TFMLP(4 * nx, config, name='mlp')

    def call(self, inputs, training=False):
        a = self.ln_1(inputs)
        output_attn = self.attn([a, a, a])
        a = output_attn[0]  # output_attn: a, present, (attentions)
        x = inputs + a

        m = self.ln_2(x)
        m = self.mlp(m, training=training)
        x = x + m

        return x  # x, present, (attentions)


class SpikingGPT2MainLayer(tf.keras.layers.Layer):

    def get_config(self):
        return {'config': self.config}

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # self.output_hidden_states = config.output_hidden_states
        # self.output_attentions = config.output_attentions
        self.num_hidden_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.config = config

        if config.embedding:
            self.wte = SymbolAndPositionEmbedding(config.maxlen, vocab_size=config.vocab_size, embed_dim=config.d_model,
                                                  embeddings_initializer=config.initializer,
                                                  name='wte', symbol_embedding='zero_mean',
                                                  position_embedding='None',
                                                  factorized_dim=int(np.sqrt(config.vocab_size)))
        else:
            self.wte = lambda x: x

        self.drop = tf.keras.layers.Dropout(config.attention_dropout)
        self.h = [
            SpikingBlock(config, name='h_._{}'.format(i))
            for i in range(config.num_layers)
        ]
        self.ln_f = tf.keras.layers.LayerNormalization(name='ln_f') if config.layer_normalizations else lambda x: x

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    def call(self, inputs, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
             training=None):

        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        # assert len(inputs) == 1
        if self.config.embedding:
            inputs_embeds = self.wte(inputs, mode='embedding')
        else:
            inputs_embeds = inputs

        hidden_states = self.drop(inputs_embeds)
        for i, block in enumerate(self.h):
            # outputs = block([hidden_states, layer_past, attention_mask, head_mask[i]], training=training)
            b = block(hidden_states)
            if self.config.skip_connections:
                hidden_states = hidden_states + b
            else:
                hidden_states = b

        hidden_states = self.ln_f(hidden_states)

        if self.config.embedding:
            output = self.wte(hidden_states, mode='projection')
        else:
            output = hidden_states

        return output  # last hidden state, presents, (all hidden_states), (attentions)


class spikingPerformer(SpikingGPT2MainLayer):

    def get_config(self):
        return {'num_neurons': self.num_neurons, 'config': self.config}

    def __init__(self, num_neurons=None, *args, **kwargs):

        self.num_neurons = num_neurons
        config = SpikingPerformerAttentionConfig
        config.d_model = num_neurons
        if num_neurons > 10 and num_neurons % 10 == 0:
            config.num_heads = 10
        else:
            config.num_heads = 2

        config.vocab_size = 73
        config.causal = True
        config.spiking = True
        config.normalize_output = True
        config.attention_dropout = 0.0
        config.use_orthogonal_features = True
        config.num_layers = 6
        config.thr = 0.
        config.noise_std = 0.
        config.learnable_threshold = True
        config.embedding = False
        config.beta_noise = False
        config.synthesizer = False
        config.use_linear_layers = True

        # e.g.
        # comments = 'spikingperformer::-normalize_output:True-use_linear_layers:True-synthesizer:True-' \
        #         'attention_dropout:.2-learnable_threshold:True-beta_noise:True-noise_std:.1-skip_connections:True'

        props = ['normalize_output', 'use_linear_layers', 'synthesizer', 'attention_dropout',
                 'learnable_threshold', 'beta_noise', 'noise_std', 'skip_connections']
        if 'spikingperformer::' in kwargs['comments']:
            options = kwargs['comments'].split('-')

            for p in props:
                if p in options:
                    o = [o.replace(p, '').replace(':', '') for o in options if p in o][0]

                    if o == 'False':
                        o = False
                    elif o == 'True':
                        o = True
                    else:
                        o = float(o)

                    setattr(config, p, o)

        del kwargs['comments']
        self.config = config

        super().__init__(config=config, *args, **kwargs)


if __name__ == '__main__':
    import numpy as np

    vb = 50000
    bs, sl, dm = 2, 3, 768
    nh = 12

    config = SpikingPerformerAttentionConfig
    config.d_model = dm
    config.num_heads = nh
    config.vocab_size = vb
    config.causal = True
    config.spiking = True
    config.normalize_output = True
    config.attention_dropout = 0.0
    config.use_orthogonal_features = True
    config.num_layers = 12

    sgpt2 = SpikingGPT2MainLayer(config)

    il = tf.keras.layers.Input((sl,))
    out = sgpt2(il)
    model = tf.keras.models.Model(il, out)

    model.summary()

    batch = np.random.choice(vb, size=(bs, sl,))
    prediction = model.predict(batch)
    print(prediction.shape)
