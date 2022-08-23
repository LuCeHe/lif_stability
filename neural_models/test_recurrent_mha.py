import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

phi = lambda x: tf.math.softplus(x)

def performer_attention(q, k, v, mask, attn_type='cumsum'):
    k = phi(k)
    q = phi(q)

    # Causal, during training
    if attn_type =='cumsum':

        key_T = tf.transpose(k, (0, 1, 3, 2))
        s = tf.math.cumsum(tf.matmul(key_T, v), axis=1)  # (2, 3, dk, vd)
        z = tf.math.cumsum(key_T, axis=1)  # (2, 3, dk, vd)
        print('z: ', z.shape, 's: ', s.shape)
        print('q: ', q.shape)

        # has to be the initial shape after this multiplication (batch_size, num_heads, q_len, v_d)
        num = q@s

        # Causal, at inference time
    elif attn_type =='recurrent':
        # self.z = k_prime_t if self.z is None else self.z + k_prime_t  # Incrementally sum over positions
        # denom = q_prime @ self.z
        pass


    return num, tf.zeros_like(num)


class MinimalMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

    def get_config(self):
        config = {"d_model": self.d_model, "num_heads": self.num_heads, }
        return super().get_config().update(**config)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None, training=None):
        v, k, q = inputs
        assert tf.shape(q)[-1] == self.d_model
        batch_size = tf.shape(q)[0]

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, d_model)
        output = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        return output, attention_weights



class MinimalPerformerCumsum(MinimalMultiHeadAttention):

    def call(self, inputs, mask=None, training=None):
        v, k, q = inputs
        assert tf.shape(q)[-1] == self.d_model
        batch_size = tf.shape(q)[0]

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        k = phi(k)
        q = phi(q)

        key_T = tf.transpose(k, (0, 1, 3, 2))
        s = tf.math.cumsum(tf.matmul(key_T, v), axis=1)  # (2, 3, dk, vd)
        scaled_attention = q @ s

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, d_model)
        output = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        return output, tf.zeros_like(output)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


if __name__ == '__main__':
    bs, sl, dm = 2, 3, 4
    nh = 2
    x = tf.random.normal((bs, sl, dm))
    look_ahead_mask = create_look_ahead_mask(sl)

    mha = MinimalMultiHeadAttention(dm, nh)

    o, _ = mha(x, x, x, look_ahead_mask)
    print(x)
    print(o)
