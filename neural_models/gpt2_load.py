import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import numpy as np
from tensorflow_addons.callbacks import TimeStopping
from transformers import TFGPT2LMHeadModel

from GenericTools.keras_tools.esoteric_layers.dropword import DropWord
from GenericTools.keras_tools.esoteric_optimizers.AdaBelief import AdaBelief

from stochastic_spiking.generate_data.huggingface_generator import HuggingfaceGenerator
from stochastic_spiking.neural_models.configuration_performer_attention_spiking import SpikingPerformerAttentionConfig
from stochastic_spiking.neural_models.gpt2_recurrent import linearGPT2
from stochastic_spiking.neural_models.spiking_performer import SpikingGPT2MainLayer

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'wikitext2'))
GPT2PATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'gpt2'))
TOKPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'gpt2_tokenizer'))

if not os.path.isdir(DATAPATH): os.mkdir(DATAPATH)
if not os.path.isdir(GPT2PATH): os.mkdir(GPT2PATH)
if not os.path.isdir(TOKPATH): os.mkdir(TOKPATH)


def load_gpt2_modified(pretrained, gpt2_config, net_name, kernel_type):
    if pretrained:
        return load_pretrained(gpt2_config, net_name, kernel_type)
    else:
        return load_nonpretrained(gpt2_config, net_name.replace('_freezed', ''), kernel_type)


def load_nonpretrained(gpt2_config, net_name, kernel_type):
    vocab_size = gpt2_config.vocab_size
    num_layers = gpt2_config.n_layer
    d_model = gpt2_config.n_embd
    num_heads = gpt2_config.n_head
    max_seq_len = gpt2_config.n_positions

    # assert num_layers == 12 and d_model == 768 and num_heads == 12 \
    #        and max_seq_len == 1024 and vocab_size == 50257

    if net_name == 'gpt2':

        from transformers import TFAutoModelForCausalLM, AutoConfig
        input_sentence = Input((None,))
        sentence = Lambda(lambda x: tf.cast(x, tf.int32))(input_sentence)
        # gpt2 = TFAutoModelForCausalLM.from_config(AutoConfig.from_pretrained('gpt2'))
        gpt2 = TFGPT2LMHeadModel.from_pretrained('gpt2')

        output = gpt2(sentence).logits
        model = Model(input_sentence, output)

    elif 'spiking_gpt2' in net_name:
        config = SpikingPerformerAttentionConfig
        if ':' in net_name:
            # e.g.: net_name = 'spiking_gpt2:d_model = 768:num_heads = 12:vocab_size = 50257:spiking = True:normalize_output = True:use_orthogonal_features = True:num_layers = 4:synthesizer = True:learnable_noise = True:learnable_threshold = True:use_linear_layers = True'
            # setattr(config, attr, 'magic')
            pass

        else:
            config.d_model = d_model
            config.num_heads = num_heads
            config.vocab_size = vocab_size
            config.causal = True
            config.spiking = True
            config.normalize_output = True
            config.attention_dropout = 0.1
            config.use_orthogonal_features = True
            config.num_layers = 4 #num_layers
            config.synthesizer = True
            config.learnable_noise = True
            config.learnable_threshold = True
            config.use_linear_layers = True



        sgpt2 = SpikingGPT2MainLayer(config)

        il = tf.keras.layers.Input((None,))
        out = sgpt2(il)
        model = tf.keras.models.Model(il, out)

    else:
        input_sentence = Input((None,))
        output = linearGPT2(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads,
            max_seq_len=max_seq_len, vocab_size=vocab_size,
            net_name=net_name, kernel_type=kernel_type)(input_sentence)
        # 124439808 params, original small gpt2: 124439808
        model = Model(input_sentence, output, name=net_name)
    return model


not_gpt_layers = ['thr', 'tau', 'beta_lsnn', 'synth_q', 'synth_k', 'n_std_lsnn']


def load_pretrained(gpt2_config, net_name, kernel_type):
    model = load_nonpretrained(gpt2_config, net_name.replace('_freezed', ''), kernel_type)
    from transformers import TFGPT2LMHeadModel

    weights = []
    original_gpt2 = TFGPT2LMHeadModel.from_pretrained(GPT2PATH)
    gpt2_layer_names = [l.name for l in original_gpt2.layers[0].weights]
    # appr_layer_names = [l.name for l in model.layers[1].weights]

    idx = 1 if not net_name == 'gpt2' else 2
    print('here')
    print(model.layers)
    for w in model.layers[idx].weights:
        wng = None
        n = 0
        clue = 0
        for word in ['/wte/', '/wpe/']:
            if word in w.name:
                n = [l for l in gpt2_layer_names if word in l][0]
                wng = [w for w in original_gpt2.layers[0].weights if w.name == n][0]

        if any([word in w.name for word in not_gpt_layers]):
            wng = w

        elif 'h_._' in w.name:
            list_scopes = w.name.split('/')
            number = int([w for w in list_scopes if 'h_._' in w][0].replace('h_._', ''))
            needed_i = w.name.split('/').index('h_._{}'.format(number))
            clue = 'h_._{}/'.format(number) + '/'.join(w.name.split('/')[needed_i + 1:]).replace('kernel', 'weight')

            n = [l for l in gpt2_layer_names if clue in l][0]

            wng = [w for w in original_gpt2.layers[0].weights if w.name == n][0]

        elif 'ln_f' in w.name:
            needed_i = w.name.split('/').index('ln_f')
            clue = '/'.join(w.name.split('/')[needed_i:])
            n = [l for l in gpt2_layer_names if clue in l][0]
            wng = [w for w in original_gpt2.layers[0].weights if w.name == n][0]

        if wng is None:
            wng = w

        print()
        print(clue)
        print(w.name)
        print(n)
        wg = wng

        if wg.shape in [(1, 2304), (1, 768), (1, 3072)] and not net_name == 'gpt2':
            weights.append(wg[0].numpy())
        else:
            weights.append(wg.numpy())

    model.set_weights(weights)
    del original_gpt2
    from transformers import TFGPT2LMHeadModel
    original_gpt2 = TFGPT2LMHeadModel.from_pretrained(GPT2PATH)

    # for layer, gpts_layer in zip([model.layers[1]], original_gpt2.layers):
    #     for w, wg in zip(layer.weights, gpts_layer.weights):
    #         print(w.numpy() == wg.numpy())

    gpt2_layer_names = [l.name for l in original_gpt2.layers[0].weights]

    import numpy as np

    for w in model.layers[idx].weights:
        print(w.name)
        wng = None
        for word in ['/wte/', '/wpe/']:
            if word in w.name:
                n = [l for l in gpt2_layer_names if word in l][0]
                wng = [w for w in original_gpt2.layers[0].weights if w.name == n][0]

        if any([word in w.name for word in not_gpt_layers]):
            wng = np.zeros_like(w)

        elif 'h_._' in w.name:
            list_scopes = w.name.split('/')
            number = int([w for w in list_scopes if 'h_._' in w][0].replace('h_._', ''))
            needed_i = w.name.split('/').index('h_._{}'.format(number))
            clue = 'h_._{}/'.format(number) + '/'.join(w.name.split('/')[needed_i + 1:]).replace('kernel', 'weight')
            n = [l for l in gpt2_layer_names if clue in l][0]
            wng = [w for w in original_gpt2.layers[0].weights if w.name == n][0]

        elif 'ln_f' in w.name:
            needed_i = w.name.split('/').index('ln_f')
            clue = '/'.join(w.name.split('/')[needed_i:])
            n = [l for l in gpt2_layer_names if clue in l][0]
            wng = [w for w in original_gpt2.layers[0].weights if w.name == n][0]

        try:
            is_eq = np.all(w.numpy() == wng.numpy())
        except:
            is_eq = np.all(w.numpy() == wng)

        print('loaded from gpt2: ', is_eq)
        if 'freezed' in net_name:
            if not any([word in w.name for word in not_gpt_layers]):
                w._trainable = False
        print('trainable: ', w.trainable)

    del original_gpt2
    return model


def study_pratrained_layers():
    from transformers import TFGPT2Model, GPT2Config
    # original_gpt2 = TFGPT2LMHeadModel.from_pretrained('gpt2')
    configuration = GPT2Config()
    original_gpt2 = TFGPT2Model(configuration)
    original_gpt2.summary()

    matrices = [w for w in original_gpt2.layers[0].weights if not w.shape[0] == 1]
    biases = [w for w in original_gpt2.layers[0].weights if w.shape[0] == 1]

    import matplotlib.pyplot as plt
    import numpy as np

    for m in matrices:
        print(m.shape)
        # _, s, _ = np.linalg.svd(m.numpy(), full_matrices=True)
        s, _, _ = tf.linalg.svd(m)

        plt.hist(s, 50, density=True, facecolor='g', alpha=0.75)
        plt.show()
        print(np.mean(s), np.std(s))
    print(len(matrices))
    plt.plot()


def download_models():
    from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
    original_gpt2 = TFGPT2LMHeadModel.from_pretrained('gpt2')
    original_gpt2.save_pretrained(GPT2PATH)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.save_pretrained(TOKPATH)


def quantify_quality_logits(x=None, T=1):
    if x is None:
        x = tf.constant([[[0., 1.], [1., 1.]]])
    s, _, _ = tf.linalg.svd(x)
    loss = tf.math.reduce_mean(tf.abs(tf.math.reduce_std(s, axis=1)))

    # how many classes have more prob than if it was uniform
    uniform_prob = tf.cast(1 / tf.shape(x)[-1], tf.float32)
    probs = tf.math.softmax(x / T)
    multim = tf.cast(tf.math.greater(probs, uniform_prob), tf.float32)
    mean_multimodality = tf.reduce_mean(tf.reduce_sum(multim, axis=2))
    max_multimodality = tf.reduce_max(tf.reduce_sum(multim, axis=2))

    dict_results = {
        'mean x': tf.reduce_mean(x).numpy(),
        'std x': tf.math.reduce_std(x).numpy(),
        'mean s': tf.reduce_mean(s).numpy(),
        # 'mean abs s': tf.reduce_mean(tf.abs(s)).numpy(),
        'mean abs std': loss.numpy(),
        'uniform_prob': uniform_prob.numpy(),
        'mean multimodality': mean_multimodality.numpy(),
        'max multimodality': max_multimodality.numpy(),
    }
    # from pprint import pprint
    # pprint(dict_results)
    return dict_results


def study_logits_behavior(test_random=False, test_real=True):
    from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
    import numpy as np
    import pandas as pd
    from ndatasets import load_dataset

    reps_random = 1
    reps_real = 6
    batch_size, seq_len = 4, 100

    models = {
        # 'gpt2-u': TFAutoModelForCausalLM.from_config(AutoConfig.from_pretrained('gpt2')),
        'gpt2-t': TFGPT2LMHeadModel.from_pretrained(GPT2PATH)
    }

    metrics = quantify_quality_logits()
    df = pd.DataFrame(metrics.keys(), columns=['metrics'])

    for k, gpt2_model in models.items():

        # to random input
        if test_random:
            test_type = 'random'
            print('\n{}\n'.format(test_type))
            for _ in range(reps_random):
                print('--------------------------')
                random_batch = np.random.choice(gpt2_model.config.vocab_size, size=(batch_size, seq_len))
                prediction = gpt2_model.predict(random_batch)
                prediction = prediction if not hasattr(prediction, 'logits') else prediction.logits
                metrics = quantify_quality_logits(prediction)

                df['{} {}'.format(k, test_type)] = metrics.values()

        # to language input
        if test_real:
            test_type = 'real'
            print('\n{}\n'.format(test_type))
            tokenizer = GPT2Tokenizer.from_pretrained(TOKPATH)
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            # print(tokenizer.eos_token)
            # print(tokenizer.bos_token)
            # print(tokenizer.pad_token_id)
            # print(tokenizer.decode(0))

            dataset = load_dataset('wikitext', 'wikitext-2-v1', split='validation[:10%]', cache_dir=DATAPATH)
            dataset = dataset.filter(lambda example: not example['text'] == '')
            for _ in range(reps_real):
                print('--------------------------')
                # initial_sample = np.random.choice(50)
                batch_samples = np.random.choice(len(dataset), batch_size, replace=False).tolist()
                print(dataset[:3])
                batch = np.array(dataset['text'])[batch_samples].tolist()
                print(batch)
                generated = tokenizer(batch, return_tensors="tf", padding=True, truncation=True)
                print(generated['attention_mask'].shape)
                print(generated['input_ids'].shape)
                prediction = gpt2_model.predict(generated['input_ids'])
                prediction = prediction if not hasattr(prediction, 'logits') else prediction.logits
                T = .5 * tf.pow(10, tf.random.uniform((), minval=-1, maxval=1.))
                metrics = quantify_quality_logits(prediction, T)

                rounded_T = np.round(T, 2)
                print(rounded_T)
                df['{} {} {}'.format(k, test_type, rounded_T)] = metrics.values()

    df = df.set_index('metrics')
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    pd.set_option('max_colwidth', 1)
    pd.set_option('precision', 2)
    pd.options.display.width = 500

    print(df)


def finetune_random_input(test_random=False, test_real=True):
    from transformers import TFGPT2LMHeadModel, TFAutoModelForCausalLM, AutoConfig

    reps_random = 1
    reps_real = 6
    batch_size, seq_len = 4, 100

    models = {
        'gpt2-u': TFAutoModelForCausalLM.from_config(AutoConfig.from_pretrained('gpt2')),
        'gpt2-t': TFGPT2LMHeadModel.from_pretrained(GPT2PATH)
    }


def generator_distill(model, batch_size, seq_len, vocab_size):
    while True:
        random_batch = np.random.choice(vocab_size, size=(batch_size, seq_len))
        prediction = model.predict(random_batch)
        yield random_batch, prediction.logits


class DistillLoss(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        teacher, student = inputs

        is_train = tf.cast(tf.keras.backend.learning_phase(), tf.float32)

        T = is_train * .5 * tf.pow(10., tf.random.uniform((), minval=-1, maxval=1.)) + (1 - is_train) * 1

        softmax_teacher = tf.math.softmax(teacher.logits / T, axis=-1)
        softmax_student = tf.math.softmax(student / T, axis=-1)
        loss = tf.keras.losses.CategoricalCrossentropy()(softmax_teacher, softmax_student)
        self.add_loss(loss)

        self.add_metric(loss, name='distill_distance', aggregation='mean')
        return student



def distill_gpt2(training_type, recurrent_model, gpt2_model, batch_size, seq_len, epochs, steps_per_epoch,
                 learning_rate, clipnorm, weight_decay):
    if training_type == 'random':

        generator = generator_distill(gpt2_model, batch_size, seq_len, gpt2_model.config.vocab_size)
        optimizer = AdaBelief(learning_rate=learning_rate, clipnorm=clipnorm, weight_decay=weight_decay)
        recurrent_model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['cosine_similarity', 'mse', 'categorical_crossentropy'])
        history = recurrent_model.fit(
            generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=1,
            validation_data=generator)

    elif training_type == 'distillation':

        new_input = Input((None,), dtype=tf.int32)
        mask = Input((None, 1))
        mlm = DropWord(.1, gpt2_model.config.vocab_size)(new_input)

        gpt_output = gpt2_model(new_input)
        rec_output = recurrent_model(mlm)

        rec_output = rec_output if not hasattr(rec_output, 'logits') else rec_output.logits
        rec_output = DistillLoss()([gpt_output, rec_output])
        masked_output = mask * rec_output
        distillation_model = Model([new_input, mask], masked_output)
        optimizer = AdaBelief(learning_rate=learning_rate, clipnorm=clipnorm, weight_decay=weight_decay)

        # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=.1)
        distillation_model.compile(
            optimizer=optimizer,
            loss=lambda x: 0,
            metrics=['cosine_similarity', 'mse', 'categorical_crossentropy'])

        train_generator = HuggingfaceGenerator(epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                                               train_val_test='train', dataset_name='bookcorpus', mode='distillation',
                                               maxlen=seq_len)
        val_generator = HuggingfaceGenerator(epochs=epochs, batch_size=batch_size, steps_per_epoch=2,
                                             train_val_test='validation', dataset_name='wikitext-2', mode='distillation',
                                             maxlen=seq_len)
        callbacks = [
            TimeStopping(43200, 1),  # 22h=79200 s, 21h=75600 s, 12h = 43200 s
        ]
        history = distillation_model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks)
        # raise NotImplementedError

    elif training_type == 'bookcorpus':

        new_input = Input((None,), dtype=tf.int32)
        mask = Input((None, 1))
        mlm = DropWord(.1, gpt2_model.config.vocab_size)(new_input)

        rec_output = recurrent_model(mlm)

        rec_output = rec_output if not hasattr(rec_output, 'logits') else rec_output.logits
        masked_output = mask * rec_output
        distillation_model = Model([new_input, mask], masked_output)
        optimizer = AdaBelief(learning_rate=learning_rate, clipnorm=clipnorm, weight_decay=weight_decay)

        # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=.1)
        distillation_model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['cosine_similarity', 'mse', 'categorical_crossentropy'])

        train_generator = HuggingfaceGenerator(epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                                               train_val_test='train', dataset_name='bookcorpus', mode='distillation',
                                               maxlen=seq_len)
        val_generator = HuggingfaceGenerator(epochs=epochs, batch_size=batch_size, steps_per_epoch=2,
                                             train_val_test='validation', dataset_name='wikitext-2', mode='distillation',
                                             maxlen=seq_len)
        callbacks = [
            TimeStopping(75600, 1),  # 22h=79200 s, 21h=75600 s, 12h = 43200 s
        ]
        history = distillation_model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks)
        # raise NotImplementedError
    else:
        raise NotImplementedError
    return history


if __name__ == '__main__':
    download_models()
    # study_pratrained_layers()
    # download_dataset()
    # study_logits_behavior()
    # finetune_random_input()
