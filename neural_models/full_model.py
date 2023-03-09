import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.losses import sparse_categorical_crossentropy

from GenericTools.keras_tools.esoteric_layers import *
from GenericTools.keras_tools.esoteric_layers.combine_tensors import CombineTensors
# from GenericTools.keras_tools.esoteric_models.model import modifiedModel
from GenericTools.keras_tools.esoteric_optimizers.optimizer_selection import get_optimizer
from GenericTools.stay_organized.utils import str2val
from GenericTools.keras_tools.esoteric_losses.loss_redirection import get_loss
from GenericTools.keras_tools.esoteric_losses.advanced_losses import *

import sg_design_lif.neural_models as models
from GenericTools.keras_tools.esoteric_tasks.time_task_redirection import language_tasks

metrics = [
    sparse_categorical_accuracy,
    bpc,
    perplexity,
    # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    sparse_mode_accuracy,
    sparse_categorical_crossentropy,
]


def layers_combination(output_cell, all_outputs, in_expert, comments, i, n_neurons, tff, drate):
    if 'addrestrellis' in comments:
        output_cell = tf.keras.layers.Add()([output_cell, in_expert])

    elif 'combinedconcrestrellis' in comments and len(all_outputs) > 1:
        sigmoidal_gating = True if 'sigcombined' in comments else False
        axis = 2 if 'trellis2' in comments else None
        ct = CombineTensors(n_tensors=len(all_outputs[:-1]), sigmoidal_gating=sigmoidal_gating, axis=axis)(
            all_outputs[:-1])
        output_cell = Concatenate(axis=-1)([output_cell, ct])

    elif 'combinedrestrellis' in comments and len(all_outputs) > 1:
        sigmoidal_gating = True if 'sigcombined' in comments else False
        axis = 2 if 'trellis2' in comments else None
        output_cell = CombineTensors(
            n_tensors=len(all_outputs), sigmoidal_gating=sigmoidal_gating, axis=axis
        )(all_outputs)

    elif 'concrestrellis' in comments:
        output_cell = Concatenate(axis=-1)([output_cell, in_expert[:, :, :n_neurons]])

    if 'transformer' in comments:

        if 'dilation' in comments:
            dilation = 2 ** i
        else:
            dilation = 1

        if 'drop1' in comments:
            output_cell = tf.keras.layers.Dropout(drate)(output_cell)

        conv = tff(n_neurons, dilation, comments)(output_cell)

        if 'drop2' in comments:
            conv = tf.keras.layers.Dropout(drate)(conv)

        if 'appenddff' in comments:
            all_outputs.append(conv)

        if 'combineddff' in comments and len(all_outputs) > 1:
            sigmoidal_gating = True if 'sigcombined' in comments else False
            axis = 2 if 'dff2' in comments else None
            output_cell = CombineTensors(
                n_tensors=len(all_outputs), sigmoidal_gating=sigmoidal_gating, axis=axis
            )(all_outputs)
        else:
            output_cell = conv + output_cell

    return output_cell, all_outputs


def TransformerFF(n_neurons, dilation, comments):
    conv1 = tf.keras.layers.SeparableConv1D(filters=int(1.2 * n_neurons), kernel_size=4, dilation_rate=dilation,
                                            padding='causal', depth_multiplier=1)

    if 'nospikedff' in comments:
        spikes = lambda x: x
    else:
        spikes = lambda x: SurrogatedStep(string_config=comments)(x)
    output_shape = 2 * n_neurons if 'conc' in comments else n_neurons
    conv2 = tf.keras.layers.SeparableConv1D(filters=output_shape, kernel_size=4, dilation_rate=dilation,
                                            padding='causal', depth_multiplier=1)

    def call(inputs):
        return conv2(spikes(conv1(inputs)))

    return call


class Expert:

    def __init__(self, i, j, stateful, task_name, net_name, n_neurons, tau, initializer,
                 tau_adaptation, n_out, comments, initial_state=None):
        ij = '_{}_{}'.format(i, j)
        self.net_name = net_name
        self.comments = comments
        self.initial_state = initial_state
        thr = str2val(comments, 'thr', float, .01)

        if 'convWin' in comments:
            kernel_size = str2val(comments, 'ksize', int, 4)
            self.win = tf.keras.layers.Conv1D(filters=int(n_neurons), kernel_size=kernel_size, dilation_rate=1,
                                              padding='causal')
        else:
            self.win = lambda x: x

        batch_size = str2val(comments, 'batchsize', int, 1)
        maxlen = str2val(comments, 'maxlen', int, 100)
        nin = str2val(comments, 'nin', int, 1) if not 'convWin' in comments else n_neurons

        stack_info = '_stacki:{}'.format(i)
        if 'LSNN' in net_name:
            cell = models.net(net_name)(
                num_neurons=n_neurons, tau=tau, tau_adaptation=tau_adaptation,
                initializer=initializer, config=comments + stack_info, thr=thr)
            rnn = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True, name='encoder' + ij,
                                      stateful=stateful)
            rnn.build((batch_size, maxlen, nin))

        elif 'Performer' in net_name or 'GPT' in net_name:
            rnn = models.net(net_name)(num_neurons=n_neurons, comments=comments)

        elif net_name == 'LSTM':
            cell = tf.keras.layers.LSTMCell(units=n_neurons)
            rnn = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True, name='encoder' + ij,
                                      stateful=stateful)
            rnn.build((batch_size, maxlen, nin))

        elif net_name == 'GRU':
            cell = tf.keras.layers.GRUCell(units=n_neurons)
            rnn = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True, name='encoder' + ij,
                                      stateful=stateful)
            rnn.build((batch_size, maxlen, nin))

        else:
            raise NotImplementedError

        self.rnn = rnn

        # reg = models.RateVoltageRegularization(1., config=comments + task_name + stack_info, name='reg' + ij)

    def __call__(self, inputs):
        # skipped_connection_input = inputs
        skipped_connection_input = self.win(inputs)
        if 'LSNN' in self.net_name:
            all_out = self.rnn(inputs=skipped_connection_input, initial_state=self.initial_state)
            outputs, states = all_out[:4], all_out[4:]
            b, v, thr, v_sc = outputs

            if 'readout_voltage' in self.comments:
                output_cell = v
            else:
                output_cell = b

        elif 'LSTM' in self.net_name or 'GRU' in self.net_name:
            all_out = self.rnn(inputs=skipped_connection_input, initial_state=self.initial_state)
            output_cell, states = all_out[0], all_out[1:]
        else:
            raise NotImplementedError

        return output_cell, states


class ModelBuilder:
    def __init__(self, task_name, net_name, n_neurons, lr, stack,
                 loss_name, embedding, optimizer_name, lr_schedule, weight_decay, clipnorm,
                 initializer, comments, in_len, n_in, out_len, n_out, final_epochs,
                 initial_state=None, seed=None, get_embedding=False, timesteps=None):

        self.task_name, self.net_name, self.n_neurons = task_name, net_name, n_neurons
        self.n_in, self.n_out = n_in, n_out
        # task_name, net_name, n_neurons, lr, stack,
        # loss_name, embedding, optimizer_name, lr_schedule, weight_decay, clipnorm,
        self.loss_name, self.embedding, self.optimizer_name = loss_name, embedding, optimizer_name
        self.lr_schedule, self.weight_decay, self.clipnorm = lr_schedule, weight_decay, clipnorm

        # initializer, comments, in_len, n_in, out_len, n_out, final_epochs,
        # initializer, comments, in_len, n_in, out_len, n_out, final_epochs,
        # initializer, comments, in_len, n_in, out_len, n_out, final_epochs,
        self.final_epochs, self.lr = final_epochs, lr
        self.initial_state, self.get_embedding, self.timesteps = initial_state, get_embedding, timesteps

        comments = comments if task_name in language_tasks else comments.replace('embproj', 'simplereadout')

        tau_adaptation = str2val(comments, 'taub', float, default=int(in_len / 2))
        tau = str2val(comments, 'tauv', float, default=.1)
        self.drate = str2val(comments, 'dropout', float, .1)
        # network definition
        # weights initialization
        self.embedding = embedding if task_name in language_tasks else False
        stateful = True if 'ptb' in task_name else False

        if 'stateful' in comments: stateful = True

        self.loss = get_loss(loss_name)

        if 'shared' in comments:
            extper = Expert(0, 0, stateful, task_name, net_name, n_neurons, tau=tau, initializer=initializer,
                            tau_adaptation=tau_adaptation, n_out=n_out, comments=comments, init_states=initial_state)
            self.expert = lambda i, j, c, n, init_s: extper
        else:
            self.expert = lambda i, j, c, n, init_s: \
                Expert(i, j, stateful, task_name, net_name, n_neurons=n, tau=tau,
                       initializer=initializer, tau_adaptation=tau_adaptation, n_out=n_out,
                       comments=c, initial_state=init_s)

        self.emb = []
        if not self.embedding is False:
            self.emb = SymbolAndPositionEmbedding(
                maxlen=in_len, vocab_size=n_out, embed_dim=n_neurons, embeddings_initializer=initializer,
                from_string=embedding, name=embedding.replace(':', '_')
            )

            self.emb.build((1, n_out))
            self.emb.sym_emb.build((1, n_out))
            mean = np.mean(np.mean(self.emb.sym_emb.embeddings, axis=-1), axis=-1)
            var = np.mean(np.var(self.emb.sym_emb.embeddings, axis=-1), axis=-1)
            comments = str2val(comments, 'taskmean', replace=mean)
            comments = str2val(comments, 'taskvar', replace=var)
            comments = str2val(comments, 'embdim', replace=self.emb.embed_dim)

        self.readout = tf.keras.layers.Dense(n_out, name='decoder', kernel_initializer=initializer)

        # graph
        self.batch_size = str2val(comments, 'batchsize', int, 1)

        if isinstance(stack, str):
            self.stack = [int(s) for s in stack.split(':')]
        elif isinstance(stack, int):
            self.stack = [n_neurons for _ in range(stack)]

        self.comments = comments

        n_states = 4 if 'LSNN' in self.net_name else 2

        self.rnns = []
        self.all_input_states = []

        for i, layer_width in enumerate(self.stack):
            if i == 0:
                if not self.embedding is False:
                    nin = self.emb.embed_dim
                else:
                    nin = n_in
            else:
                nin = self.stack[i - 1]

            if not initial_state is None:
                initial_state = tuple([
                    tf.keras.layers.Input([layer_width, ], name=f'state_{i}_{si}')
                    for si in range(n_states)
                ])
                self.all_input_states.extend(initial_state)

            c = str2val(self.comments, 'nin', replace=nin)
            rnn = self.expert(i, 0, c, n=layer_width, init_s=initial_state)
            self.rnns.append(rnn)

    def input2output(self):
        input_words = tf.keras.layers.Input([self.timesteps, self.n_in], name='input_spikes',
                                            batch_size=self.batch_size)
        output_words = tf.keras.layers.Input([self.timesteps], name='target_words', batch_size=self.batch_size)

        x = input_words

        if not self.embedding is False:
            # in_emb = Lambda(lambda z: tf.math.argmax(z, axis=-1), name='Argmax')(x)
            if x.shape[-1] == 1:
                in_emb = tf.keras.layers.Lambda(lambda z: tf.squeeze(z, axis=-1), name='Squeeze')(x)
            else:
                in_emb = tf.keras.layers.Lambda(lambda z: tf.math.argmax(z, axis=-1), name='Argmax')(x)

            x = self.emb(in_emb)

        all_states = []
        for i, rnn in enumerate(self.rnns):
            x = tf.keras.layers.Dropout(self.drate, name=f'dropout_{i}')(x)
            x, states = rnn(x)

            if not self.initial_state is None:
                all_states.extend(states)

        if 'embproj' in self.comments:
            output_net = self.emb(x, mode='projection')
        else:
            output_net = self.readout(x)

        loss = str2val(self.comments, 'loss', output_type=str, default=self.loss)
        output_net = AddLossLayer(loss=loss)([output_words, output_net])
        output_net = AddMetricsLayer(metrics=metrics)([output_words, output_net])
        output_net = tf.keras.layers.Lambda(lambda z: z, name='output_net')(output_net)

        # train model
        if self.initial_state is None:
            # train_model = modifiedModel([input_words, output_words], output_net, name=net_name)
            train_model = tf.keras.models.Model([input_words, output_words], output_net, name=self.net_name)
        else:
            train_model = tf.keras.models.Model(
                [input_words, output_words] + self.all_input_states,
                [output_net] + all_states, name=self.net_name
            )
        exclude_from_weight_decay = ['decoder'] if 'dontdecdec' in self.comments else []

        optimizer_name = str2val(self.comments, 'optimizer', output_type=str, default=self.optimizer_name)
        lr_schedule = str2val(self.comments, 'lrs', output_type=str, default=self.lr_schedule)
        optimizer = get_optimizer(optimizer_name=optimizer_name, lr_schedule=lr_schedule,
                                  total_steps=self.final_epochs, lr=self.lr, weight_decay=self.weight_decay,
                                  clipnorm=self.clipnorm, exclude_from_weight_decay=exclude_from_weight_decay)

        train_model.compile(optimizer=optimizer, loss=None, run_eagerly=True)
        return train_model

    def input2embedding(self):
        input_words = tf.keras.layers.Input([self.timesteps, self.n_in], name='input_spikes',
                                            batch_size=self.batch_size)
        x = input_words

        if not self.embedding is False:
            # in_emb = Lambda(lambda z: tf.math.argmax(z, axis=-1), name='Argmax')(x)
            if x.shape[-1] == 1:
                in_emb = tf.keras.layers.Lambda(lambda z: tf.squeeze(z, axis=-1), name='Squeeze')(x)
            else:
                in_emb = tf.keras.layers.Lambda(lambda z: tf.math.argmax(z, axis=-1), name='Argmax')(x)

            x = self.emb(in_emb)

        emb_model = tf.keras.models.Model(input_words, x)
        return emb_model

    def embedding2output(self):


        input_embedding = tf.keras.layers.Input([self.timesteps, self.emb.embed_dim], name='input_spikes',
                                            batch_size=self.batch_size)
        output_words = tf.keras.layers.Input([self.timesteps], name='target_words', batch_size=self.batch_size)

        x = input_embedding

        all_states = []
        for i, rnn in enumerate(self.rnns):
            x = tf.keras.layers.Dropout(self.drate, name=f'dropout_{i}')(x)
            x, states = rnn(x)

            if not self.initial_state is None:
                all_states.extend(states)

        if 'embproj' in self.comments:
            output_net = self.emb(x, mode='projection')
        else:
            output_net = self.readout(x)

        loss = str2val(self.comments, 'loss', output_type=str, default=self.loss)
        output_net = AddLossLayer(loss=loss)([output_words, output_net])
        output_net = AddMetricsLayer(metrics=metrics)([output_words, output_net])
        output_net = tf.keras.layers.Lambda(lambda z: z, name='output_net')(output_net)

        # train model
        if self.initial_state is None:
            # train_model = modifiedModel([input_words, output_words], output_net, name=net_name)
            model = tf.keras.models.Model([input_embedding, output_words], output_net, name=self.net_name)
        else:
            model = tf.keras.models.Model(
                [input_embedding, output_words] + self.all_input_states,
                [output_net] + all_states, name=self.net_name
            )

        return model


def build_model(task_name, net_name, n_neurons, lr, stack,
                loss_name, embedding, optimizer_name, lr_schedule, weight_decay, clipnorm,
                initializer, comments, in_len, n_in, out_len, n_out, final_epochs,
                initial_state=None, seed=None, get_embedding=False, timesteps=None):
    model_builder = ModelBuilder(task_name, net_name, n_neurons, lr, stack,
                                 loss_name, embedding, optimizer_name, lr_schedule, weight_decay, clipnorm,
                                 initializer, comments, in_len, n_in, out_len, n_out, final_epochs,
                                 initial_state, seed, get_embedding, timesteps)

    if not get_embedding:
        model = model_builder.input2output()
        return model
    else:
        model = model_builder.input2output()

        noemb_model = model_builder.embedding2output()
        embedding_model = model_builder.input2embedding()
        return model, noemb_model, embedding_model
