from tensorflow.keras.layers import *
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.losses import sparse_categorical_crossentropy

from GenericTools.keras_tools.esoteric_layers import *
from GenericTools.keras_tools.esoteric_layers.combine_tensors import CombineTensors
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
        output_cell = Add()([output_cell, in_expert])

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
            output_cell = Dropout(drate)(output_cell)

        conv = tff(n_neurons, dilation, comments)(output_cell)

        if 'drop2' in comments:
            conv = Dropout(drate)(conv)

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
    conv1 = SeparableConv1D(filters=int(1.2 * n_neurons), kernel_size=4, dilation_rate=dilation,
                            padding='causal', depth_multiplier=1)

    if 'nospikedff' in comments:
        spikes = lambda x: x
    else:
        spikes = lambda x: SurrogatedStep(string_config=comments)(x)
    output_shape = 2 * n_neurons if 'conc' in comments else n_neurons
    conv2 = SeparableConv1D(filters=output_shape, kernel_size=4, dilation_rate=dilation,
                            padding='causal', depth_multiplier=1)

    def call(inputs):
        return conv2(spikes(conv1(inputs)))

    return call


def Expert(i, j, stateful, task_name, net_name, n_neurons, tau, initializer,
           tau_adaptation, n_out, comments, initial_state=None):
    ij = '_{}_{}'.format(i, j)

    thr = str2val(comments, 'thr', float, .01)

    if 'convWin' in comments:
        kernel_size = str2val(comments, 'ksize', int, 4)
        win = Conv1D(filters=int(n_neurons), kernel_size=kernel_size, dilation_rate=1, padding='causal')
    else:
        win = lambda x: x

    batch_size = str2val(comments, 'batchsize', int, 1)
    maxlen = str2val(comments, 'maxlen', int, 100)
    nin = str2val(comments, 'nin', int, 1) if not 'convWin' in comments else n_neurons

    stack_info = '_stacki:{}'.format(i)
    if 'LSNN' in net_name:
        cell = models.net(net_name)(
            num_neurons=n_neurons, tau=tau, tau_adaptation=tau_adaptation,
            initializer=initializer, config=comments + stack_info, thr=thr)
        rnn = RNN(cell, return_state=True, return_sequences=True, name='encoder' + ij, stateful=stateful)
        rnn.build((batch_size, maxlen, nin))

    elif 'Performer' in net_name or 'GPT' in net_name:
        rnn = models.net(net_name)(num_neurons=n_neurons, comments=comments)

    elif net_name == 'LSTM':
        cell = tf.keras.layers.LSTMCell(units=n_neurons)
        rnn = RNN(cell, return_state=True, return_sequences=True, name='encoder' + ij, stateful=stateful)
        rnn.build((batch_size, maxlen, nin))

    elif net_name == 'GRU':
        cell = tf.keras.layers.GRUCell(units=n_neurons)
        rnn = RNN(cell, return_state=True, return_sequences=True, name='encoder' + ij, stateful=stateful)
        rnn.build((batch_size, maxlen, nin))

    else:
        raise NotImplementedError

    lsb = LayerSupervision(n_classes=n_out, name='b' + ij)
    lsv = LayerSupervision(n_classes=n_out, name='v' + ij)
    lst = LayerSupervision(n_classes=n_out, name='t' + ij)

    reg = models.RateVoltageRegularization(1., config=comments + task_name + stack_info, name='reg' + ij)

    def call(inputs):
        skipped_connection_input, output_words = inputs
        skipped_connection_input = win(skipped_connection_input)
        if 'LSNN' in net_name:
            all_out = rnn(inputs=skipped_connection_input, initial_state=initial_state)
            outputs, states = all_out[:4], all_out[4:]
            b, v, thr, v_sc = outputs

            b = reg([b, v_sc])

            if 'layersup' in comments:
                b = lsb([b, output_words])
                v = lsv([v, output_words])
                thr = lst([thr, output_words])
                b = CloseGraph()([b, v, thr, v_sc])

            if 'readout_voltage' in comments:
                output_cell = v
            else:
                output_cell = b

        elif 'LSTM' in net_name or 'GRU' in net_name:
            all_out = rnn(inputs=skipped_connection_input, initial_state=initial_state)
            output_cell, states = all_out[0], all_out[1:]
        else:
            output_cell = rnn(inputs=skipped_connection_input)

        return output_cell, states

    return call


def build_model(task_name, net_name, n_neurons, lr, stack,
                loss_name, embedding, optimizer_name, lr_schedule, weight_decay, clipnorm,
                initializer, comments, in_len, n_in, out_len, n_out, final_epochs,
                initial_state=None, seed=None):
    comments = comments if task_name in language_tasks else comments.replace('embproj', 'simplereadout')

    tau_adaptation = str2val(comments, 'taub', float, default=int(in_len / 2))
    tau = str2val(comments, 'tauv', float, default=.1)
    drate = str2val(comments, 'dropout', float, .1)
    # network definition
    # weights initialization
    embedding = embedding if task_name in language_tasks else False
    stateful = True if 'ptb' in task_name else False

    if 'stateful' in comments: stateful = True

    loss = get_loss(loss_name)

    n_experts = 1
    if 'experts:' in comments:
        n_experts = int([s for s in comments.split('_') if 'experts:' in s][0].replace('experts:', ''))

    if not embedding is False:
        embs = []
        if 'experts:' in comments and 'expertemb' in comments:
            for i in range(n_experts):
                emb = SymbolAndPositionEmbedding(
                    maxlen=in_len, vocab_size=n_out, embed_dim=n_neurons, embeddings_initializer=initializer,
                    from_string=embedding, name=embedding.replace(':', '_')
                )
                embs.append(emb)

        else:
            emb = SymbolAndPositionEmbedding(
                maxlen=in_len, vocab_size=n_out, embed_dim=n_neurons, embeddings_initializer=initializer,
                from_string=embedding, name=embedding.replace(':', '_')
            )
            embs.append(emb)

            emb.build((1, n_out))
            emb.sym_emb.build((1, n_out))
            mean = np.mean(np.mean(emb.sym_emb.embeddings, axis=-1), axis=-1)
            var = np.mean(np.var(emb.sym_emb.embeddings, axis=-1), axis=-1)
            comments = str2val(comments, 'taskmean', replace=mean)
            comments = str2val(comments, 'taskvar', replace=var)
            comments = str2val(comments, 'embdim', replace=emb.embed_dim)

    # graph
    # input_words = Input([in_len, n_in], name='input_spikes', batch_size=batch_size)
    # output_words = Input([out_len], name='target_words', batch_size=batch_size)
    batch_size = str2val(comments, 'batchsize', int, 1)

    input_words = tf.keras.layers.Input([None, n_in], name='input_spikes', batch_size=batch_size)
    output_words = tf.keras.layers.Input([None], name='target_words', batch_size=batch_size)

    x = input_words

    if not embedding is False:
        # in_emb = Lambda(lambda z: tf.math.argmax(z, axis=-1), name='Argmax')(x)
        if x.shape[-1] == 1:
            in_emb = Lambda(lambda z: tf.squeeze(z, axis=-1), name='Squeeze')(x)
        else:
            in_emb = Lambda(lambda z: tf.math.argmax(z, axis=-1), name='Argmax')(x)

        # rnn_input = [emb(in_emb) for emb in embs]
        rnn_input = []
        for emb in embs:
            e = emb(in_emb)
            if 'transformer' in comments or 'addrestrellis' in comments:
                e = Resizing1D(in_len, n_neurons)(e)
            rnn_input.append(e)

    else:
        rnn_input = [x]  # [input_scaling * x]

    skip_input = rnn_input
    output = None

    if 'shared' in comments:
        extper = Expert(0, 0, stateful, task_name, net_name, n_neurons, tau=tau, initializer=initializer,
                        tau_adaptation=tau_adaptation, n_out=n_out, comments=comments, init_states=initial_state)
        expert = lambda i, j, c, n, init_s: extper
    else:
        expert = lambda i, j, c, n, init_s: \
            Expert(i, j, stateful, task_name, net_name, n_neurons=n, tau=tau,
                   initializer=initializer, tau_adaptation=tau_adaptation, n_out=n_out,
                   comments=c, initial_state=init_s)

    if 'sharedff' in comments:
        tffshared = TransformerFF(n_neurons, 2, comments)
        tff = lambda n_neurons, dilation, comments: tffshared
    else:
        tff = lambda n_neurons, dilation, comments: TransformerFF(n_neurons, dilation, comments)

    all_outputs = []

    if 'appendemb' in comments:
        all_outputs.append(*rnn_input)

    if isinstance(stack, str):
        stack = [int(s) for s in stack.split(':')]
    elif isinstance(stack, int):
        stack = [n_neurons for _ in range(stack)]

    all_states = []
    all_input_states = []
    n_states = 4 if 'LSNN' in net_name else 2
    for i, layer_width in enumerate(stack):
        skip_input = [skip_input] if not isinstance(skip_input, list) else skip_input
        os_e = []
        for j in range(n_experts):

            if 'experts:' in comments and 'expertemb' in comments and i == 0:
                in_expert = skip_input[j]
            else:
                in_expert = skip_input[0]

            if 'concrestrellis' in comments and i == 0:
                expanded_input = Lambda(lambda x: tf.concat([x, tf.zeros_like(x)], axis=-1))(in_expert)
                # output_cell = Concatenate(axis=-1)([output_cell, in_expert])
                # all_outputs.append(expanded_input)
            else:
                expanded_input = in_expert

            expanded_input = Dropout(drate, name=f'dropout_{i}')(expanded_input)

            if i == 0:
                if not embedding is False:
                    nin = emb.embed_dim
                else:
                    nin = n_in
            else:
                nin = stack[i - 1]

            c = str2val(comments, 'nin', replace=nin)
            # print(i, layer_width)

            if not initial_state is None:
                initial_state = tuple([
                    Input([layer_width, ], name=f'state_{i}_{si}')
                    for si in range(n_states)
                ])
            output_cell, states = expert(i, j, c, n=layer_width, init_s=initial_state)([expanded_input, output_words])

            if not initial_state is None:
                all_input_states.extend(initial_state)
                all_states.extend(states)

            all_outputs.append(output_cell)

            output_cell, all_outputs = layers_combination(
                output_cell, all_outputs, in_expert, comments, i, layer_width, tff, drate
            )

            os_e.append(output_cell)

        if len(os_e) > 1:
            output_cell = Concatenate(axis=-1)(os_e)
        else:
            output_cell = os_e[0]

        if 'skipGraves' in comments:
            # pass
            if len(os_e) > 1 and i == 0:
                rnn_input = rnn_input * len(os_e)
                rnn_input = [Concatenate(axis=-1)(rnn_input)]
            skip_input = Add()([*rnn_input, output_cell])

        elif 'concatinputs' in comments:
            skip_input = Concatenate(axis=-1)([*rnn_input, output_cell])
        else:
            skip_input = output_cell

        if 'outputGraves' in comments:
            output = output_cell if output is None else Add()([output, output_cell])

        elif 'concatoutputs' in comments:
            output = output_cell if output is None else Concatenate(axis=-1)([output, output_cell])

        else:
            output = output_cell

    if 'skipinout' in comments:
        output = Add()([output, rnn_input])

    if 'nonlinreadout' in comments:
        convread = Conv1D(64, int(out_len / 10), activation='relu', padding='causal', kernel_initializer=initializer)
        readout = Dense(n_out, name='decoder', kernel_initializer=initializer)
        ro = readout(convread(output))

    elif 'spike2linreadout' in comments:
        convread_1 = Conv1D(32, int(out_len / 10), padding='causal', kernel_initializer=initializer)
        spikes = Lambda(lambda x: models.ChoosePseudoHeaviside(x, 1., comments))
        convread_2 = Conv1D(32, int(out_len / 10), dilation_rate=3, padding='causal', kernel_initializer=initializer)
        readout = Dense(n_out, name='decoder', kernel_initializer=initializer)
        ro = readout(spikes(convread_2(spikes(convread_1(output)))))

    elif '2linreadout' in comments:
        convread_1 = Conv1D(32, int(out_len / 10), padding='causal', kernel_initializer=initializer)
        convread_2 = Conv1D(32, int(out_len / 10), dilation_rate=3, padding='causal', kernel_initializer=initializer)
        readout = Dense(n_out, name='decoder', kernel_initializer=initializer)
        ro = readout(convread_2(convread_1(output)))

    elif 'embproj' in comments:
        ro = emb(output, mode='projection')

    elif 'convembproj' in comments:
        convread_1 = Conv1D(n_neurons, int(out_len / 10), padding='causal', kernel_initializer=initializer)
        ro = emb(convread_1(output), mode='projection')

    elif 'conv2embproj' in comments:
        convread_1 = Conv1D(n_neurons, int(out_len / 10), padding='causal', kernel_initializer=initializer)
        convread_2 = Conv1D(n_neurons, int(out_len / 10), dilation_rate=3, padding='causal',
                            kernel_initializer=initializer)
        ro = emb(convread_2(convread_1(output)), mode='projection')

    elif 'simplereadout' in comments:
        readout = Dense(n_out, name='decoder', kernel_initializer=initializer)
        ro = readout(output)

    else:
        readout = Dense(n_out, name='decoder', kernel_initializer=initializer)
        ro = readout(output)

    output_net = ro

    if 'contrastive' in comments:
        output_net = ContrastiveLossLayer(string_config=comments)([output_words, output_net])

    loss = str2val(comments, 'loss', output_type=str, default=loss)
    output_net = AddLossLayer(loss=loss)([output_words, output_net])
    output_net = AddMetricsLayer(metrics=metrics)([output_words, output_net])
    output_net = Lambda(lambda z: z, name='output_net')(output_net)

    # train model
    if initial_state is None:
        # train_model = modifiedModel([input_words, output_words], output_net, name=net_name)
        train_model = tf.keras.models.Model([input_words, output_words], output_net, name=net_name)
    else:
        train_model = tf.keras.models.Model(
            [input_words, output_words] + all_input_states,
            [output_net] + all_states, name=net_name
        )
    exclude_from_weight_decay = ['decoder'] if 'dontdecdec' in comments else []

    optimizer_name = str2val(comments, 'optimizer', output_type=str, default=optimizer_name)
    lr_schedule = str2val(comments, 'lrs', output_type=str, default=lr_schedule)
    optimizer = get_optimizer(optimizer_name=optimizer_name, lr_schedule=lr_schedule,
                              total_steps=final_epochs, lr=lr, weight_decay=weight_decay,
                              clipnorm=clipnorm, exclude_from_weight_decay=exclude_from_weight_decay)
    # train_model.compile(optimizer=optimizer, loss=lambda x, y: 0.)
    train_model.compile(optimizer=optimizer, loss=None, run_eagerly=True)

    return train_model
