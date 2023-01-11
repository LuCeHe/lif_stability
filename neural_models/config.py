from GenericTools.keras_tools.esoteric_tasks.time_task_redirection import language_tasks


def default_config(stack, batch_size, embedding, n_neurons, lr, task_name, net_name, setting='LIF'):
    assert setting in ['LIF', 'LSC']

    if n_neurons is None:
        if task_name in language_tasks:
            n_neurons = 1300
        elif task_name in ['heidelberg', 'lca']:
            n_neurons = 256
        elif 'mnist' in task_name:
            n_neurons = 128
        else:
            raise NotImplementedError

    if lr is None:
        if setting == 'LIF':
            if task_name in language_tasks:
                lr = 3.16e-5
            elif task_name in ['heidelberg', 'lca']:
                lr = 3.16e-4
            elif 'mnist' in task_name:
                lr = 3.16e-4
            else:
                raise NotImplementedError

        elif setting == 'LSC':
            if net_name in ['maLSNN', 'maLSNNb']:
                lr = 1e-3
            elif net_name == 'LSTM':

                if task_name == 'wordptb':
                    lr = 3.16e-4
                elif task_name in ['heidelberg', 'lca']:
                    lr = 1e-3
                elif 'mnist' in task_name:
                    lr = 1e-2
                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    if batch_size is None:
        if task_name in language_tasks:
            batch_size = 32
        elif task_name in ['heidelberg', 'lca']:
            batch_size = 128 if setting == 'LIF' else 100
            if stack == 7:
                batch_size = 50
        elif 'mnist' in task_name:
            batch_size = 256
        else:
            raise NotImplementedError

    if stack is None:
        if 'mnist' in task_name or task_name in ['heidelberg', 'lca']:
            stack = 2
        elif task_name in language_tasks:
            stack = '1700:300' if setting == 'LIF' else '1300:300'
            embedding = 'learned:None:None:300'
            if net_name == 'LSTM':
                stack = '700:300' if setting == 'LIF' else '500:300'
        else:
            raise NotImplementedError

    if embedding == None:
        embedding = False

    if task_name == 'heidelberg':
        sLSTM_factor = .37  # 1 / 3
    elif task_name == 'sl_mnist':
        sLSTM_factor = 1 / 3
    else:
        sLSTM_factor = 1 / 3

    n_neurons = n_neurons if not 'LSTM' in net_name else int(n_neurons * sLSTM_factor)

    return stack, batch_size, embedding, n_neurons, lr
