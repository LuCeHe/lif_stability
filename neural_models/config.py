from GenericTools.keras_tools.esoteric_tasks.time_task_redirection import language_tasks


def default_config(stack, batch_size, embedding, n_neurons, lr, task_name, net_name):
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
        if task_name in language_tasks:
            lr = 3.16e-5
        elif task_name in ['heidelberg', 'lca']:
            lr = 3.16e-4
        elif 'mnist' in task_name:
            lr = 3.16e-4
        else:
            raise NotImplementedError

    if batch_size is None:
        if task_name in language_tasks:
            batch_size = 32
        elif task_name in ['heidelberg', 'lca']:
            batch_size = 256
        elif 'mnist' in task_name:
            batch_size = 256
        else:
            raise NotImplementedError

    if stack is None:
        if 'mnist' in task_name or task_name in ['heidelberg', 'lca']:
            stack = 2
        elif task_name in language_tasks:
            stack = '1700:300'
            embedding = 'learned:None:None:300'
            # if not lsc:
            #     stack = '1700:300'
            #     embedding = 'learned:None:None:300'
            # else:
            #     stack = '1000:300'
            #     embedding = 'learned:None:None:300'
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
    stack = '700:300' if ('LSTM' in net_name and task_name == 'wordptb') else stack

    return stack, batch_size, embedding, n_neurons, lr
