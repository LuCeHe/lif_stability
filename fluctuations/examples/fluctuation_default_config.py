def default_shd():
    return {
        'beta': 20,
        'nb_conv_blocks': 1,
        'nb_hidden_layers': 3,
        'nb_classes': 20,
        'nb_filters': [16, 32, 64],  # Number of features per layer
        'kernel_size': [21, 7, 7],  # Convolutional operation parameters
        'stride': [10, 3, 3],
        'padding': [0, 0, 0],
        'recurrent_kwargs': {'kernel_size': 5, 'stride': 1, 'padding': 2},
        'maxpool_kernel_size': 2,
        'dropout_p': 0.0,
        'batch_size': 400,
        'lr': 5e-3,
        'epochs': 200,
        'upperBoundL2Threshold': 7,
        'nu': 15.8,
    }


def default_cifar10():
    return {
        'beta': 20,
        'nb_conv_blocks': 1,
        'nb_hidden_layers': 2,
        'nb_classes': 11,
        'nb_filters': [16, 32],
        'kernel_size': 3,
        'stride': 1,
        'padding': 2,
        'maxpool_kernel_size': 2,
        'recurrent_kwargs': {},
        'dropout_p': 0.0,
        'batch_size': 8,  # 16
        'lr': 5e-3,
        'epochs': 200,
        'upperBoundL2Threshold': 10,
        'nu': 9.2,
    }


def default_dvs():
    return {
        'beta': 20,
        'nb_conv_blocks': 3,
        'nb_hidden_layers': 2,
        'nb_classes': 11,
        'nb_filters': [16, 32],
        'kernel_size': 3,
        'stride': 1,
        'padding': 2,
        'maxpool_kernel_size': 2,
        'recurrent_kwargs': {},
        'dropout_p': 0.0,
        'batch_size': 8,  # 16
        'lr': 5e-3,
        'epochs': 200,
        'upperBoundL2Threshold': 10,
        'nu': 9.2,
    }


def default_config(name, deep=False):
    if name == "cifar10":
        config = default_cifar10()
        if deep:
            config['nb_conv_blocks'] = 2
        return config
    elif name == "dvs":
        config = default_dvs()
        if deep:
            config['nb_conv_blocks'] = 4
        return config
    elif name == "shd":
        config = default_shd()
        if deep:
            config['nb_hidden_layers'] = 7
        return config
    else:
        raise ValueError("Unknown dataset name: {}".format(name))
