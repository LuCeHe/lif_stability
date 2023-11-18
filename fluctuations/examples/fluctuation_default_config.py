def default_shd(deep=False):
    config = {
        'beta': 20,
        'nb_conv_blocks': 1,
        'nb_hidden_layers': 3,
        'nb_classes': 20,
        'nb_filters': [16, 32, 64],  # Number of features per layer
        'kernel_size': [21, 7, 7],  # Convolutional operation parameters
        'stride': [10, 3, 3],
        'padding': 2,
        'recurrent_kwargs': {'kernel_size': 5, 'stride': 1, 'padding': 2},
        'maxpool_kernel_size': 2,
        'dropout_p': 0.0,
        'batch_size': 400,
        'lr': 5e-3,
        'epochs': 200,
        'upperBoundL2Threshold': 7,
        'nu': 15.8,
    }
    if deep:
        config['nb_hidden_layers'] = 7
        config['nb_filters'] = [16, 32, 64, 64, 64, 64, 64]
        config['kernel_size'] = [7, 5, 5, 5, 5, 5, 5]
        config['stride'] = [3, 2, 2, 2, 2, 2, 2]
        config['padding'] = 2
        config['batch_size'] = 100

    return config


def default_cifar10(deep=False):
    config =  {
        'beta': 20,
        'nb_conv_blocks': 1,
        'nb_hidden_layers': 2,
        'nb_classes': 10,
        'nb_filters': [32, 32],
        'kernel_size': 3,
        'stride': 1,
        'padding': 2,
        'maxpool_kernel_size': 2,
        'recurrent_kwargs': {},
        'dropout_p': 0.0,
        'batch_size': 16,  # 128
        'lr': 5e-3,
        'epochs': 100,
        'upperBoundL2Threshold': 10,
        'nu': 9.2,
    }
    if deep:
        config['nb_conv_blocks'] = 2
        config['nb_filters'] = [32, 32, 64, 64]
        # config['batch_size'] = 128

    return config


def default_dvs(deep=False):
    config =  {
        'beta': 20,
        'nb_conv_blocks': 3,
        'nb_hidden_layers': 2,
        'nb_classes': 11,
        'nb_filters': [32, 32, 64, 64, 128, 128],
        'kernel_size': 3,
        'stride': 1,
        'padding': 2,
        'maxpool_kernel_size': 2,
        'recurrent_kwargs': {},
        'dropout_p': 0.0,
        'batch_size': 8,  # 16
        'lr': 5e-3,
        'epochs': 100,
        'upperBoundL2Threshold': 10,
        'nu': 9.2,
    }

    if deep:
        config['nb_conv_blocks'] = 4
        config['nb_filters'] = [32, 32, 64, 64, 128, 128, 128, 128]
        config['batch_size'] = 6


    return config


def default_config(name, deep=False):
    if name == "cifar10":
        config = default_cifar10(deep)
        return config
    elif name == "dvs":
        config = default_dvs(deep)
        return config
    elif name == "shd":
        config = default_shd(deep)
        return config
    else:
        raise ValueError("Unknown dataset name: {}".format(name))
