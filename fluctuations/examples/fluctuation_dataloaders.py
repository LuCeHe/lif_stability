import os
import numpy as np
import torch
import torchvision.transforms as tvtf
import torchvision.datasets as tvds
import tonic
import logging

logger = logging.getLogger(__name__)

from lif_stability.fluctuations.stork.datasets import HDF5Dataset, DatasetView

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATA = os.path.abspath(os.path.join(CDIR, "..", "..", "..", "data", "zenke_datasets"))

datasets_available = ['cifar10', 'dvs', 'shd']


def _get_cifar10_dataset(train=True, valid=True, test=True):
    datadir = os.path.join(DATA, "cifar10")
    os.makedirs(datadir, exist_ok=True)

    target_size = 32  # downscale to 32x32
    input_shape = [3, target_size, target_size]  # in the article it seems 32x32
    duration = .5  # 1 second
    time_step = dt = 2e-3
    nb_time_steps = int(duration / time_step)

    valid_split = 0.1
    logger.info("Loading CIFAR10")

    all_transforms = tvtf.Compose([
        tvtf.ToTensor(),
        tvtf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        tvtf.Lambda(lambda tensor: tensor.reshape((1, 3, 32, 32))),
        tvtf.Lambda(lambda tensor: tensor.expand((nb_time_steps, -1, -1, -1))),
    ])

    test_transforms = all_transforms
    train_transforms = all_transforms

    if train:
        ds_train = tvds.CIFAR10(
            datadir,
            train=True,
            download=True,
            transform=train_transforms
        )
        logger.info("Generated {} training data".format(len(ds_train)))

    else:
        ds_train = False

    if valid:
        split_lengths = [int(len(ds_train) * (1 - valid_split)), int(len(ds_train) * valid_split)]
        ds_train, ds_valid = torch.utils.data.dataset.random_split(ds_train, split_lengths)
        logger.info("Generated {} validation data".format(len(ds_valid)))

    else:
        ds_valid = False

    if test:
        ds_test = tvds.CIFAR10(
            datadir,
            train=False,
            download=True,
            transform=test_transforms
        )
        logger.info("Generated {} testing data".format(len(ds_test)))

    else:
        ds_test = False

    data_config = {
        'duration': duration,
        'dt': dt,
        'nb_time_steps': nb_time_steps,
        'target_size': target_size,
        'input_shape': input_shape,
        'nb_inputs': input_shape
    }

    return {"train": ds_train, "valid": ds_valid, "test": ds_test, "data_config": data_config}


def _get_DVSgestures_dataset(train=True, valid=True, test=True):
    datadir = os.path.join(DATA, "dvs")
    os.makedirs(datadir, exist_ok=True)

    target_size = 32  # downscale to 32x32
    input_shape = [2, target_size, target_size]
    nb_classes = 11
    valid_split = 0.1
    duration = 1.  # 1 second
    time_step = dt = 2e-3
    nb_time_steps = int(duration / time_step)
    dropevent_p = 0.5
    bool_spiketrain = False  # Whether to call a boolean operation on the spiketrain
    # (Prevents spikes with an amplitude >1)

    # Transforms

    # Drop random events
    tf_dropevent = tonic.transforms.DropEvent(p=dropevent_p)

    # Convert to milliseconds
    tf_convert_to_ms = tonic.transforms.Downsample(time_factor=1e-3,
                                                   spatial_factor=target_size / 128)

    # Assemble frames according to timestep
    tf_frame = tonic.transforms.ToFrame(sensor_size=(target_size, target_size, 2),
                                        time_window=time_step * 1000)

    # CUSTOM TRANSFORMS

    class ToTensorTransform:
        """ Custom ToTensor transform that supports 4D arrays"""

        def __init__(self, bool_spiketrain=False):
            self.bool_spiketrain = bool_spiketrain

        def __call__(self, x):
            if self.bool_spiketrain:
                return torch.as_tensor(x).bool().float()
            else:
                return torch.as_tensor(x).float()

    tf_tensor = ToTensorTransform(bool_spiketrain)

    class TimeCropTransform:
        """ Custom transform that randomly crops the time dimension"""

        def __init__(self, timesteps):
            self.timesteps = int(timesteps)

        def __call__(self, x):
            start = np.random.randint(0, high=x.shape[0] - self.timesteps)
            # print(x[start:start + self.timesteps, :, :, :].shape) torch.Size([500, 2, 32, 32])
            return x[start:start + self.timesteps, :, :, :]

    tf_timecrop = TimeCropTransform(nb_time_steps)

    all_transforms = tonic.transforms.Compose([tf_dropevent,
                                               tf_convert_to_ms,
                                               tf_frame,
                                               tf_tensor,
                                               tf_timecrop])

    if train:
        ds_train = tonic.datasets.DVSGesture(datadir,
                                             train=True,
                                             transform=all_transforms)

        logger.info("Generated {} training data".format(len(ds_train)))
        print("Generated {} training data".format(len(ds_train)))

    else:
        ds_train = False

    if valid:
        new_train_length = int(len(ds_train) * (1 - valid_split))
        val_length = len(ds_train) - new_train_length
        split_lengths = [new_train_length, val_length]
        ds_train, ds_valid = torch.utils.data.dataset.random_split(ds_train, split_lengths)
        logger.info("Generated {} validation data".format(len(ds_valid)))

    else:
        ds_valid = False

    if test:
        ds_test = tonic.datasets.DVSGesture(datadir,
                                            train=False,
                                            transform=all_transforms)

        logger.info("Generated {} testing data".format(len(ds_test)))

    else:
        ds_test = False

    data_config = {
        'duration': duration,
        'dt': dt,
        'nb_time_steps': nb_time_steps,
        'target_size': target_size,
        'input_shape': input_shape,
        'nb_inputs': input_shape
    }

    return {"train": ds_train, "valid": ds_valid, "test": ds_test, "data_config": data_config}


def _get_shd_dataset(train=True, valid=True, test=True):
    # ***To locally run this notebook on your system, download the SHD dataset from: [https://zenkelab.org/datasets/](https://zenkelab.org/datasets/).***
    # *We need 'shd_train.h5' and 'shd_test.h5'. Move the downloaded files into a folder `data/datasets/hdspikes` in this repo, or change the `datadir` variable below.
    datadir = os.path.join(DATA, "hdspikes")
    os.makedirs(datadir, exist_ok=True)

    # #### Specifying dataset parameters
    nb_inputs = 700
    duration = 0.7
    time_step = dt = 2e-3
    nb_time_steps = int(duration / time_step)
    time_scale = 1
    unit_scale = 1
    validation_split = 0.9
    input_shape = (1, nb_inputs)

    gen_kwargs = dict(
        nb_steps=nb_time_steps,
        time_scale=time_scale / time_step,
        unit_scale=unit_scale,
        nb_units=nb_inputs,
        preload=True,
        precompute_dense=False,
        unit_permutation=None
    )

    # #### Load and split dataset into train / validation / test
    train_dataset = HDF5Dataset(os.path.join(datadir, "shd_train.h5"), **gen_kwargs)

    # Split into train and validation set
    mother_dataset = train_dataset
    elements = np.arange(len(mother_dataset))
    np.random.shuffle(elements)
    split = int(validation_split * len(mother_dataset))
    valid_dataset = DatasetView(mother_dataset, elements[split:])
    train_dataset = DatasetView(mother_dataset, elements[:split])

    test_dataset = HDF5Dataset(os.path.join(datadir, "shd_test.h5"), **gen_kwargs)

    data_config = {
        'duration': duration,
        'dt': dt,
        'nb_time_steps': nb_time_steps,
        'input_shape': input_shape,
        'nb_inputs': nb_inputs,
    }

    return {"train": train_dataset, "valid": valid_dataset, "test": test_dataset, "data_config": data_config}


def load_dataset(name):
    assert name in datasets_available

    if name == "cifar10":
        return _get_cifar10_dataset()
    elif name == "dvs":
        return _get_DVSgestures_dataset()
    elif name == "shd":
        return _get_shd_dataset()
    else:
        raise NotImplementedError
