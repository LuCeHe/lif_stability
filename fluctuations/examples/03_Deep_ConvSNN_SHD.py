# %% [markdown]
# # Train a deep, recurrent convolutional SNN on the SHD dataset
#
# In this notebook, we demonstrate the training of a 3-layer convolutional SNN with recurrent connections in each hidden layer on the [SHD dataset](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).
#
# We will introduce the use of the `layer` module to initialize feed-forward and recurrent connections at the same time, from the same target parameter $\sigma_U$.

# %%


# %%
# First, imports
import os

import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

import sg_design_lif.fluctuations.stork.datasets
from sg_design_lif.fluctuations.stork.datasets import HDF5Dataset, DatasetView

from sg_design_lif.fluctuations.stork.models import RecurrentSpikingModel
from sg_design_lif.fluctuations.stork.nodes import InputGroup, ReadoutGroup, LIFGroup
from sg_design_lif.fluctuations.stork.connections import Connection
from sg_design_lif.fluctuations.stork.generators import StandardGenerator
from sg_design_lif.fluctuations.stork.initializers import FluctuationDrivenCenteredNormalInitializer, DistInitializer
from sg_design_lif.fluctuations.stork.layers import ConvLayer

import sg_design_lif.fluctuations.stork as stork


def main():
    # %% [markdown]
    # ## Load Dataset
    #
    # ***To locally run this notebook on your system, download the SHD dataset from: [https://zenkelab.org/datasets/](https://zenkelab.org/datasets/).***
    # *We need 'shd_train.h5' and 'shd_test.h5'. Move the downloaded files into a folder `data/datasets/hdspikes` in this repo, or change the `datadir` variable below.

    # %%
    FILENAME = os.path.realpath(__file__)
    CDIR = os.path.dirname(FILENAME)
    DATA = os.path.abspath(os.path.join(CDIR, "..", "..", "..", "data"))
    datadir = os.path.join(DATA, "zenke_datasets", "hdspikes")
    os.makedirs(datadir, exist_ok=True)
    # datadir = "../data/datasets/hdspikes"

    # %% [markdown]
    # #### Specifying dataset parameters

    # %%
    nb_inputs = 700
    duration = 0.7
    time_step = dt = 2e-3
    nb_time_steps = int(duration / time_step)
    time_scale = 1
    unit_scale = 1
    validation_split = 0.9

    gen_kwargs = dict(
        nb_steps=nb_time_steps,
        time_scale=time_scale / time_step,
        unit_scale=unit_scale,
        nb_units=nb_inputs,
        preload=True,
        precompute_dense=False,
        unit_permutation=None
    )

    # %% [markdown]
    # #### Load and split dataset into train / validation / test

    # %%
    train_dataset = HDF5Dataset(os.path.join(datadir, "shd_train.h5"), **gen_kwargs)

    # Split into train and validation set
    mother_dataset = train_dataset
    elements = np.arange(len(mother_dataset))
    np.random.shuffle(elements)
    split = int(validation_split * len(mother_dataset))
    valid_dataset = DatasetView(mother_dataset, elements[split:])
    train_dataset = DatasetView(mother_dataset, elements[:split])

    test_dataset = HDF5Dataset(os.path.join(datadir, "shd_test.h5"), **gen_kwargs)

    # %% [markdown]
    # ## Set up the model

    # %%
    # Model Parameters
    # # # # # # # # # # #

    beta = 20
    nb_hidden_layers = 3
    nb_classes = 20
    nb_filters = [16, 32, 64]  # Number of features per layer

    kernel_size = [21, 7, 7]  # Convolutional operation parameters
    stride = [10, 3, 3]
    padding = [0, 0, 0]

    recurrent_kwargs = {'kernel_size': 5,
                        'stride': 1,
                        'padding': 2}

    # Neuron Parameters
    # # # # # # # # # # #

    neuron_group = LIFGroup
    tau_mem = 20e-3
    tau_syn = 10e-3
    tau_readout = duration

    # Training parameters
    # # # # # # # # # # #

    batch_size = 400 if torch.cuda.is_available() else 8
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    dtype = torch.float
    lr = 5e-3
    nb_epochs = 200

    # %% [markdown]
    # #### SuperSpike and loss function setup

    # %%

    act_fn = stork.activations.SuperSpike
    act_fn.beta = beta

    loss_stack = stork.loss_stacks.MaxOverTimeCrossEntropy()

    # %% [markdown]
    # #### Optimizer setup

    # %%
    opt = stork.optimizers.SMORMS3
    generator = StandardGenerator(nb_workers=4)

    # %% [markdown]
    # #### Regularizer setup

    # %%
    # Define regularizer parameters (set regularizer strenght to 0 if you don't want to use them)
    upperBoundL2Strength = 0.01
    upperBoundL2Threshold = 7  # Regularizes spikecount: 7 spikes ~ 10 Hz in 700ms simulation time

    # Define regularizer list
    regs = []

    regUB = stork.regularizers.UpperBoundL2(upperBoundL2Strength,
                                            threshold=upperBoundL2Threshold,
                                            dims=[-2, -1])
    regs.append(regUB)

    # %% [markdown]
    # #### Initializer setup
    # We initialize in the fluctuation-driven regime with a target membrane potential standard deviation $\sigma_U=1.0$. Additionally, we set the proportion of membrane potential fluctuations driven by feed-forward inputs to $\alpha=0.9$.

    # %%
    sigma_u = 1.0
    nu = 15.8

    initializer = FluctuationDrivenCenteredNormalInitializer(
        sigma_u=sigma_u,
        nu=nu,
        timestep=dt,
        alpha=0.9
    )

    readout_initializer = DistInitializer(
        dist=torch.distributions.Normal(0, 1),
        scaling='1/sqrt(k)'
    )

    # %% [markdown]
    # #### Assemble the model

    # %%
    model = RecurrentSpikingModel(
        batch_size,
        nb_time_steps,
        nb_inputs,
        device,
        dtype)
    # INPUT LAYER
    # # # # # # # # # # # # # # #
    input_shape = (1, nb_inputs)
    input_group = model.add_group(InputGroup(input_shape))

    # Set input group as upstream of first hidden layer
    upstream_group = input_group

    # HIDDEN LAYERS
    # # # # # # # # # # # # # # #
    neuron_kwargs = {'tau_mem': 20e-3,
                     'tau_syn': 10e-3,
                     'activation': act_fn}

    for layer_idx in range(nb_hidden_layers):
        # Generate Layer name and config
        layer_name = str('ConvLayer') + ' ' + str(layer_idx)

        # Make layer
        layer = ConvLayer(name=layer_name,
                          model=model,
                          input_group=upstream_group,
                          kernel_size=kernel_size[layer_idx],
                          stride=stride[layer_idx],
                          padding=padding[layer_idx],
                          nb_filters=nb_filters[layer_idx],
                          recurrent=True,
                          neuron_class=neuron_group,
                          neuron_kwargs=neuron_kwargs,
                          recurrent_connection_kwargs=recurrent_kwargs,
                          regs=regs,
                          )

        # Initialize Parameters
        initializer.initialize(layer)

        # Set output as input to next layer
        upstream_group = layer.output_group

    # READOUT LAYER
    # # # # # # # # # # # # # # #
    readout_group = model.add_group(ReadoutGroup(
        nb_classes,
        tau_mem=tau_readout,
        tau_syn=neuron_kwargs['tau_syn'],
        initial_state=-1e-3))

    readout_connection = model.add_connection(Connection(upstream_group,
                                                         readout_group,
                                                         flatten_input=True))

    # Initialize readout connection
    readout_initializer.initialize(readout_connection)

    # %% [markdown]
    # #### Add monitors for spikes and membrane potential

    # %%
    for i in range(nb_hidden_layers):
        model.add_monitor(stork.monitors.SpikeCountMonitor(model.groups[1 + i]))

    for i in range(nb_hidden_layers):
        model.add_monitor(stork.monitors.StateMonitor(model.groups[1 + i], "out"))

    # %% [markdown]
    # #### Configure model for training

    # %%
    model.configure(input=input_group,
                    output=readout_group,
                    loss_stack=loss_stack,
                    generator=generator,
                    optimizer=opt,
                    optimizer_kwargs=dict(lr=lr),
                    time_step=dt)

    # %% [markdown]
    # ## Monitoring activity before training

    # %%
    # plt.figure(dpi=150)
    # stork.plotting.plot_activity_snapshot(
    #     model,
    #     data=test_dataset,
    #     nb_samples=5,
    #     point_alpha=0.3)

    # %% [markdown]
    # ## Training
    #
    # takes around 85 minutes using a powerful GPU

    # %%
    results = {}

    history = model.fit_validate(
        train_dataset,
        valid_dataset,
        nb_epochs=nb_epochs,
        verbose=False)

    results["train_loss"] = history["loss"].tolist()
    results["train_acc"] = history["acc"].tolist()
    results["valid_loss"] = history["val_loss"].tolist()
    results["valid_acc"] = history["val_acc"].tolist()

    # %% [markdown]
    # ## Test

    # %%
    scores = model.evaluate(test_dataset).tolist()
    results["test_loss"], _, results["test_acc"] = scores

    # %% [markdown]
    # #### Visualize performance

    # %%
    fig, ax = plt.subplots(2, 2, figsize=(5, 3), dpi=150)

    for i, n in enumerate(["train_loss", "train_acc", "valid_loss", "valid_acc"]):

        if i < 2:
            a = ax[0][i]
        else:
            a = ax[1][i - 2]

        a.plot(results[n], color="black")
        a.set_xlabel("Epochs")
        a.set_ylabel(n)

    ax[0, 1].set_ylim(0, 1)
    ax[1, 1].set_ylim(0, 1)

    sns.despine()
    plt.tight_layout()

    print("Test loss: ", results["test_loss"])
    print("Test acc.: ", results["test_acc"])

    print("\nValidation loss: ", results["valid_loss"][-1])
    print("Validation acc.: ", results["valid_acc"][-1])

    # %% [markdown]
    # #### Snapshot after training

    # %%
    # plt.figure(dpi=150)
    # stork.plotting.plot_activity_snapshot(
    #     model,
    #     data=test_dataset,
    #     nb_samples=5,
    #     point_alpha=0.3)

    # %%


if __name__ == '__main__':
    main()
