#!/bin/python
# -----------------------------------------------------------------------------
# File Name : train_lenet_decolle
# Author: Emre Neftci
#
# Creation Date : Sept 2. 2019
# Last Modified :
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
# -----------------------------------------------------------------------------
import json, shutil, time

from pyaromatics.stay_organized.utils import NumpyEncoder, str2val
from sg_design_lif.decolle_code.decolle.base_model import LIFLayerPlus
from sg_design_lif.decolle_code.torchneuromorphic.nmnist import nmnist_dataloaders
from sg_design_lif.decolle_code.torchneuromorphic.dvs_gestures import dvsgestures_dataloaders
from sg_design_lif.decolle_code.decolle.lenet_decolle_model import LenetDECOLLE, DECOLLELoss, LIFLayerVariableTau, \
    LIFLayer
from sg_design_lif.decolle_code.decolle.utils import parse_args, train, test, accuracy, save_checkpoint, \
    load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot
import datetime, os, socket, tqdm
import numpy as np
import torch
import importlib
import os

CDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.abspath(os.path.join(CDIR, '..', '..', '..', 'data'))
np.set_printoptions(precision=4)


def main(args):
    stop_time = time.perf_counter() + args.stop_time - 30 * 60
    starting_epoch = 0
    early_stop = 12
    device = args.device if torch.cuda.is_available() else 'cpu'

    # get name of this file with code that is windows and linux compatible
    name = os.path.split(__file__)[1].split('.')[0]
    args.file_name = name
    results = {}

    params, writer, dirs = prepare_experiment(name=name, args=args)
    log_dir = dirs['log_dir']
    checkpoint_dir = dirs['checkpoint_dir']

    if 'test' in args.comments:
        params['Nhid'] = [11, 17, 4]
        params['Mhid'] = [11]
        params['batch_size'] = 16
        params['num_layers'] = 3
        params['num_conv_layers'] = 3

    # print args with json
    args.__dict__.update(dirs)
    print(json.dumps(args.__dict__, indent=2))
    print(json.dumps(params, indent=2))
    results.update(params)

    if 'nmnist' in params['dataset']:
        dataset = nmnist_dataloaders
        create_data = dataset.create_dataloader
        root = os.path.join(DATADIR, 'nmnist', 'n_mnist.hdf5')
    else:
        dataset = dvsgestures_dataloaders
        create_data = dataset.create_dataloader
        root = os.path.join(DATADIR, 'dvsgesture', 'dvsgestures.hdf5')
        # root = os.path.join(DATADIR, 'dvs', 'dvsgestures')

    ## Load Data
    gen_train, gen_val, gen_test = create_data(
        root=root,
        chunk_size_train=params['chunk_size_train'],
        chunk_size_test=params['chunk_size_test'],
        batch_size=params['batch_size'],
        dt=params['deltat'],
        num_workers=params['num_dl_workers']
    )

    data_batch, target_batch = next(iter(gen_train))
    data_batch = torch.Tensor(data_batch).to(device)
    target_batch = torch.Tensor(target_batch).to(device)

    # d, t = next(iter(gen_train))
    input_shape = data_batch.shape[-3:]

    # Backward compatibility
    if 'dropout' not in params.keys():
        params['dropout'] = [.5]

    curve_name = str2val(args.comments, 'sgcurve', str, default='dfastsigmoid')
    if 'condIV' in args.comments:
        print('Using condition IV')
        lif_layer_type = lambda *args, **kwargs: LIFLayerPlus(rule='IV', curve_name=curve_name, *args, **kwargs)
    elif 'condIII' in args.comments:
        print('Using condition III')
        lif_layer_type = lambda *args, **kwargs: LIFLayerPlus(rule='III', curve_name=curve_name, *args, **kwargs)
    elif 'condI_III_IV' in args.comments:
        print('Using condition I/III/IV')
        lif_layer_type = lambda *args, **kwargs: LIFLayerPlus(rule='I_III_IV', curve_name=curve_name, *args, **kwargs)
    elif 'condI_IV' in args.comments:
        print('Using condition I/IV')
        lif_layer_type = lambda *args, **kwargs: LIFLayerPlus(rule='I_IV', curve_name=curve_name, *args, **kwargs)
    elif 'condI' in args.comments:
        print('Using condition I')
        lif_layer_type = lambda *args, **kwargs: LIFLayerPlus(rule='I', curve_name=curve_name, *args, **kwargs)
    else:
        lif_layer_type = lambda *args, **kwargs: LIFLayerPlus(rule='0', curve_name=curve_name, *args, **kwargs)
        # lif_layer_type = LIFLayer

    ## Create Model, Optimizer and Loss
    net = LenetDECOLLE(out_channels=params['out_channels'],
                       Nhid=params['Nhid'],
                       Mhid=params['Mhid'],
                       kernel_size=params['kernel_size'],
                       pool_size=params['pool_size'],
                       input_shape=params['input_shape'],
                       alpha=params['alpha'],
                       alpharp=params['alpharp'],
                       dropout=params['dropout'],
                       beta=params['beta'],
                       num_conv_layers=params['num_conv_layers'],
                       num_mlp_layers=params['num_mlp_layers'],
                       lc_ampl=params['lc_ampl'],
                       lif_layer_type=lif_layer_type,
                       method=params['learning_method'],
                       with_output_layer=params['with_output_layer']).to(device)

    if hasattr(params['learning_rate'], '__len__'):
        from sg_design_lif.decolle_code.decolle.utils import MultiOpt

        opts = []
        for i in range(len(params['learning_rate'])):
            opts.append(
                torch.optim.Adamax(net.get_trainable_parameters(i), lr=params['learning_rate'][i],
                                   betas=params['betas']))
        opt = MultiOpt(*opts)
    else:
        opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])

    reg_l = params['reg_l'] if 'reg_l' in params else None

    if 'loss_scope' in params and params['loss_scope'] == 'global':
        loss = [None for i in range(len(net))]
        if net.with_output_layer:
            loss[-1] = cross_entropy_one_hot
        else:
            raise RuntimeError('bptt mode needs output layer')
        decolle_loss = DECOLLELoss(net=net, loss_fn=loss, reg_l=reg_l)
    else:
        loss = [torch.nn.SmoothL1Loss() for i in range(len(net))]
        if net.with_output_layer:
            loss[-1] = cross_entropy_one_hot
        decolle_loss = DECOLLELoss(net=net, loss_fn=loss, reg_l=reg_l)

    ##Initialize
    net.init_parameters(data_batch[:32])

    from sg_design_lif.decolle_code.decolle.init_functions import init_LSUV

    if not 'test' in args.comments:
        init_LSUV(net, data_batch[:32])

    ##Resume if necessary
    if args.resume_from is not None:
        print("Checkpoint directory " + checkpoint_dir)
        if not os.path.exists(checkpoint_dir) and not args.no_save:
            os.makedirs(checkpoint_dir)
        starting_epoch = load_model_from_checkpoint(checkpoint_dir, net, opt)
        print('Learning rate = {}. Resumed from checkpoint'.format(opt.param_groups[-1]['lr']))

    # Printing parameters
    if args.verbose:
        print('Using the following parameters:')
        m = max(len(x) for x in params)
        for k, v in zip(params.keys(), params.values()):
            print('{}{} : {}'.format(k, ' ' * (m - len(k)), v))

    print('\n------Starting training with {} DECOLLE layers-------'.format(len(net)))

    # --------TRAINING LOOP----------
    if not args.no_train:
        train_losses = []
        val_losses = []
        val_accs = []
        val_acc_hist = []
        best_val = np.inf
        best_epoch = -1
        for e in range(starting_epoch, params['num_epochs']):

            interval = e // params['lr_drop_interval']
            lr = opt.param_groups[-1]['lr']
            if interval > 0:
                print('Changing learning rate from {} to {}'.format(lr, opt.param_groups[-1]['lr']))
                opt.param_groups[-1]['lr'] = np.array(params['learning_rate']) / (interval * params['lr_drop_factor'])
            else:
                print('Changing learning rate from {} to {}'.format(lr, opt.param_groups[-1]['lr']))
                opt.param_groups[-1]['lr'] = np.array(params['learning_rate'])

            print('---------------Epoch {}-------------'.format(e))
            if not args.no_save:
                print('---------Saving checkpoint---------')
                save_checkpoint(e, checkpoint_dir, net, opt)

            val_loss, val_acc = test(gen_val, decolle_loss, net, params['burnin_steps'], print_error=True)
            val_acc_hist.append(val_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = e

            # if not args.no_save:
            #     write_stats(e, val_acc, val_loss, writer)
            #     np.save(log_dir + '/test_acc.npy', np.array(val_acc_hist), )

            total_loss, act_rate = train(gen_train, decolle_loss, net, opt, e, params['burnin_steps'],
                                         online_update=params['online_update'])
            train_losses.append(total_loss)

            # if not args.no_save:
            #     for i in range(len(net)):
            #         writer.add_scalar('/act_rate/{0}'.format(i), act_rate[i], e)

            results.update(train_losses=train_losses, test_losses=val_losses, test_accs=val_accs)
            if time.perf_counter() > stop_time:
                print('Time assigned to this job is over, stopping')
                break

            if e - best_epoch > early_stop:
                print('Early stopping')
                break



    test_loss, test_acc = test(gen_test, decolle_loss, net, params['burnin_steps'], print_error=True)
    results.update(test_loss=test_loss, test_acc=test_acc)

    return args, results


if __name__ == '__main__':
    args = parse_args()

    time_start = time.perf_counter()
    args, results = main(args)
    time_elapsed = (time.perf_counter() - time_start)

    results.update(time_elapsed=time_elapsed)
    results.update(hostname=socket.gethostname())

    args = args.__dict__
    for d in [args, results]:
        string_result = json.dumps(d, indent=4, cls=NumpyEncoder)
        var_name = [k for k, v in locals().items() if v is d if not k == 'd'][0]
        print(var_name)

        path = os.path.join(args['log_dir'], var_name + '.txt')
        with open(path, "w") as f:
            f.write(string_result)

    # remove events files
    events = [e for e in os.listdir(args['log_dir']) if 'events' in e]
    for e in events:
        if os.path.exists(os.path.join(args['log_dir'], e)):
            os.remove(os.path.join(args['log_dir'], e))

    # remove checkpoints folder
    if os.path.exists(os.path.join(args['log_dir'], 'checkpoints')):
        shutil.rmtree(os.path.join(args['log_dir'], 'checkpoints'))

    shutil.make_archive(args['log_dir'], 'zip', args['log_dir'])
    print('All done, in ' + str(time_elapsed) + 's')
