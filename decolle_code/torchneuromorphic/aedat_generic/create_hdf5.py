#!/bin/python
#-----------------------------------------------------------------------------
# File Name : create_hdf5.py
# Author: Emre Neftci
#
# Creation Date : Tue Nov  5 13:15:54 2019
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import numpy as np
from tqdm import tqdm
import scipy.misc
import h5py
import glob
import torch.utils.data
from ..events_timeslices import *
from ..utils import *
import os

def gather_aedat(directory, filename_prefix = ''):
    if not os.path.isdir(directory):
        raise FileNotFoundError("aedat files not found, looked at: {}".format(directory))
    import glob
    fns = []
    search_mask = directory+'/'+filename_prefix+'*.aedat'
    glob_out = glob.glob(search_mask)
    if len(glob_out)>0:
        fns+=glob_out
    return fns


def create_events_hdf5(directory, hdf5_filename):
    fns = gather_aedat(directory)

    keys = []
    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()

        key = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        for file_d in tqdm(fns):
            istrain = file_d in fns
            data, labels_starttime = aedat_to_events(file_d)
            tms = data[:,0]
            ads = data[:,1:]
            #lbls = labels_starttime[:,0]
            out = []

            keys.append(key)
            s_ = get_slice(tms, ads, start_tms[i], end_tms[i])
            times = s_[0]
            addrs = s_[1]
            metas.append({}) 
            subgrp = data_grp.create_group(str(key))
            tm_dset = subgrp.create_dataset('times' , data=times, dtype=np.uint32)
            ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype=np.uint8)
            #lbl_dset= subgrp.create_dataset('labels', data=lbls[i]-1, dtype=np.uint8)
            subgrp.attrs['meta_info']= str(metas[-1])
            key += 1
        extra_grp.create_dataset('keys', data=train_keys)
        extra_grp.attrs['N'] = len(keys) 
            


if __name__=="__main__":
    out = gather_aedat('./')



