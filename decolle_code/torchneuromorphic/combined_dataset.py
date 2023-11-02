#!/bin/python
#-----------------------------------------------------------------------------
# File Name : double_nmnist_dataloaders.py
# Author: Emre Neftci
#
# Creation Date : Tue 26 May 2020 02:25:41 PM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : Apache License, Version 2.0
#----------------------------------------------------------------------------- 

import struct
import time
import numpy as np
import scipy.misc
import h5py
import torch.utils.data
from .neuromorphic_dataset import NeuromorphicDataset 
from .events_timeslices import *
from .transforms import *

import os


class CombinedDataset(NeuromorphicDataset):
    def __init__(self,
                base_datasets,
                train = True,
                transform = None,
                target_transform = None,
                chunk_size = 500):

        self.download_and_create   = False
        super(CombinedDataset, self).__init__(
        root = None,
        transform=transform,
        target_transform=target_transform )

        self.base_datasets = d = base_datasets
        self.nd = len(base_datasets)
        self.lens = lens = [len(d) for d in base_datasets]
        self.lensmd = lensmd = np.cumsum(lens[:-1], dtype=int)
    
        self.keys = lambda d: [d//l for l in lensmd]+[d%lens[0]]
        self.locks = lambda ts: sum(ts*np.array([d[i].nclasses for i in range(1,len(d))]+[1]))
 
    def __len__(self):
        return np.prod(self.lens,0)


    def __getitem__(self, key):
        #Important to open and close in getitem to enable num_workers>0
        keys_ = self.keys(key)
        d = self.base_datasets
        datas, targets = list(zip(*[d[i][k] for i,k in enumerate(keys_)] ))
        data = sum(datas)
        target = self.locks([t[-1].argmax(0) for i,t in enumerate(targets)])
        
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

