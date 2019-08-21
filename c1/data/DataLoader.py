import glob
import numpy as np
import torch
import os
import sys
import pandas as pd
import pydicom

from .mask_functions import *

class DataLoader:
    def __init__(self, train_path, test_path, labels_path, batchsize = 16, im_height = 1024, im_width = 1024, im_chan = 1, create_val = True, create_test = True):
        self.train_glob  = train_path
        self.labels_glob = labels_path
        self.test_glob   = test_path
        self.batchsize   = batchsize
        self.create_val  = create_val
        self.create_test = create_test
        self.fnames_full = sorted(glob.glob(self.train_glob))
        self.tfnames_full= sorted(glob.glob(self.test_glob))
        self.labels_full = pd.read_csv(self.labels_glob, index_col = 'ImageId')
        #self.labels_full = pd.read_csv(self.labels_glob, header = None, index_col = 0)
        self.num_samples = len(self.fnames_full)
        self.num_batches = int(self.num_samples/self.batchsize)
        self.fname_dict  = dict()
        self.tfname_dict = dict()
        self.fname_ids   = list()
        self.tfname_ids  = list()
        self.train_split = list()
        self.val_split   = list()
        self.test_split  = list()
        self.train_bnum  = 0
        self.im_height   = im_height
        self.im_width    = im_width
        self.im_c        = im_chan

    def construct_fnameids(self, mode = "TRAIN"):
        if mode == "TRAIN":
            self.fname_dict = dict()
            fnames     = self.fnames_full
            fname_dict = self.fname_dict
            fname_ids  = self.fname_ids
            num_samples= len(self.fnames_full)
        elif mode == "TEST":
            self.tfname_dict = dict()
            fnames     = self.tfnames_full
            fname_dict = self.tfname_dict
            fname_ids  = self.tfname_ids
            num_samples= len(self.tfnames_full)
        else:
            print("pass TRAIN or TEST for mode")
            exit()

        for _id, fname in enumerate(fnames):
            fname_dict[_id] = fname

        '''
        if mode == "TRAIN":
            self.fname_ids = list(fname_dict)
        else:
            self.tfname_ids = list(fname_dict)
        '''

    def shuffle_data(self):
        np.random.shuffle(self.fname_ids)

    def create_datafolds(self, train_, test_, val_):
        #this must be called after every epoch.
        if (len(self.train_split) > 0):
            del self.train_split[:]
        if len(self.val_split) > 0:
            del self.val_split[:]
        if len(self.test_split) > 0:
            del self.test_split[:]

        self.train_split = [-1] * int(float(self.num_batches) * train_ / 100.)
        self.val_split   = [-1] * int(float(self.num_batches) * val_ / 100.)
        self.test_split  = [-1] * int(float(self.num_batches) * test_ / 100.)

        id_list = [i for i in range(len(self.fnames_full))]
        np.random.shuffle(id_list)

        start_val = 0
        end_val   = (len(self.train_split) * self.batchsize)
        self.train_split = id_list[start_val:end_val]

        start_val = end_val
        end_val   = start_val + (len(self.val_split) * self.batchsize)
        self.val_split   = id_list[start_val:end_val]

        start_val = end_val
        end_val   = start_val + (len(self.test_split) * self.batchsize)
        self.test_split = id_list[start_val:end_val]

    def load_data(self):
        self.construct_fnameids()
        if self.create_val and self.create_test:
            self.create_datafolds(80.0, 10.0, 10.0)
        elif not self.create_val and self.create_test:
            self.create_datafolds(85.0, 15.0, 0.0)
        elif self.create_val and not self.create_test:
            print("pass False for create_val")
        else:
            self.create_datafolds(100.0, 0.0, 0.0)
            self.construct_fnameids(mode = "TEST")
            #we are not going to create a datafold of test data as it is not required.
            #let us write a different routine to pass the test data

    def get_sample_batch(self):
        X_batch =  np.zeros((self.batchsize, self.im_c, self.im_height, self.im_width), dtype = np.uint8)
        Y_batch =  np.zeros((self.batchsize, self.im_c, self.im_height, self.im_width), dtype = np.uint8)
        print(f"Getting batches of X_shape: {X_batch.shape}, Y_shape: {Y_batch.shape}")

        for n, i_fname in enumerate(range(self.batchsize)):
            _id = self.fname_dict[self.train_split[i_fname]]
            data = pydicom.read_file(_id)
            X_batch[n] = np.expand_dims(data.pixel_array, axis = 0)
            test = _id.split('/')[-1][:-4]
            k = self.labels_full.loc[test,1]
            if k != '-1':
                Y_batch[n] = np.expand_dims(rle2mask(self.labels_full.loc[_id.split('/')[-1][:-4],1], 1024, 1024).T, axis = 0)
        return X_batch, Y_batch




    def get_next_batch(self):
        X_batch =  np.zeros((self.batchsize, self.im_c, self.im_height, self.im_width), dtype = np.uint8)
        Y_batch =  np.zeros((self.batchsize, self.im_c, self.im_height, self.im_width), dtype = np.uint8)
        #print(f"Getting batches of X_shape: {X_batch.shape}, Y_shape: {Y_batch.shape}")

        i_start = self.train_bnum * self.batchsize
        i_end   = i_start + self.batchsize
        #print(f"start_ind, end_ind: {i_start}, {i_end}")
        for n, i_fname in enumerate(range(i_start, i_end)):
            _id = self.fname_dict[self.train_split[i_fname]]
            data = pydicom.read_file(_id)
            X_batch[n] = np.expand_dims(data.pixel_array, axis = 0)
            try:
                if '-1' in self.labels_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']:
                    Y_batch[n] = np.zeros((self.im_c, self.im_height, self.im_width))
                else:
                    if type(self.labels_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']) == str:
                        Y_batch[n] = np.expand_dims(rle2mask(self.labels_full.loc[_id.split('/')[-1][:-4],' EncodedPixels'], self.im_height, self.im_width).T, axis=0)
                    else:
                        Y_batch[n] = np.zeros((self.im_c, self.im_height, self.im_width))
                        for x in self.labels_full.loc[_id.split('/')[-1][:-4], ' EncodedPixels']:
                            Y_batch[n] = Y_batch[n] + np.expand_dims(rle2mask(x, self.im_height, self.im_width).T, axis=0)
            except KeyError:
                #print(f"Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")
                Y_batch[n] = np.zeros((self.im_c, self.im_height, self.im_width))
        #print("Loaded batch")
        self.train_bnum += 1
        return X_batch, Y_batch