# Weather4cast 2023 Starter Kit
#
# The files from this repository make up the Weather4cast 2023 Starter Kit.
# 
# It builds on and extends the Weather4cast 2022 Starter Kit, the
# original copyright and GPL license notices for which are included
# below.
#
# In line with the provisions of that license, all changes and
# additional code are also released under the GNU General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.

# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
#
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


# from cv2 import CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_COMMIT
import numpy as np
from torch.utils.data import Dataset
import os
import sys
import time
from timeit import default_timer as timer
from random import random

from utils.data_utils import *

# folder to load config file
# CONFIG_PATH = "../"

# VERBOSE = True
VERBOSE = False

"""
Assumptions: 
- the data is already cropped to the right dimensions 
- Data format - [Samples, C, T, W, H] 
"""

class RainData(Dataset):
    def __init__(
        self,
        data_split,
        project_root="",
        data_root="",
        input_product="REFL-BT",
        compute_seq=True,
        output_product="RATE",
        sat_bands=[],
        preprocess_OPERA=None,
        size_target_center=None,
        full_opera_context=None,
        preprocess_HRIT=None,
        path_to_sample_ids="",
        len_seq_in=4,
        len_seq_predict=32,
        regions=["boxi_0015"],
        regions_def={},
        generate_samples=False,
        latlon_path="",
        altitudes_path="",
        splits_path=None,
        swap_time_ch=False,
        years=None,
        padding=None,
        crop_edge=None,
        data_cleansing=False,
        shuffle=False,
        collapse_time=False,
        size=126,
        **kwargs
    ):
        start = timer()
        # Data Dimensions
        self.len_seq_in = len_seq_in
        self.len_seq_predict = len_seq_predict
        self.channel_dim = 1  # where to concat channels in structure

        # type of data & processing variables
        self.sat_bands = sat_bands
        self.regions = regions
        self.input_product = input_product
        self.output_product = output_product
        self.preprocess_target = preprocess_OPERA
        self.size_target_center = size_target_center
        self.full_opera_context = full_opera_context
        self.crop = int(
            (self.full_opera_context - self.size_target_center) / 2
        )  # calculate centre of image to begin crop
        self.preprocess_input = preprocess_HRIT
        self.path_to_sample_ids = path_to_sample_ids
        self.regions_def = regions_def
        self.generate_samples = generate_samples
        self.path_to_sample_ids = path_to_sample_ids
        self.swap_time_ch = swap_time_ch
        self.years = years
        
        self.padding = padding
        self.crop_edge = crop_edge
        self.shuffle = shuffle
        self.collapse_time = collapse_time
        self.size = size

        # data splits to load (training/validation/test)
        self.root = project_root
        self.data_root = data_root
        self.data_split = data_split
        self.splits_df = load_timestamps(splits_path)
        # prepare all elements to load - sample idx will use the object 'self.idx'
        self.idxs = load_sample_ids(
            self.data_split,
            self.splits_df,
            self.len_seq_in,
            self.len_seq_predict,
            self.regions,
            self.generate_samples,
            self.years,
            self.path_to_sample_ids
        )

        if self.data_split not in ['test', 'heldout'] and shuffle:
            print(f'shuffle {self.data_split} dataset...')
            np.random.shuffle(self.idxs)

        # LOAD DATASET
        self.in_ds = load_dataset(
            self.data_root, self.data_split, self.regions, years, self.input_product
        )
        if self.data_split not in ["test", "heldout"]:
            self.out_ds = load_dataset(
                self.data_root, self.data_split, self.regions, years, self.output_product
            )
        else:
            self.out_ds = []

        self.data_cleansing = data_cleansing
        ## Remove all-zero gt sequences	
        if (self.data_split == 'training') & (self.data_cleansing==True):
            root_path = 'utils'
            os.makedirs(root_path, exist_ok=True)
            file_name = 'th100.0_rate1.pkl'
            file_path = os.path.join(root_path, file_name)
            
            if not os.path.exists(file_path):
                threshold = 100.0
                remove_rate = 1         # 1 for remove all, 0 for remove nothing
                
                save_name = 'th{}_rate{}.pkl'.format(threshold, remove_rate)
                save_path = os.path.join(root_path, save_name)
                
                rm_idxs = []
                idx = 0
                while idx < self.__len__():
                    output_data, out_masks = get_sequence(self.idxs[idx][1], self.data_root, self.data_split, self.idxs[idx][2],
                                                        self.output_product, [], self.preprocess_target, self.swap_time_ch, self.out_ds)
                    output_sum = np.sum(output_data)
                    rand_num = random()
                    if output_sum < 100.0 and remove_rate > rand_num:
                        rm_idxs.append(idx)
                    idx += 1
                print("{} sequences need to be removed!!".format(len(rm_idxs)))
                
                with open(save_path, 'wb') as f:
                    pickle.dump(rm_idxs, f)
            else:
                with open(file_path, 'rb') as f:
                    rm_idxs = pickle.load(f)
                print("{} sequences need to be removed!!".format(len(rm_idxs)))
                
            ## Remove sequences
            orig_len = self.__len__()
            delete_multiple_element(self.idxs, rm_idxs)
            n_remove = orig_len - self.__len__()
            print("{} sequences are removed!!".format(n_remove)) 

    def __len__(self):
        """total number of samples (sequences of in:4-out:1 in our case) to train"""
        # print(len(self.idxs), "-------------------", self.data_split)
        return len(self.idxs)

    def load_in(self, in_seq, seq_r, metadata, loaded_input=False):
        in0 = time.time()
        input_data, in_masks = get_sequence(
            in_seq,
            self.data_root,
            self.data_split,
            seq_r,
            self.input_product,
            self.sat_bands,
            self.preprocess_input,
            self.swap_time_ch,
            self.in_ds,
        )
        
        if self.sat_bands is not ['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073']:
            input_data = get_channels(input_data, self.sat_bands)

        if self.crop_edge is not None:
            input_data = crop_numpy(input_data, self.crop_edge, self.padding)
            
        if self.collapse_time:
            input_data = time_2_channels(input_data, self.size, self.size)
            
        if VERBOSE:
            print(np.shape(input_data), time.time() - in0, "in sequence time")
        return input_data, metadata

    def load_out(self, out_seq, seq_r, metadata):
        t1 = time.time()
        # GROUND TRUTH (OUTPUT)
        if self.data_split not in ["test", "heldout"]:
            output_data, out_masks = get_sequence(
                out_seq,
                self.data_root,
                self.data_split,
                seq_r,
                self.output_product,
                [],
                self.preprocess_target,
                self.swap_time_ch,
                self.out_ds,
            )

            # collapse time to channels
            metadata["target"]["mask"] = out_masks
        else:  # Just return [] if its test/heldout data
            output_data = np.array([])
        if VERBOSE:
            print(time.time() - t1, "out sequence")
        return output_data, metadata

    def load_in_out(self, in_seq, out_seq=None, seq_r=None):
        metadata = {
            "input": {"mask": [], "timestamps": in_seq},
            "target": {"mask": [], "timestamps": out_seq},
        }

        t0 = time.time()
        input_data, metadata = self.load_in(in_seq, seq_r, metadata)
        output_data, metadata = self.load_out(out_seq, seq_r, metadata)

        if VERBOSE:
            print(time.time() - t0, "seconds")
        return input_data, output_data, metadata

    def __getitem__(self, idx):
        """load 1 sequence (1 sample)"""
        in_seq = self.idxs[idx][0]
        out_seq = self.idxs[idx][1]
        seq_r = self.idxs[idx][2]
# #        print("=== DEBUG in_seq: ",in_seq, file=sys.stderr);
#         print("=== DEBUG in_seq: ",in_seq);
        return self.load_in_out(in_seq, out_seq, seq_r)


class Normalise(object):
    """Dataset Transform: "Normalise values for each band."""

    def __init__(self, mean, std):
        """Normalise values for each band
        Args:
            mean (list): mean value of bands
            std (list): standard deviation of bands
        """
        self.mean = mean
        self.std = std
        super().__init__()

    def __call__(self, sample):
        """Normalise values for each band
        Args:
            sample (Tensor, Tensor): sample and labels for sample as tensor
        Returns:
            sample (Tensor, Tensor): sample and labels for sample normalized
        """
        data, labels = sample
        # For every channel, subtract the mean, and divide by the standard deviation\
        # possible approach = loop through band dim and access right values in corresponding dims of mean / stdev
        for t, m, s in zip(data, self.mean, self.std):
            t.sub_(m).div_(s)
        return (data, labels)
