# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os

import numpy as np


class SamplePlacementTracker:
    """
    This class tracks and stores memory for the iterative sample reduction NeRF training.
    Per pixel in all training images, it stores 128 bit that correspond to the state of sample locations being active.
    Starts initialized to all 1s.
    """

    def __init__(self, num_images, width, height, max_sample_count=128):
        self.num_images = 0
        self.max_sample_count = max_sample_count
        self.width = width
        self.height = height
        self.num_images = num_images

        self.bit_data = np.packbits(np.ones(shape=[self.num_images, self.height, self.width, self.max_sample_count], dtype=bool), axis=-1)

        # Test extraction/recovery of data
        #raw_data = np.unpackbits(self.boolean_data, axis=-1).reshape([self.height, self.width, self.max_sample_count])
        #raw_data[0, 210, 3:5] = False
        #self.boolean_data = np.packbits(raw_data, axis=-1)
        #raw_data = np.unpackbits(self.boolean_data, axis=-1).reshape([self.height, self.width, self.max_sample_count])

    def get_unpacked_image(self, index):
        unpacked_samples = np.unpackbits(self.bit_data[index, ...]).reshape([self.height, self.width, self.max_sample_count])

        return unpacked_samples

    def set_2_samples_test(self):
        self.bit_data[:, :, :, :] = np.array([128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    def set_32_samples_test(self):
        self.bit_data[:, :, :, :] = np.array([128 + 8] * 16)

    def set_16_samples_test(self):
        self.bit_data[:, :, :, :] = np.array([128] * 16)

    def replace_samples_batch(self, samples, batch_0, image_index):
        # From the unpacked data, we can simply pack it and replace the bit_data
        batch_size = samples.shape[1]

        # Pack the new samples
        packed_samples = np.packbits(samples.cpu().numpy().astype(bool), axis=-1)

        # Put them into the full representation
        self.bit_data[image_index].reshape(-1, self.bit_data.shape[-1])[batch_0:batch_0 + batch_size,:] = packed_samples.squeeze()

    def save(self, path):
        np.save(path, self.bit_data)

    def load(self, path):
        self.bit_data = np.load(path)

