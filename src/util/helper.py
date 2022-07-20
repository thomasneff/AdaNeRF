# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch

import numpy as np


# we define this here again to avoid a circular import
class FeatureSetKeyConstants:
    network_output = 'NetworkOutputBatch'
    input_depth_range = "InputDepthRange"
    input_depth = "InputDepth"


# torch to numpy
def t2np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        a.device)
    return torch.index_select(a, dim, order_index)


def config_to_name(in_features, out_features, models, encodings, enc_args_in, losses, loss_weights, loss_components,
                   loss_c_weights, loss_blending_start, loss_blending_duration, loss_alpha, loss_beta):
    name = ""
    for i in range(len(in_features)):
        if i > 0:
            name += "_"
        enc_args = f"({enc_args_in[i]})" if enc_args_in[i] not in ["", "none"] else ""
        enc = f"({encodings[i]}{enc_args})" if encodings[i] not in ["", "none"] else ""

        loss_alpha_beta = ""

        if len(loss_alpha) > i and len(loss_beta) > i:
            loss_alpha_beta = f"l{loss_alpha[i]}_{loss_beta[i]}_"

        name += f"{loss_alpha_beta}{in_features[i].get_string()}{enc}-{models[i].name}-{out_features[i].get_string()}"

    print_loss_weights = False
    temp = ""
    for i, weight in enumerate(loss_weights):
        if i == 0:
            temp += "_["
        else:
            temp += "_"
        temp += f"{weight}"
        print_loss_weights = print_loss_weights or weight != 1.0

    if print_loss_weights:
        temp += f"]"
        name += temp

    if loss_blending_start > 0 and loss_blending_duration > 0:
        name += f"_[{loss_blending_start / 1000:g}k_{loss_blending_duration / 1000:g}k]"

    for i, loss in enumerate(losses):
        if loss == "NeRFWeightMultiplicationLoss":
            for j, comp in enumerate(loss_components):
                name += f"_{comp[0]}"
                if loss_c_weights[j] > 0.0:
                    name += f"({loss_c_weights[j]})"

    return name


def init_full_image_dict(train_config, inference_dict, complete=True):
    dim_h = train_config.dataset_info.h
    dim_w = train_config.dataset_info.w

    inference_size = train_config.config_file.inferenceChunkSize

    inference_dict_full_list = []

    for idx in range(len(inference_dict)):
        inference_dict_full = {}

        for key, value in inference_dict[idx].items():
            if complete or key in train_config.config_file.outputNetworkRaw:
                if key == FeatureSetKeyConstants.network_output and value.shape[0] != inference_size:
                    shape = list(value.size())
                    shape[0] = int(shape[0] / inference_size)
                    shape.insert(0, dim_h * dim_w)
                elif key == FeatureSetKeyConstants.input_depth and value.shape[0] != inference_size:
                    shape = list(value.size())
                    del shape[0]
                    shape[0] = dim_h * dim_w
                elif key == FeatureSetKeyConstants.input_depth_range:
                    inference_dict_full[key] = value
                    continue
                else:
                    shape = list(value.size())
                    shape[0] = dim_h * dim_w

                inference_dict_full[key] = torch.zeros(shape, device="cpu", dtype=torch.float32)

        inference_dict_full_list.append(inference_dict_full)

    return inference_dict_full_list


def add_inference_dict(inference_dict, inference_dict_full, start, end):
    size = end - start

    for idx in range(len(inference_dict)):
        for key, value in inference_dict[idx].items():
            if key in inference_dict_full[idx]:
                if key == FeatureSetKeyConstants.network_output and value.shape[0] != end - start:
                    shape = list(value.size())
                    shape[0] = int(shape[0] / size)
                    shape.insert(0, size)
                    value = value.view(shape)
                elif key == FeatureSetKeyConstants.input_depth and value.shape[0] != size:
                    value = value[0]
                elif key == FeatureSetKeyConstants.input_depth_range:
                    continue

                inference_dict_full[idx][key][start:end] = value[:size]
