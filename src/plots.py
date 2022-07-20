# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import imageio
import os
import torch

import numpy as np
import pandas as pd
import util.depth_transformations as depth_transforms

from matplotlib import cm, colors, pyplot as plt
from datasets import create_sample_wrapper, CameraViewCellDataset
from train_data import TrainConfig
from features import FeatureSetKeyConstants
from util.saveimage import save_img, transform_img
from util.helper import init_full_image_dict, add_inference_dict
from tqdm import tqdm
from nerf_raymarch_common import nerf_get_normalization_function_abbr


def calculate_mse(diff):
    return torch.mean(diff ** 2)


def calculate_psnr(mse):
    return 10 * torch.log10(1. / mse)


def render_img(train_config, img_samples, img_name=None, model_idxs=None):
    targets = []
    imgs = []
    imgs_train_inference = []
    gt_depth = None
    gt_depth_world = None
    estimated_depth = None
    adaptive_samples = None
    oracle_weights = None
    nerf_weights = None
    nerf_alphas = None

    dim_h = train_config.dataset_info.h
    dim_w = train_config.dataset_info.w

    start_index = 0
    inference_chunk_size = train_config.config_file.inferenceChunkSize
    for batch in img_samples.batches(inference_chunk_size):
        img_parts, inf_dicts = train_config.inference(batch, gradient=False, is_inference=True)
        img_parts_train, inf_dicts_train = train_config.inference(batch, gradient=False, is_inference=False)

        # we create the tensors once and then only slice the results
        if len(imgs) == 0:
            for i in range(len(img_parts)):
                imgs.append(torch.zeros((dim_h * dim_w, img_parts[i].shape[-1]), device=train_config.device,
                                        dtype=torch.float32))
                imgs_train_inference.append(torch.zeros((dim_h * dim_w, img_parts_train[i].shape[-1]),
                                                        device=train_config.device, dtype=torch.float32))

                train_target = batch.get_train_target(i)

                if train_target is not None:
                    targets.append(torch.zeros((dim_h * dim_w, train_target.shape[-1]),
                                               device=train_config.device, dtype=torch.float32))
                else:
                    targets.append(None)

            if estimated_depth is None and FeatureSetKeyConstants.nerf_estimated_depth in inf_dicts[-1]:
                estimated_depth = torch.zeros((dim_h * dim_w,
                                               inf_dicts[-1][FeatureSetKeyConstants.nerf_estimated_depth].shape[-1]),
                                              device=train_config.device, dtype=torch.float32)

            if gt_depth is None and FeatureSetKeyConstants.input_depth_groundtruth in inf_dicts_train[-1]:
                gt_depth = torch.zeros((dim_h * dim_w,
                                        inf_dicts_train[-1][FeatureSetKeyConstants.input_depth_groundtruth].shape[-1]),
                                       device=train_config.device, dtype=torch.float32)

            if gt_depth_world is None and FeatureSetKeyConstants.input_depth_groundtruth_world in inf_dicts_train[-1]:
                gt_depth_world = torch.zeros((dim_h * dim_w,
                                              inf_dicts_train[-1][FeatureSetKeyConstants.input_depth_groundtruth_world].shape[-1]),
                                             device=train_config.device, dtype=torch.float32)

            if adaptive_samples is None and FeatureSetKeyConstants.adaptive_sample_positions in inf_dicts[-1]:
                adaptive_samples = torch.zeros((dim_h * dim_w), device=train_config.device, dtype=torch.float32)

            if oracle_weights is None and FeatureSetKeyConstants.oracle_weights in inf_dicts[-1]:
                oracle_weights = torch.zeros((dim_h * dim_w,
                                              inf_dicts[-1][FeatureSetKeyConstants.oracle_weights].shape[-1]),
                                             device=train_config.device, dtype=torch.float32)
            if nerf_weights is None and FeatureSetKeyConstants.nerf_weights_output in inf_dicts[-1]:
                nerf_weights = torch.zeros((dim_h * dim_w,
                                            inf_dicts[-1][FeatureSetKeyConstants.nerf_weights_output].shape[-1]),
                                           device=train_config.device, dtype=torch.float32)
            if nerf_alphas is None and FeatureSetKeyConstants.nerf_alpha_output in inf_dicts[-1]:
                nerf_alphas = torch.zeros((dim_h * dim_w,
                                            inf_dicts[-1][FeatureSetKeyConstants.nerf_alpha_output].shape[-1]),
                                           device=train_config.device, dtype=torch.float32)


        end_index = min(start_index + inference_chunk_size, dim_w * dim_h)

        for i in range(len(imgs)):
            imgs[i][start_index:end_index, :] = img_parts[i][:inference_chunk_size, :]
            imgs_train_inference[i][start_index:end_index, :] = img_parts_train[i][:inference_chunk_size, :]

            if targets[i] is not None:
                targets[i][start_index:end_index, :] = batch.get_train_target(i)[:inference_chunk_size, :]

        if estimated_depth is not None:
            estimated_depth[start_index:end_index, :] = \
                inf_dicts[-1][FeatureSetKeyConstants.nerf_estimated_depth][:inference_chunk_size, :]

        if gt_depth is not None:
            gt_depth[start_index:end_index, :] = \
                inf_dicts_train[-1][FeatureSetKeyConstants.input_depth_groundtruth][:inference_chunk_size, :]

        if gt_depth_world is not None:
            gt_depth_world[start_index:end_index, :] = \
                inf_dicts_train[-1][FeatureSetKeyConstants.input_depth_groundtruth_world][:inference_chunk_size, :]

        if adaptive_samples is not None:
            adaptive_samples[start_index:end_index] = \
                inf_dicts_train[-1][FeatureSetKeyConstants.adaptive_sample_positions][:inference_chunk_size]

        if oracle_weights is not None:
            oracle_weights[start_index:end_index] = \
                inf_dicts[-1][FeatureSetKeyConstants.oracle_weights][:inference_chunk_size]

        if nerf_weights is not None:
            nerf_weights[start_index:end_index] = \
                inf_dicts[-1][FeatureSetKeyConstants.nerf_weights_output][:inference_chunk_size]
        
        if nerf_alphas is not None:
            nerf_alphas[start_index:end_index] = \
                inf_dicts[-1][FeatureSetKeyConstants.nerf_alpha_output][:inference_chunk_size]

        start_index = end_index

    if model_idxs is None:
        for i in range(len(imgs)):
            save_img(imgs[i], train_config.dataset_info, f"{train_config.logDir}{img_name}_{i}.png")
            save_img(imgs_train_inference[i], train_config.dataset_info, f"{train_config.logDir}{img_name}_{i}_train_input.png")

            if targets[i] is not None:
                save_img(targets[i], train_config.dataset_info, f"{train_config.logDir}{img_name}_{i}_train_targets.png")
            
    else:
        for i in range(len(model_idxs)):
            save_img(imgs[i], train_config.dataset_info, f"{train_config.logDir}{img_name}_{i}.png")
            save_img(imgs_train_inference[i], train_config.dataset_info, f"{train_config.logDir}{img_name}_{i}_train_input.png")

            if targets[i] is not None:
                save_img(targets[i], train_config.dataset_info, f"{train_config.logDir}{img_name}_{i}_train_targets.png")

    if gt_depth is not None:
        save_img(gt_depth, train_config.dataset_info, f"{train_config.logDir}{img_name}_gt_depth.png")

    if estimated_depth is not None:
        color_map = cm.get_cmap('plasma')
        save_img(color_map(estimated_depth.detach().cpu().numpy()), train_config.dataset_info,
                 f"{train_config.logDir}{img_name}_estimated_depth.png")

    if adaptive_samples is not None and torch.sum(adaptive_samples) != dim_w * dim_h:
        color_map = cm.get_cmap('coolwarm')
        save_img(color_map(adaptive_samples.detach().cpu().numpy()), train_config.dataset_info,
                 f"{train_config.logDir}{img_name}_adaptive_samples.png")

    class CustomSize:
        def __init__(self, dimh, dimw):
            self.h = dimh
            self.w = dimw

    if oracle_weights is not None:
        histogram = (oracle_weights * 255).flatten().cpu().numpy()

        histogram = np.histogram(histogram, 255, (0.0, 255.0), density=False)

        cmap = colors.LinearSegmentedColormap.from_list("", ["red", "green", "blue"])

        plt.bar(histogram[1][:-1], histogram[0], color=cmap(histogram[1][:-1]/255))

        plt.title("Oracle Activations Histogram")
        plt.xlabel("Activation Ranges")
        plt.ylabel("Number of Samples")

        plt.savefig(f"{train_config.logDir}{img_name}_oracle_histogram.png")
        plt.close()

        oracle_weights = oracle_weights[int(dim_h * 0.5 * dim_w):int(((dim_h * 0.5) + 1) * dim_w)].detach().cpu().numpy()
        oracle_weights = np.transpose(oracle_weights)
        oracle_weights = np.flip(oracle_weights, 0)
        color_map = cm.get_cmap("magma")

        save_img(color_map(oracle_weights), CustomSize(oracle_weights.shape[0], dim_w),
                 f"{train_config.logDir}{img_name}_oracle_weights.png")

    if nerf_weights is not None:
        nerf_weights = nerf_weights[int(dim_h * 0.5 * dim_w):int(((dim_h * 0.5) + 1) * dim_w)].detach().cpu().numpy()
        nerf_weights = np.transpose(nerf_weights)
        nerf_weights = np.flip(nerf_weights, 0)
        color_map = cm.get_cmap("magma")

        save_img(color_map(nerf_weights), CustomSize(nerf_weights.shape[0], dim_w),
                 f"{train_config.logDir}{img_name}_nerf_weights.png")

    if targets[-1] is not None:
        print(f'\nRender img PSNR {img_name}: {calculate_psnr(calculate_mse(targets[-1] - imgs[-1]))}\n')


def render_all_imgs(train_config: TrainConfig, subfolder_name="", dataset_name="test"):
    os.makedirs(os.path.join(train_config.logDir, subfolder_name, dataset_name), exist_ok=True)

    inference_chunk_size = train_config.config_file.inferenceChunkSize
    data_set, _ = train_config.get_data_set_and_loader(dataset_name)
    saved_full_images = data_set.full_images
    data_set.full_images = True

    psnrs = []

    dim_w = train_config.dataset_info.w
    dim_h = train_config.dataset_info.h

    # we use the dataset here, as using the data_loader would make it necessary to handle all the different batch sizes
    for i, sample_data in enumerate(tqdm(data_set, desc=f"rendering all images ({dataset_name})", position=0, leave=True)):
        img_samples = create_sample_wrapper(sample_data, train_config, True)

        base_filename = os.path.splitext(os.path.basename(data_set.image_filenames[i]))[0]

        imgs = []
        target = None
        inference_dict_full_list = None

        start_index = 0
        for batch in img_samples.batches(inference_chunk_size):
            img_part, inference_dict_part = train_config.inference(batch, gradient=False, is_inference=True)
            if len(imgs) == 0:
                for j in range(len(img_part)):
                    imgs.append(torch.zeros((dim_h * dim_w, img_part[j].shape[-1]), device=train_config.device,
                                            dtype=torch.float32))

                inference_dict_full_list = init_full_image_dict(train_config, inference_dict_part, complete=False)

                target = torch.zeros((dim_h * dim_w, batch.get_train_target(-1).shape[-1]),
                                     device=train_config.device, dtype=torch.float32)

            end_index = min(start_index + train_config.config_file.inferenceChunkSize, dim_w * dim_h)

            for j in range(len(img_part)):
                imgs[j][start_index:end_index] = img_part[j][:inference_chunk_size]

            add_inference_dict(inference_dict_part, inference_dict_full_list, start_index, end_index)

            target[start_index:end_index, :] = batch.get_train_target(-1)[:inference_chunk_size, :]

            start_index = end_index

        # Reshape all values to [h, w] from dict

        for j in range(len(inference_dict_full_list)):
            for key in inference_dict_full_list[j]:
                if inference_dict_full_list[j][key].ndim != 0 and FeatureSetKeyConstants.input_depth_range not in key:
                    inference_dict_full_list[j][key] = torch.reshape(inference_dict_full_list[j][key], [train_config.dataset_info.h, train_config.dataset_info.w, *inference_dict_full_list[j][key].shape[1:]])

        for net_idx, img in enumerate(imgs):
            save_img(img, train_config.dataset_info, f"{train_config.logDir}{subfolder_name}{dataset_name}/{net_idx}_{base_filename}.png")

        if FeatureSetKeyConstants.input_depth_groundtruth in inference_dict_full_list[-1]:
            save_img(inference_dict_full_list[-1][FeatureSetKeyConstants.input_depth_groundtruth], train_config.dataset_info, f"{train_config.logDir}{subfolder_name}{dataset_name}/{base_filename}_input_depth_gth.png")

        if FeatureSetKeyConstants.nerf_estimated_depth in inference_dict_full_list[-1]:
            save_img(inference_dict_full_list[-1][FeatureSetKeyConstants.nerf_estimated_depth], train_config.dataset_info, f"{train_config.logDir}{subfolder_name}{dataset_name}/{base_filename}_estimated_depth.png")

        raw_save_suffix = ""
        if "lin" not in train_config.config_file.depthTransform:
            raw_save_suffix += train_config.config_file.depthTransform[0:2]

        if train_config.config_file.rayMarchNormalization is not None and len(train_config.config_file.rayMarchNormalization) > 0:
            raw_save_suffix += nerf_get_normalization_function_abbr(train_config.config_file.rayMarchNormalization[-1])

        if FeatureSetKeyConstants.nerf_estimated_depth in inference_dict_full_list[-1]:
            # Load depth range and depth map
            depth_range = inference_dict_full_list[-1][FeatureSetKeyConstants.input_depth_range]
            depth_map = inference_dict_full_list[-1][FeatureSetKeyConstants.nerf_estimated_depth]

            # In case the depth range contains more than 2 elements due to the export
            input_depth_range = depth_range[:2]

            # Fix nans in predicted depth
            #depth_map[torch.isnan(depth_map)] = input_depth_range[-1][-1]

            print(f"\nDepth map {base_filename} min: {depth_map.min()}, max: {depth_map.max()}")
            world_depth = train_config.f_in[-1].depth_transform.to_world(depth_map, input_depth_range[-1])

            print(f"\nDepth map (to_world) {base_filename} min: {world_depth.min()}, max: {world_depth.max()}")
            np.savez(f"{train_config.logDir}{subfolder_name}{dataset_name}/{base_filename}_depth.npz", world_depth)
            save_img(depth_map[..., None], train_config.dataset_info, f"{train_config.logDir}{subfolder_name}{dataset_name}/{base_filename}_{raw_save_suffix}_depth.png")

            # Export TermiNeRF files
            # We use the normalized nerf weights here - the depth ranges are the same and defined in the dataset
            if FeatureSetKeyConstants.nerf_weights_output in inference_dict_full_list[-1]:
                nerf_weights = inference_dict_full_list[-1][FeatureSetKeyConstants.nerf_weights_output]
                nerf_weights_quantized = (nerf_weights * 255).to(torch.uint8)
                torch.save(nerf_weights_quantized,
                           f"{train_config.logDir}{subfolder_name}{dataset_name}/{base_filename}_weights.trch")
                a = 3



        else:
            for j in range(len(inference_dict_full_list)):
                for key in inference_dict_full_list[j]:
                    torch.save(inference_dict_full_list[j][key],
                               f"{train_config.logDir}{subfolder_name}{dataset_name}/{i}_{j}_{key}_{raw_save_suffix}.raw")

        psnrs.append(calculate_psnr(calculate_mse(target - imgs[-1])))

    print("\n")
    psnrs_np = []
    for i in range(len(psnrs)):
        print(f"Render all img psnr {i} {psnrs[i]}")
        psnrs_np.append(psnrs[i].cpu().numpy())

    print(f"Average PSNR: {np.array(psnrs_np).mean()}")

    data_set.full_images = saved_full_images


def render_video(train_config: TrainConfig, vid_name=None, out_dir=None):
    c_file = train_config.config_file
    data = train_config.dataset_info

    camera_dataset = CameraViewCellDataset(c_file, train_config, data)

    dim_w = train_config.dataset_info.w
    dim_h = train_config.dataset_info.h
    inference_chunk_size = train_config.config_file.inferenceChunkSize

    video_output_dir = c_file.logDir

    if out_dir is not None:
        video_output_dir = out_dir

    imgs_8bit = [None] * len(train_config.models)
    # render 1 image at a time, because of memory
    for i in tqdm(range(len(camera_dataset)), desc=f'rendering video {vid_name}', position=0, leave=True):
        img_samples = create_sample_wrapper(camera_dataset[i], train_config, True)

        imgs = []
        start_index = 0
        for batch in img_samples.batches(inference_chunk_size):
            img_part, inference_dict_part = train_config.inference(batch, gradient=False, is_inference=True)
            if len(imgs) == 0:
                for j in range(len(img_part)):
                    imgs.append(torch.zeros((dim_h * dim_w, img_part[j].shape[-1]), device=train_config.device,
                                            dtype=torch.float32))

            end_index = min(start_index + inference_chunk_size, dim_w * dim_h)

            for j in range(len(img_part)):
                imgs[j][start_index:end_index] = img_part[j][:inference_chunk_size]

            start_index = end_index

        for net_idx, img in enumerate(imgs):
            img = transform_img(img, data)
            img8bit = (img[None] * 255).astype(np.uint8)
            if imgs_8bit[net_idx] is None:
                imgs_8bit[net_idx] = img8bit
            else:
                imgs_8bit[net_idx] = np.concatenate((imgs_8bit[net_idx], img8bit), axis=0)

    for net_idx, v in enumerate(imgs_8bit):
        imageio.mimwrite(os.path.join(video_output_dir, f"{vid_name}_{net_idx}.mp4"), v, fps=30, quality=8)


def plot_training_stats(dir, csv_path, x_column, y_column):
    fig, ax = plt.subplots()
    df = pd.read_csv(f"{dir}{csv_path}")
    df.plot(ax=ax, x=x_column, y=y_column)
    plt.savefig(f'{dir}{x_column}_{y_column}.pdf')
    plt.close(fig)
