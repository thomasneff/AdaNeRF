# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import sys
import json
import torch
import imageio
import importlib

import numpy as np
import util.depth_transformations as depth_transforms

from tqdm import trange
from torch.utils.data import Dataset, get_worker_info
from util.raygeneration import generate_ray_directions
from util.sample_placement_tracker import SamplePlacementTracker


class DatasetKeyConstants:
    color_image_full = "ColorImageFull"
    color_image_samples = "ColorImageSamples"
    depth_image_full = "DepthImageFull"
    depth_image_samples = "DepthImageSamples"
    image_sample_indices = "ImageSampleIndices"
    image_pose = "ImagePose"
    image_rotation = "ImageRotation"
    ray_directions = "RayDirections"
    image_file_names = "FileNames"
    ray_directions_samples = "RayDirectionsSamples"
    batch_input_dir = "BatchInputDir"
    train_target = "TrainTarget"
    sample_placement = "SamplePlacement"
    batch_0 = "Batch0"


def create_sample_wrapper(sample_data, train_config, single=False):
    batch_input_dirs = []
    train_targets = []

    for i in range(len(train_config.f_in)):
        batch_input_dirs.append(sample_data[f"{DatasetKeyConstants.batch_input_dir}_{i}"])

        # if we are not training, we have no targets
        target_string = f"{DatasetKeyConstants.train_target}_{i}"
        if target_string in sample_data:
            if sample_data[target_string].dtype == torch.bool:
                train_targets.append(None)
            else:
                train_targets.append(sample_data[target_string])

    if train_config.copy_to_gpu:
        for idx in range(len(batch_input_dirs)):
            for key in batch_input_dirs[idx]:
                if isinstance(batch_input_dirs[idx][key], torch.Tensor):
                    batch_input_dirs[idx][key] = batch_input_dirs[idx][key].to(train_config.config_file.device,
                                                                               non_blocking=True)

        for idx in range(len(train_targets)):
            if isinstance(train_targets[idx], torch.Tensor):
                train_targets[idx] = train_targets[idx].to(train_config.config_file.device, non_blocking=True)

    return SampleDataWrapper(batch_input_dirs, train_targets, single)


class SampleDataWrapper:
    def __init__(self, batch_input_dirs, train_targets, single):
        self.batch_input_dirs = batch_input_dirs
        self.train_targets = train_targets
        self.single = single

    def get_batch_input(self, index):
        return self.batch_input_dirs[index]

    def get_train_target(self, index):
        return self.train_targets[index]

    def batches(self, batch_size):
        n_samples = -1

        samples_location = 0 if self.single else 1

        if DatasetKeyConstants.ray_directions_samples in self.batch_input_dirs[0]:
            n_samples = self.batch_input_dirs[0][DatasetKeyConstants.ray_directions_samples].shape[samples_location]
        elif DatasetKeyConstants.color_image_samples in self.batch_input_dirs[0]:
            n_samples = self.batch_input_dirs[0][DatasetKeyConstants.color_image_samples].shape[samples_location]
        elif DatasetKeyConstants.depth_image_samples in self.batch_input_dirs[0]:
            n_samples = self.batch_input_dirs[0][DatasetKeyConstants.depth_image_samples].shape[samples_location]

        if n_samples == -1:
            print("ERROR: unable to batch sample data!")

        for batch0 in range(0, n_samples, batch_size):
            batch_input_dirs = []
            train_targets = []

            for idx in range(len(self.batch_input_dirs)):
                inner_dir = {}

                for key in self.batch_input_dirs[idx]:
                    if key == DatasetKeyConstants.color_image_samples or \
                            key == DatasetKeyConstants.depth_image_samples or \
                            key == DatasetKeyConstants.ray_directions_samples or \
                            key == DatasetKeyConstants.sample_placement:
                        if self.single:
                            inner_dir[key] = self.batch_input_dirs[idx][key][None, batch0:batch0 + batch_size, :]
                        else:
                            inner_dir[key] = self.batch_input_dirs[idx][key][:, batch0:batch0 + batch_size, :]
                    else:
                        if self.single:
                            inner_dir[key] = self.batch_input_dirs[idx][key][None]
                        else:
                            inner_dir[key] = self.batch_input_dirs[idx][key]

                inner_dir[DatasetKeyConstants.batch_0] = batch0
                batch_input_dirs.append(inner_dir)

            for idx in range(len(self.train_targets)):

                if self.train_targets[idx] is None:
                    train_targets.append(None)
                    continue

                if self.single:
                    train_targets.append(self.train_targets[idx][batch0:batch0 + batch_size])
                else:
                    train_targets.append(self.train_targets[idx][0, batch0:batch0 + batch_size])

            yield SampleDataWrapper(batch_input_dirs, train_targets, False)


class View:
    def __init__(self):
        self.fov = 0.0
        self.focal = 0.0
        self.camera_scale = 1.0
        self.view_cell_center = [0, 0, 0]
        self.view_cell_size = [0, 0, 0]
        self.base_rotation = None


class DatasetInfo:
    def __init__(self, config, train_config):
        self.config = config
        self.train_config = train_config
        self.dataset_path = config.data
        self.view = View()
        self.scale = config.scale

        from features import SpherePosDir
        self.use_warped_depth_range = []
        warped = False
        for feature_id in range(len(train_config.f_in)):
            warped = warped or isinstance(train_config.f_in[feature_id], SpherePosDir)
            self.use_warped_depth_range.append(warped)

        # read in dataset specific .json
        with open(os.path.join(self.dataset_path, "dataset_info.json"), "r") as f:
            dataset_info = json.load(f)
        self.view.view_cell_center = dataset_info["view_cell_center"]
        self.view.view_cell_size = dataset_info["view_cell_size"]
        self.view.camera_scale = 1.
        if "camera_scale" in dataset_info:
            self.view.camera_scale = float(dataset_info["camera_scale"])

        if "camera_base_orientation" in dataset_info:
            self.view.base_rotation = np.array(dataset_info["camera_base_orientation"])

        self.w, self.h = dataset_info["resolution"][0], dataset_info["resolution"][1]
        if self.scale > 1:
            self.w = self.w // self.scale
            self.h = self.h // self.scale

        self.train_config.h = self.h
        self.train_config.w = self.w

        self.view.fov = float(dataset_info["camera_angle_x"])
        self.view.focal = float(.5 * self.w / np.tan(.5 * self.view.fov))

        # vertically flip loaded depth files
        self.flip_depth = dataset_info["flip_depth"]

        # adjustments if depth is based on distance to camera plane, and not distance to camera origin
        self.depth_distance_adjustment = dataset_info["depth_distance_adjustment"]

        if "depth_ignore" not in dataset_info or "depth_range" not in dataset_info or \
                "depth_range_warped_log" not in dataset_info or \
                "depth_range_warped_lin" not in dataset_info:
            print("error: necessary depth range information not found in 'dataset_info.json'")
            sys.exit(-1)

        self.depth_ignore = float(dataset_info["depth_ignore"])

        self.depth_range = [float(dataset_info["depth_range"][0]), float(dataset_info["depth_range"][1])]

        self.depth_max = self.depth_range[1]

        if config.depthTransform == "linear":
            self.depth_transform = depth_transforms.LinearTransform
            self.depth_range_warped = [float(dataset_info["depth_range_warped_lin"][0]),
                                       float(dataset_info["depth_range_warped_lin"][1])]
        elif config.depthTransform == "log":
            self.depth_transform = depth_transforms.LogTransform
            self.depth_range_warped = [float(dataset_info["depth_range_warped_log"][0]),
                                       float(dataset_info["depth_range_warped_log"][1])]
        else:
            self.depth_transform = depth_transforms.NoneTransform
            self.depth_range_warped = [0, 1]
            self.depth_range = [0, 1]


class ViewCellDataset(Dataset):
    def __init__(self, config, train_config, dataset_info, set_name="train", num_samples=2048):
        self.config = config
        self.train_config = train_config
        self.dataset_path = config.data
        self.set_name = set_name
        self.num_samples = num_samples
        self.image_filenames = []
        self.depth_filenames = None
        self.nogt_weights_filenames = None
        self.view = dataset_info.view
        self.scale = dataset_info.scale
        self.flip_depth = dataset_info.flip_depth
        self.depth_distance_adjustment = dataset_info.depth_distance_adjustment
        self.depth_ignore = dataset_info.depth_ignore
        self.depth_max = dataset_info.depth_max
        self.depth_range = dataset_info.depth_range
        self.depth_range_warped = dataset_info.depth_range_warped
        self.depth_transform = dataset_info.depth_transform
        self.w = dataset_info.w
        self.h = dataset_info.h
        self.num_items = 0
        self.device = config.device
        self.is_inference = False  # if True, does not give training targets
        self.full_images = False  # if True, gives sample data for whole images
        self.poses = None
        self.rotations = None
        self.directions = None
        self.sample_placement_tracker = None
        self.use_nerf_depth_map = config.useNerfDepthMap

        self.depth_images = None
        self.nogt_weights = None
        
        self.load_depth = config.trainWithGTDepth or config.useNerfDepthMap

        self.base_ray_z = np.abs(
            generate_ray_directions(self.w, self.h, self.view.fov, self.view.focal)[:, :, 2]).astype(np.float32)

        if set_name == "test":
            # we do not set is_inference to True, because we still want training targets for images
            self.full_images = True
        elif set_name == "vid":
            self.is_inference = True
            self.full_images = True

    def preprocess_pos_and_dir(self, transforms):
        self.poses = transforms[:, :3, 3:].reshape(-1, 3)
        self.poses = torch.tensor(self.poses, device=self.device, dtype=torch.float32)

        self.rotations = torch.tensor(transforms[:, :3, :3], device=self.device, dtype=torch.float32)

        npdirs = generate_ray_directions(self.w, self.h, self.view.fov, self.view.focal)
        self.directions = torch.tensor(npdirs.flatten().reshape(-1, 3), device=self.device, dtype=torch.float32)

    def scale_image(self, image):
        return cv2.resize(image, (image.shape[1] // self.scale,
                                  image.shape[0] // self.scale), interpolation=cv2.INTER_AREA)

    def load_color_image(self, file_name):
        color_image = imageio.imread(file_name).astype(np.float32)

        if self.scale > 1:
            color_image = self.scale_image(color_image)

        if color_image.shape[0] != self.h or color_image.shape[1] != self.w:
            print(f"ERROR: loaded image sizes do not match expectation! Expected {self.w}x{self.h}, but got "
                  f"{color_image.shape[1]}x{color_image.shape[0]} instead!")
            sys.exit(-2)

        color_image = color_image / 255.
        return color_image[:, :, :3]

    def transform_depth_image(self, depth_image, do_not_transform=False):
        depth_image = depth_image.astype(np.float32)
        depth_image = np.resize(depth_image, (self.h * self.scale, self.w * self.scale))

        if self.flip_depth and do_not_transform is False:
            depth_image = np.flip(depth_image, 0)

        depth_only_max = depth_image.copy()
        depth_only_max[depth_only_max != self.depth_ignore] = 0
        depth_only_max = self.scale_image(depth_only_max)

        if self.scale > 1:
            if self.config.scaleInterpolation == "area":
                depth_image = self.scale_image(depth_image)
            elif self.config.scaleInterpolation == "median":
                # we first need to create a "stacked" version of the image where we have each pixel in one of the stacks
                stacked_depths = []
                for i in range(self.scale):
                    for j in range(self.scale):
                        stacked_depths.append(depth_image[i::self.scale, j::self.scale])

                depth_sorted = np.sort(np.dstack(stacked_depths), -1)
                # Take the element that is just smaller than the median
                depth_image = depth_sorted[:, :, self.scale - 1]
            else:
                depth_image = depth_image[0::self.scale, 0::self.scale]

        depth_image[depth_only_max != 0] = self.depth_ignore

        if do_not_transform:
            return depth_image.reshape(1, self.h, self.w, 1)

        if self.depth_distance_adjustment:
            depth_image = depth_image / self.base_ray_z[:, :]

        depth_image = (depth_image - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])

        depth_image = self.depth_transform.from_world(
            depth_transforms.LinearTransform.to_world(depth_image, self.depth_range), self.depth_range)
        depth_image[depth_only_max != 0] = 1.

        depth_image = depth_image.reshape(1, self.h, self.w, 1)
        return depth_image

    def load_depth_image(self, file_name):
        np_file = np.load(file_name)
        depth_image = np_file["depth"] if "depth" in np_file.files else np_file[np_file.files[0]]

        return self.transform_depth_image(depth_image)

    def load_exported_nerf_depth(self, file_name):
        quantized_weights_dict_full = torch.load(file_name)
        depth_image = quantized_weights_dict_full['OutputDepthMap'].cpu().numpy()
        exported_depth_range = quantized_weights_dict_full['InputDepthRange'].cpu().numpy()

        return self.depth_transform.from_world(self.transform_depth_image(depth_image, do_not_transform=True), exported_depth_range)

    def get_random_sample_indices(self, device="cpu"):
        if not self.full_images:
            rand_pixels_2d = self.train_config.pixel_idx_sequence_gen.get_discrete_tensor_subset(self.num_samples,
                                                                                                 device="cpu",
                                                                                                 minv=0,
                                                                                                 maxv=torch.tensor(
                                                                                                     [self.h, self.w],
                                                                                                     dtype=torch.long))

            random_sample_indices = rand_pixels_2d[:, 0] + self.h * rand_pixels_2d[:, 1]
        else:
            random_sample_indices = torch.tensor([i for i in range(self.w * self.h)], device=device)

        return random_sample_indices.to(device)

    def determine_color_image_name(self, frame, dataset_info):
        file_path = os.path.join(self.dataset_path, frame["file_path"][2:])
        file_name = file_path + ".png"

        return file_name, file_path

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        pass


# worker init function for OnThFlyViewCellDataset
def worker_offset_sequence(worker_id):
    worker_info = get_worker_info()
    # we set the offset such that all workers start at different offsets
    offset = int((worker_info.dataset.h * worker_info.dataset.w / worker_info.num_workers) * worker_id)
    worker_info.dataset.train_config.pixel_idx_sequence_gen.set_offset(offset)


class OnTheFlyViewCellDataset(ViewCellDataset):
    def __init__(self, config, train_config, dataset_info, set_name="train", num_samples=2048):
        super(OnTheFlyViewCellDataset, self).__init__(config, train_config, dataset_info, set_name, num_samples)

        # on the fly loading only works on CPU
        self.device = "cpu"

        # read in transformation .json
        with open(os.path.join(self.dataset_path, f"transforms_{set_name}.json"), "r") as f:
            json_data = json.load(f)

        self.num_items = len(json_data["frames"])
        transforms = None

        for frame_idx, frame in enumerate(json_data["frames"]):
            pose = np.array(frame["transform_matrix"]).astype(np.float32)

            file_name, file_path = self.determine_color_image_name(frame, dataset_info)

            self.image_filenames.append(file_name)
            
            pose = np.array(frame["transform_matrix"]).astype(np.float32)

            
            
            if self.load_depth:
                # store nogt_weights name
                nogt_weights_name = file_path + "_weights.trch"
                if self.nogt_weights_filenames is None and os.path.exists(nogt_weights_name):
                    self.nogt_weights_filenames = [nogt_weights_name]
                elif os.path.exists(nogt_weights_name) and self.config.outFeatures[0] == 'TermiNeRF':
                    self.nogt_weights_filenames.append(nogt_weights_name)
                
                # store depth file paths if present
                depth_name = file_path + "_depth.npz"
                if self.depth_filenames is None and os.path.exists(depth_name):
                    self.depth_filenames = [depth_name]
                elif os.path.exists(depth_name):
                    self.depth_filenames.append(depth_name)


            # store transformations for all images
            if transforms is None:
                transforms = np.empty((self.num_items, pose.shape[0], pose.shape[1]), dtype=np.float32)
            transforms[frame_idx] = pose

        self.preprocess_pos_and_dir(transforms)

    def __getitem__(self, index):
        color_image = self.load_color_image(self.image_filenames[index])
        depth_image = None
        nogt_weights = None

        if self.depth_filenames is not None:
            depth_image = self.load_depth_image(self.depth_filenames[index])

        if self.nogt_weights_filenames is not None:
            nogt_weights = self.load_nogt_weights(self.nogt_weights_filenames[index])

        random_sample_indices = self.get_random_sample_indices("cpu")

        data_item = {
                     DatasetKeyConstants.color_image_full: torch.tensor(color_image, device=self.device)[None],
                     DatasetKeyConstants.image_sample_indices: random_sample_indices,
                     DatasetKeyConstants.image_pose: self.poses[index][None, :],
                     DatasetKeyConstants.image_rotation: self.rotations[index][None, :],
                     DatasetKeyConstants.ray_directions: self.directions
                     }

        if nogt_weights is not None:
            data_item[DatasetKeyConstants.nogt_weights_full] = nogt_weights

        if depth_image is not None:
            data_item[DatasetKeyConstants.depth_image_full] = torch.tensor(depth_image, device=self.device)

        sample_dict = {}

        for feature_idx in range(len(self.train_config.f_in)):
            f_in = self.train_config.f_in[feature_idx]
            f_in.preprocess(data_item, self.device, self.config)

            # get output of prepare_batch, which is the input to later batch function call
            in_prepared_batch = f_in.prepare_batch(data_item, self.config)
            sample_dict[f"{DatasetKeyConstants.batch_input_dir}_{feature_idx}"] = in_prepared_batch

            if not self.is_inference:
                f_out = self.train_config.f_out[feature_idx]
                f_out.preprocess(data_item, self.device, self.config)

                # get output of prepare_batch
                out_prepared_batch = f_out.prepare_batch(data_item, self.config)

                train_target = f_out.batch(out_prepared_batch)
                sample_dict[f"{DatasetKeyConstants.train_target}_{feature_idx}"] = train_target

        return sample_dict


class FullyLoadedViewCellDataset(ViewCellDataset):
    def __init__(self, config, train_config, dataset_info, set_name="train", num_samples=2048):
        super(FullyLoadedViewCellDataset, self).__init__(config, train_config, dataset_info, set_name, num_samples)

        with open(os.path.join(self.dataset_path, f"transforms_{set_name}.json"), "r") as f:
            json_data = json.load(f)

        self.num_items = len(json_data["frames"])
        transforms = None
        self.color_images = None

        tqdm_range = trange(len(json_data["frames"]), desc=f"Loading dataset {set_name:5s}", leave=True)

        for frame_idx in tqdm_range:
            frame = json_data["frames"][frame_idx]
            pose = np.array(frame["transform_matrix"]).astype(np.float32)

            file_name, file_path = self.determine_color_image_name(frame, dataset_info)

            self.image_filenames.append(file_name)

            color_image = self.load_color_image(file_name)
            depth_image = None
            nogt_weights = None
            
            pose = np.array(frame["transform_matrix"]).astype(np.float32)

            if self.load_depth:
                depth_name = file_path + "_depth.npz"
                if os.path.exists(depth_name):
                    depth_image = self.load_depth_image(depth_name)

                nogt_weights_name = file_path + "_weights.trch"
                if os.path.exists(nogt_weights_name) and self.config.outFeatures[0] == 'TermiNeRF':
                    nogt_weights = self.load_nogt_weights(nogt_weights_name)

                #depth_image = (nogt_weights.max(axis=-1).indices.float() / 128).reshape(1, nogt_weights.shape[0], nogt_weights.shape[1], 1).cpu().numpy()

                nerf_depth_name = file_path + "_QuantizedWeights_lo_nSD.raw"
                if self.use_nerf_depth_map:
                    depth_image = self.load_exported_nerf_depth(nerf_depth_name)

            if self.color_images is None:
                self.color_images = np.zeros((len(self), color_image.shape[0], color_image.shape[1],
                                              color_image.shape[2]), dtype=np.float32)
                transforms = np.zeros((len(self), pose.shape[0], pose.shape[1]), dtype=np.float32)

                if depth_image is not None:
                    self.depth_images = np.zeros((len(self), depth_image.shape[1], depth_image.shape[2], 1),
                                                 dtype=np.float32)

                if nogt_weights is not None:
                    self.nogt_weights = torch.zeros((len(self), nogt_weights.shape[1], nogt_weights.shape[2], nogt_weights.shape[3]),
                                                 dtype=torch.uint8)

            self.color_images[frame_idx] = color_image
            transforms[frame_idx] = pose
            if depth_image is not None:
                self.depth_images[frame_idx] = depth_image[0]
            if nogt_weights is not None:
                self.nogt_weights[frame_idx] = nogt_weights[0]

        self.preprocess_pos_and_dir(transforms)

        self.color_images = torch.from_numpy(self.color_images).to(self.device)
        if self.depth_images is not None:
            self.depth_images = torch.from_numpy(self.depth_images).to(self.device)

        # TODO: check nogt weights
        a = 3

        #if set_name != "test":
        #    self.sample_placement_tracker = SamplePlacementTracker(len(self.color_images), self.color_images.shape[1], self.color_images.shape[2], max_sample_count=self.config.multiDepthFeatures[0])

        if all(x == self.config.multiDepthFeatures[0] for x in config.multiDepthFeatures) is False:
            raise Exception("Error: multiDepthFeatures have to be identical for sample placement to work!")

        # Load sample placement tracker if we have a config arg supplied for it
        if self.config.samplePlacementDir is not None and self.sample_placement_tracker is not None:
            self.sample_placement_tracker.load(f"{self.config.samplePlacementDir}/{set_name}/{self.config.numRaymarchSamples[-1]}.ckpt.npy")

        data = {
                DatasetKeyConstants.color_image_full: self.color_images,
                DatasetKeyConstants.image_pose: self.poses,
                DatasetKeyConstants.image_rotation: self.rotations,
                DatasetKeyConstants.ray_directions: self.directions,
                DatasetKeyConstants.image_file_names: self.image_filenames
                }

        if self.nogt_weights is not None:
            data[DatasetKeyConstants.nogt_weights_full] = self.nogt_weights

        if self.depth_images is not None:
            data[DatasetKeyConstants.depth_image_full] = self.depth_images

        if self.sample_placement_tracker is not None:
            data[DatasetKeyConstants.sample_placement] = self.sample_placement_tracker

        # we now call preprocess on all features to perform necessary preprocess steps
        for feature_idx in range(len(self.train_config.f_in)):
            f_in = self.train_config.f_in[feature_idx]
            f_in.preprocess(data, self.device, self.config)

            if self.depth_images is not None:
                self.depth_images = data[DatasetKeyConstants.depth_image_full]

            f_out = self.train_config.f_out[feature_idx]
            f_out.preprocess(data, self.device, self.config)

    def __getitem__(self, index):
        random_sample_indices = self.get_random_sample_indices(self.device)

        data_item = {
                     DatasetKeyConstants.color_image_full: self.color_images[index][None, :],
                     DatasetKeyConstants.image_sample_indices: random_sample_indices,
                     DatasetKeyConstants.image_pose: self.poses[index][None, :],
                     DatasetKeyConstants.image_rotation: self.rotations[index][None, :],
                     DatasetKeyConstants.ray_directions: self.directions
                     }

        if self.depth_images is not None:
            data_item[DatasetKeyConstants.depth_image_full] = self.depth_images[index][None, :]

        if self.nogt_weights is not None:
            data_item[DatasetKeyConstants.nogt_weights_full] = self.nogt_weights[index][None, :]

        if self.sample_placement_tracker is not None:
            data_item[DatasetKeyConstants.sample_placement] = torch.tensor(
                self.sample_placement_tracker.get_unpacked_image(index).reshape(
                    self.color_images.shape[1] * self.color_images.shape[2], -1), device=self.device)[
                random_sample_indices].squeeze()

        sample_dict = {}

        for feature_idx in range(len(self.train_config.f_in)):
            f_in = self.train_config.f_in[feature_idx]

            # get output of prepare_batch, which is the input to later batch function call
            in_prepared_batch = f_in.prepare_batch(data_item, self.config)
            sample_dict[f"{DatasetKeyConstants.batch_input_dir}_{feature_idx}"] = in_prepared_batch

            if not self.is_inference:
                f_out = self.train_config.f_out[feature_idx]

                # get output of prepare_batch
                out_prepared_batch = f_out.prepare_batch(data_item, self.config)

                train_target = f_out.batch(out_prepared_batch)
                sample_dict[f"{DatasetKeyConstants.train_target}_{feature_idx}"] = train_target

        return sample_dict


class CameraViewCellDataset(ViewCellDataset):
    def __init__(self, config, train_config, dataset_info):
        super(CameraViewCellDataset, self).__init__(config, train_config, dataset_info, "vid", 2048)

        # Infer type and dynamically import based on config string.
        # This saves us the headache of maintaining if/else checks for the class type.
        camera_type = getattr(importlib.import_module("camera"), config.camType)
        transforms = camera_type.calc_positions(config, base_rotation=dataset_info.view.base_rotation)

        self.num_items = len(transforms)

        self.preprocess_pos_and_dir(transforms)

    def __getitem__(self, index):
        random_sample_indices = self.get_random_sample_indices(self.device)

        data_item = {
                     DatasetKeyConstants.image_sample_indices: random_sample_indices,
                     DatasetKeyConstants.image_pose: self.poses[index][None, :],
                     DatasetKeyConstants.image_rotation: self.rotations[index][None, :],
                     DatasetKeyConstants.ray_directions: self.directions
                     }

        sample_dict = {}

        for feature_idx in range(len(self.train_config.f_in)):
            f_in = self.train_config.f_in[feature_idx]

            # get output of prepare_batch, which is the input to later batch function call
            in_prepared_batch = f_in.prepare_batch(data_item, self.config)
            sample_dict[f"{DatasetKeyConstants.batch_input_dir}_{feature_idx}"] = in_prepared_batch

        return sample_dict


class MultipleViewCellCameraDataset(ViewCellDataset):
    ConstantIndex = "indices"
    ConstantRadius = "radius"
    ConstantDistance = "distance"
    ConstantData = "data"
    ConstantViewCells = "viewcells"

    def __init__(self, config, dataset_info, view_cells_data):
        super(MultipleViewCellCameraDataset, self).__init__(config, None, dataset_info, "vid", 2048)

        camera_type = getattr(importlib.import_module("camera"), config.camType)
        transforms = camera_type.calc_positions(config, base_rotation=dataset_info.view.base_rotation)

        self.num_items = len(transforms)

        self.preprocess_pos_and_dir(transforms)

        self.pose_to_view_cells = []

        for pose_idx in range(self.num_items):
            pose = self.poses[pose_idx]

            view_cells = {self.ConstantIndex: [], self.ConstantRadius: [], self.ConstantDistance: []}

            for vc_idx, view_cell in enumerate(view_cells_data):
                view_cell_orientation = np.array(view_cell["view_cell_orientation"]).astype(np.float32)
                center = view_cell_orientation[:3, 3:].reshape(3)
                size = view_cell["view_cell_size"]

                view_cell_matrix_world = torch.tensor(view_cell["view_cell_matrix_world"], device=pose.device, dtype=torch.float32)
                new_pose = torch.cat((pose, torch.ones(1, device=pose.device, dtype=torch.float32)), dim=0)

                mult_pose = torch.matmul(torch.inverse(view_cell_matrix_world), new_pose)

                if -1 <= mult_pose[0] <= 1 and -1 <= mult_pose[1] <= 1 and -1 <= mult_pose[2] <= 1:

                    radius = np.linalg.norm((np.array(size) / 2.0))

                    dist = center - pose.cpu().detach().numpy()
                    distance = np.linalg.norm(dist)

                    view_cells[self.ConstantIndex].append(view_cell["view_cell_name"])
                    view_cells[self.ConstantRadius].append(radius)
                    view_cells[self.ConstantDistance].append(distance)

            if len(view_cells[self.ConstantIndex]) == 0:
                print(f"ERROR: could not find view cell for pose!")
                sys.exit(-3)

            self.pose_to_view_cells.append(view_cells)

    def __getitem__(self, index):
        random_sample_indices = self.get_random_sample_indices(self.device)

        data_item = {DatasetKeyConstants.image_sample_indices: random_sample_indices,
                     DatasetKeyConstants.image_pose: self.poses[index][None, :],
                     DatasetKeyConstants.image_rotation: self.rotations[index][None, :],
                     DatasetKeyConstants.ray_directions: self.directions}

        sample_dict = {self.ConstantData: data_item,
                       self.ConstantViewCells: self.pose_to_view_cells[index]}

        return sample_dict
