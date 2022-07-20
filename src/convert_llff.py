# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from util import load_llff
import torch
import numpy as np
import os
import configargparse
import json
from PIL import Image
from util import load_llff_nex
def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        float_list = [float(np_float) for np_float in row]
        matrix_list.append(float_list)
    return matrix_list

if __name__ == "__main__":
    p = configargparse.ArgParser()
    p.add_argument('-dir', '--dir', default="", type=str, help="directory to convert / LLFF directory")
    p.add_argument('-factor', '--factor', default=None, type=int, help="downsampling factor for images")
    p.add_argument('-nex', '--nex', default=0, type=int, help="use nex loading code")


    cl = p.parse_args()

    if cl.nex == 0:
        images, poses, bds, render_poses, i_test = load_llff.load_llff_data(cl.dir, cl.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=False)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]

    else:
        images, poses, bds, render_poses, i_test, intrinsic = load_llff_nex.load_llff_data(cl.dir, cl.factor,
                                                                            recenter=True, bd_factor=.75,
                                                                            spherify=False)
        hwf = intrinsic[:3].flatten()

    llff_hold = 8

    print('Loaded llff', images.shape, hwf, cl.dir)

    print('DEFINING BOUNDS')
    near = np.ndarray.min(bds) * .9
    far = np.ndarray.max(bds) * 1.
    print('NEAR FAR', near, far)

    # View cell center is zero when recenter is used for load_llff_data, but we're computing it anyways
    view_cell_center = poses[:, :, 3:].mean(axis=0)
    # View cell size is simply the maximum distance of the poses to the center in all axes (times 2)
    view_cell_size = 2 * np.abs(poses[:, :, 3:] - view_cell_center).max(axis=0)

    # Split data into train-val-test folders, this doesn't really matter but our dataloader expects that...
    i_test = np.arange(images.shape[0])[::llff_hold]
    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

    dataset_indices = {
        'train': i_train,
        'val': i_val,
        'test': i_test,
    }

    splits = ['train', 'val', 'test']
    metas = {}
    
    with open(os.path.join(cl.dir, "dataset_info.json"), "w") as f:
        dataset_info = {}
        dataset_info['camera_angle_x'] = 2 * np.arctan((hwf[1] * 0.5) / hwf[2])
        dataset_info['view_cell_center'] = np.squeeze(view_cell_center).tolist()
        dataset_info['view_cell_size'] = np.squeeze(view_cell_size).tolist()
        dataset_info['resolution'] = [images.shape[2], images.shape[1]]
        dataset_info['flip_depth'] = False
        dataset_info['depth_distance_adjustment'] = False
        dataset_info['depth_ignore'] = 1.01 * float(far)
        dataset_info['depth_range'] = [float(near), float(far)]
        dataset_info['depth_range_warped_log'] = [float(near), float(far)]
        dataset_info['depth_range_warped_lin'] = [float(near), float(far)]
        
        json.dump(dataset_info, f, indent=4)

    out_data = {}
    out_data["frames"] = []
    render_poses = render_poses[:, :3, :4]

    for frame_idx in range(len(render_poses)):
        pose_frame = render_poses[frame_idx, :, :]

        frame_data = {
            "p": frame_idx,
            "transform_matrix": listify_matrix(pose_frame)
        }
        frame_data["transform_matrix"].append([0.0, 0.0, 0.0, 1.0])

        out_data["frames"].append(frame_data)
    
    with open(os.path.join(cl.dir, "cam_path_spiral.json"), "w") as f:
        json.dump(out_data, f, indent=4)
		
    for s in splits:
        out_data = {}
        out_data['frames'] = []

        split_indices = dataset_indices[s]

        new_subpath = os.path.join(cl.dir, s)
        os.makedirs(new_subpath, exist_ok=True)

        for frame_idx in split_indices:
            pose_frame = poses[frame_idx, :, :]
            image_frame = images[frame_idx, :, :]

            frame_data = {
               'file_path': f"./{s}/{frame_idx:05d}",
               'rotation': 0,
               'transform_matrix': listify_matrix(pose_frame)
            }

            frame_data['transform_matrix'].append([0.0, 0.0, 0.0, 1.0])

            out_data['frames'].append(frame_data)

            # Save image
            img = Image.fromarray((image_frame * 255).astype(np.uint8), 'RGB')
            img.save(new_subpath + '/' + f"{frame_idx:05d}" + '.png')


        with open(os.path.join(cl.dir, 'transforms_{}.json'.format(s)), 'w') as fp:
            json.dump(out_data, fp, indent=4)
