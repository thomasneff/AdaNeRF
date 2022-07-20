# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
import numpy as np

import util.depth_transformations as depth_transforms

from importlib import import_module
from util.helper import tile



# Mostly taken from nerf-pytorch https://github.com/yenchenlin/nerf-pytorch
def nerf_raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, depth=None, accumulation_mult=None):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # This is a helper function ported over from the nerf-pytorch project
    raw2alpha = lambda raw, dists, act_fn=torch.nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, (torch.ones(1, device=raw.device) * 1e10).expand(dists[..., :1].shape)],
                      -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    
    if depth is not None:
        if accumulation_mult == "alpha":
            alpha = alpha * depth

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=raw.device), 1. - alpha + 1e-10], -1), -1)[:,:-1]
    
    if depth is not None:
        if accumulation_mult == "weights":
            weights = weights * depth

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=raw.device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)


    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map, alpha

# taken from nerf-pytorch
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def adaptive_raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, n_images=1, n_samples=2048,
                         mapping=None, depth=None, accumulation_mult=None):
    # we do this before assigning it to restored, to not end up with wrong values because of sigmoid
    sigmoided = torch.sigmoid(raw)

    if mapping is not None:
        max_num_ray_samples = int(mapping.shape[0] // (n_images * n_samples))

        # should hold all values like raw did for fixed ray sample count
        restored = torch.zeros((mapping.shape[0], 4), dtype=torch.float32, device=raw.device)
        restored_z = torch.zeros((mapping.shape[0], 1), dtype=torch.float32, device=raw.device)

        mapping = torch.nonzero(mapping)

        restored[mapping, :] = sigmoided.to(torch.float32)[:, None, :]
        restored_z[mapping, 0] = torch.unsqueeze(z_vals, 1)[:, None, 0]
    else:
        max_num_ray_samples = int(sigmoided.shape[0] // (n_images * n_samples))

        restored = sigmoided
        restored_z = torch.unsqueeze(z_vals, 1)

    restored = restored.view(n_images * n_samples, max_num_ray_samples, -1)
    restored_z = restored_z.view(n_images * n_samples, max_num_ray_samples)

    alpha = restored[..., 3]  # [N_rays, N_samples]
    rgb = restored[..., :3]  # [N_rays, N_samples, 3]

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(restored[..., 3].shape) * raw_noise_std
        
    if depth is not None:
        if accumulation_mult == "alpha":
            alpha = alpha * depth

    # alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=raw.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]

    if depth is not None:
        if accumulation_mult == "weights":
            weights = weights * depth

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * restored_z, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=raw.device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map, alpha


def nerf_get_ray_dirs(rotations, directions) -> torch.tensor:
    # multiply them with the camera transformation matrix
    ray_directions = torch.bmm(rotations, torch.transpose(directions, 1, 2))
    ray_directions = torch.transpose(ray_directions, 1, 2).reshape(directions.shape[0] * directions.shape[1], -1)

    return ray_directions


def nerf_get_z_vals(idx, z_near, z_far, poses, n_ray_samples, sampler_type='LinearlySpacedZNearZFar', **kwargs) -> torch.tensor:
    return getattr(import_module("nerf_raymarch_common"), sampler_type).generate(idx, z_near, z_far, idx.n_samples, n_ray_samples, idx.n_images, poses.device, **kwargs)


# Hierarchical sampling (section 5.2)
def nerf_sample_pdf(bins, weights, n_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=n_samples, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def normalization_none(x_in_world, view_cell_center, max_depth):
    return x_in_world


def normalization_center(x_in_world, view_cell_center, max_depth):
    return x_in_world - view_cell_center


def normalization_max_depth(x_in_world, view_cell_center, max_depth):
    return x_in_world / max_depth


def normalization_max_depth_centered(x_in_world, view_cell_center, max_depth):
    return (x_in_world - view_cell_center) / max_depth


def normalization_log_centered(x_in_world, view_cell_center, max_depth):
    localized = x_in_world - view_cell_center
    local = torch.linalg.norm(localized, dim=-1)
    log_transformed = depth_transforms.LogTransform.from_world(local, [0, max_depth])
    p = localized * (log_transformed / local)[..., None]
    return p


def normalization_inverse_dist_centered(x_in_world, view_cell_center, max_depth):
    localized = x_in_world - view_cell_center
    local = torch.linalg.norm(localized, dim=-1)
    p = localized * (1. - 1. / (1. + local))[..., None]
    return p


def normalization_inverse_sqrt_dist_centered(x_in_world, view_cell_center, max_depth):
    localized = x_in_world - view_cell_center
    local = torch.sqrt(torch.linalg.norm(localized, dim=-1))
    res = localized / (math.sqrt(max_depth) * local[..., None])
    return res


def nerf_get_normalization_function(name):
    switcher = {
        None: normalization_max_depth,
        "None": normalization_none,
        "Centered": normalization_center,
        "MaxDepth": normalization_max_depth,
        "MaxDepthCentered": normalization_max_depth_centered,
        "LogCentered": normalization_log_centered,
        "InverseDistCentered": normalization_inverse_dist_centered,
        "InverseSqrtDistCentered": normalization_inverse_sqrt_dist_centered
    }
    return switcher.get(name)


def nerf_get_normalization_function_abbr(name):
    switcher = {
        None: "",
        "None": "_nN",
        "Centered": "_nC",
        "MaxDepth": "",
        "MaxDepthCentered": "_nMdC", 
        "LogCentered": "_nL",
        "InverseDistCentered": "_nD",
        "InverseSqrtDistCentered": "_nSD"
    }
    return switcher.get(name)


class LinearlySpacedZNearZFarNoDepthRange:
    """
    This just samples linearly between z_near and z_far without anything else.
    """
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.z_step = z_step
        self.num_ray_samples = num_ray_samples
        self.noise_amplitude = noise_amplitude
        if self.noise_amplitude > 0.0:
            self.print_name = f"{self.z_near}_{self.z_far}_{self.num_ray_samples}_{self.__class__.__name__}_{self.z_step}_{self.noise_amplitude}"
        else:
            self.print_name = f"{self.z_near}_{self.z_far}_{self.num_ray_samples}_{self.__class__.__name__}"

    def generate(self, idx, device, **kwargs):
        det = kwargs.get('det', True)
        t_vals = torch.linspace(0., 1., steps=int(self.num_ray_samples + 1), device=device)[0:-1] + (0.5 / self.num_ray_samples)
        near_vec = torch.ones((idx, 1), device=device) * self.z_near  # [n_images * n_samples, 1]
        far_vec = torch.ones((idx, 1), device=device) * self.z_far  # [n_images * n_samples, 1]
        z_vals = near_vec * (1. - t_vals) + far_vec * t_vals
        noise_add = (-self.z_step / 2 + self.z_step * torch.rand_like(z_vals))

        if det:
            self.noise_amplitude = 0

        z_vals_noise = z_vals + self.noise_amplitude * noise_add

        return z_vals_noise

    def get_name(self):
        return self.print_name


class LinearlySpacedZNearZFar:
    """
    This just samples linearly between z_near and z_far without anything else.
    """
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.z_step = z_step
        self.num_ray_samples = num_ray_samples
        self.noise_amplitude = noise_amplitude
        if self.noise_amplitude > 0.0:
            self.print_name = f"{self.z_near}_{self.z_far}_{self.num_ray_samples}_{self.__class__.__name__}_{self.z_step}_{self.noise_amplitude}"
        else:
            self.print_name = f"{self.z_near}_{self.z_far}_{self.num_ray_samples}_{self.__class__.__name__}"

    def generate(self, idx, device, **kwargs):
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        det = kwargs.get('det', True)

        t_vals = torch.linspace(0., 1., steps=int(self.num_ray_samples + 1), device=device)[0:-1] + (0.5 / self.num_ray_samples)
        near_vec = torch.ones((idx, 1), device=device) * self.z_near  # [n_images * n_samples, 1]
        far_vec = torch.ones((idx, 1), device=device) * self.z_far  # [n_images * n_samples, 1]
        z_vals = near_vec * (1. - t_vals) + far_vec * t_vals
        noise_add = (-self.z_step / 2 + self.z_step * torch.rand_like(z_vals))

        if det:
            self.noise_amplitude = 0

        z_vals_noise = z_vals + self.noise_amplitude * noise_add

        return depth_transform.to_world(z_vals_noise, depth_range)

    def get_name(self):
        return self.print_name


class UnitSphereLinearOutsideLog:
    """
    This samples linearly within the unit sphere, and logarithmically outside, using half the number of samples for each.
    """
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.z_step = z_step
        self.num_ray_samples = num_ray_samples
        self.noise_amplitude = noise_amplitude
        if self.noise_amplitude > 0.0:
            self.print_name = f"{self.z_near}_{self.z_far}_{self.num_ray_samples}_{self.__class__.__name__}_{self.z_step}_{self.noise_amplitude}"
        else:
            self.print_name = f"{self.z_near}_{self.z_far}_{self.num_ray_samples}_{self.__class__.__name__}"

    def generate(self, idx, device, **kwargs):
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        det = kwargs.get('det', True)
        ray_origins = kwargs.get('ray_origins', None)
        ray_directions = kwargs.get('ray_directions', None)

        # TODO:
        # intersect with unit sphere
        # create n_samples / 2 z vals between [0, 1] and transform them to [depth_range[0], intersection t (- epsilon)]
        # create n_samples / 2 z vals between [0, 1] and transform them to [intersection t (+ epsilon), depth_range[1]]
        # then we can concatenate both t value ranges and should be fine, as they are already in world space.

        # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        # assumption: unit sphere at origin (centered at [0, 0, 0], radius 1)

        u = ray_directions
        o = ray_origins
        u_dot_o = torch.sum(o * u.flatten().reshape(-1, 3), dim=1)
        delta = (u_dot_o ** 2) - ((torch.sum(o ** 2, dim=-1)) - 1)

        # We can assume that delta will always have an intersection, and just need to find the one with positive t
        t_intersection_1 = -u_dot_o + torch.sqrt(delta)
        t_intersection_2 = -u_dot_o - torch.sqrt(delta)
        t_intersection = torch.maximum(t_intersection_1, t_intersection_2)
        n_samples_per_range = self.num_ray_samples // 2

        t_intersection = tile(t_intersection, dim=0, n_tile=n_samples_per_range).reshape(-1, n_samples_per_range)
        # Generate the first sample range
        t_vals_inside = torch.linspace(0., 1., steps=int(n_samples_per_range + 1), device=device)[0:-1] + (0.5 / (n_samples_per_range))
        near_vec = torch.ones((idx, 1), device=device) * self.z_near  # [n_images * n_samples, 1]
        far_vec = torch.ones((idx, 1), device=device) * self.z_far  # [n_images * n_samples, 1]
        t_vals_inside = near_vec * (1. - t_vals_inside) + far_vec * t_vals_inside
        #noise_add = (-self.z_step / 2 + self.z_step * torch.rand_like(z_vals))

        #if det:
        #    self.noise_amplitude = 0

        #z_vals_noise = z_vals + self.noise_amplitude * noise_add

        z_vals_inside = depth_transforms.LinearTransform.to_world(t_vals_inside, [torch.ones_like(t_vals_inside) * depth_range[0], t_intersection])

        # Generate the second sample range
        t_vals_outside = torch.linspace(0. + (0.5 / (n_samples_per_range)), 1., steps=int(n_samples_per_range + 1),
                                       device=device)[0:-1] + (0.5 / (n_samples_per_range))
        near_vec = torch.ones((idx, 1), device=device) * 0 # [n_images * n_samples, 1]
        far_vec = torch.ones((idx, 1), device=device) * self.z_far  # [n_images * n_samples, 1]
        t_vals_outside = near_vec * (1. - t_vals_outside) + far_vec * t_vals_outside

        z_vals_outside = depth_transforms.LogTransform.to_world(t_vals_outside, [t_intersection, torch.ones_like(t_vals_inside) * depth_range[1]])

        return torch.cat((z_vals_inside, z_vals_outside), dim=1)

    def get_name(self):
        return self.print_name


class LinearlySpacedFromDepthNoDepthRange:
    """
    This just samples linearly between new z_near (computed from depth and original z_near/z_far spacing) and new z_far
    """
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.num_ray_samples = num_ray_samples
        self.z_step = z_step
        self.noise_amplitude = noise_amplitude
        self.print_name = f"{self.z_near}_{self.z_far}_{self.num_ray_samples}_{self.__class__.__name__}_{self.z_step}_{self.noise_amplitude}"

    def generate(self, idx, device, **kwargs):
        depth = kwargs.get('depth', None)
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        z_step = self.z_step
        noise = self.noise_amplitude

        # Add noise from -z_step/2 to +z_step/2, scaled by noise factor
        noise_add = noise * (-z_step / 2 + z_step * torch.rand_like(depth))

        depth = depth.detach()
        s_depth = depth + noise_add

        z_near = s_depth - z_step * np.floor(self.num_ray_samples / 2)

        z_vals = (z_near[..., None] + torch.linspace(0, z_step * (self.num_ray_samples - 1), int(self.num_ray_samples),
                                                     device=device, dtype=torch.float32)).reshape(
            idx, self.num_ray_samples)  # [n_images * n_samples, num_ray_samples]

        return z_vals

    def get_name(self):
        return self.print_name

class LinearlySpacedFromDepth:
    """
    This just samples linearly between new z_near (computed from depth and original z_near/z_far spacing) and new z_far
    """
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.num_ray_samples = num_ray_samples
        self.z_step = z_step
        self.noise_amplitude = noise_amplitude
        self.print_name = f"{self.z_near}_{self.z_far}_{self.num_ray_samples}_{self.__class__.__name__}_{self.z_step}_{self.noise_amplitude}"

    def generate(self, idx, device, **kwargs):
        depth = kwargs.get('depth', None)
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        z_step = self.z_step
        noise = self.noise_amplitude

        # Add noise from -z_step/2 to +z_step/2, scaled by noise factor
        noise_add = noise * (-z_step / 2 + z_step * torch.rand_like(depth))

        depth = depth.detach()
        s_depth = depth + noise_add

        z_near = s_depth - z_step * np.floor(self.num_ray_samples / 2)

        z_vals = (z_near[..., None] + torch.linspace(0, z_step * (self.num_ray_samples - 1), int(self.num_ray_samples),
                                                     device=device, dtype=torch.float32)).reshape(
            idx, self.num_ray_samples)  # [n_images * n_samples, num_ray_samples]

        return depth_transform.to_world(z_vals, depth_range)

    def get_name(self):
        return self.print_name

class FromDepthCells:
    """
    This just samples linearly between new z_near (computed from depth and original z_near/z_far spacing) and new z_far
    """
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, config=None, net_idx=-1, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.num_ray_samples = num_ray_samples
        self.z_step = z_step
        self.noise_amplitude = noise_amplitude
        self.disc = 128
        if config.multiDepthFeatures:
            self.disc = config.multiDepthFeatures[net_idx]
        self.print_name = f"fDC_{self.num_ray_samples}_{self.__class__.__name__}_{self.z_step}_{self.noise_amplitude}"

    def generate(self, idx, device, **kwargs):
        depth = kwargs.get('depth', None)
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        z_step = self.z_step
        noise = self.noise_amplitude

        # Add noise from -z_step/2 to +z_step/2, scaled by noise factor
        noise_add = noise * (-z_step / 2 + z_step * torch.rand_like(depth))

        depth = depth.detach()
        # Discretize depth
        depth = (torch.floor(depth * self.disc) + 0.5) / self.disc

        # Get the samples uniformly spaced around this depth value
        s_depth = depth + noise_add
        z_near = s_depth - z_step * np.floor(self.num_ray_samples / 2)



        z_vals = (z_near[..., None] + torch.linspace(0, z_step * (self.num_ray_samples - 1), int(self.num_ray_samples),
                                                     device=device, dtype=torch.float32)).reshape(
            idx, self.num_ray_samples)  # [n_images * n_samples, num_ray_samples]

        return depth_transform.to_world(z_vals, depth_range)

    def get_name(self):
        return self.print_name


class LinearlySpacedFromMultiDepth:
    """
    This just samples linearly around multiple reference points
    """
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, config=None, net_idx=-1, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.num_ray_samples = num_ray_samples
        self.z_step = z_step
        self.noise_amplitude = noise_amplitude
        self.net_idx = net_idx
        self.background_value = config.multiDepthIgnoreValue[net_idx]
        self.print_name = f"{self.z_near}_{self.z_far}_{self.num_ray_samples}_LSfMD_{self.z_step}_{self.noise_amplitude}"

    def generate(self, idx, device, **kwargs):
        depth = kwargs.get('depth', None)
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        z_step = self.z_step
        noise = self.noise_amplitude

        sorted_depth, ids = torch.sort(depth)
        sorted_depth = torch.clamp(sorted_depth, min=0., max=1.)

        # Add noise from -z_step/2 to +z_step/2, scaled by noise factor
        noise_add = noise * (-z_step / 2 + z_step * torch.rand_like(sorted_depth))

        sorted_depth = sorted_depth + noise_add

        starting_points = depth.shape[-1]
        samples_per_point = (self.num_ray_samples+starting_points-1) // starting_points

        z_nears = sorted_depth - z_step * samples_per_point / 2
        
        # ensure samples are z_step * (samples_per_point + 1) apart
        mind_dist = z_step * (samples_per_point + 1)
        for i in range(starting_points-1):
            dist = z_nears[:, starting_points - i - 1] - z_nears[:, starting_points - i - 2]
            off = torch.clamp(dist - mind_dist, max=0)
            z_nears[:, starting_points - i - 2] += off

        z_nears_base = torch.repeat_interleave(z_nears, samples_per_point, dim=1)

        steps = torch.linspace(0, z_step * samples_per_point, samples_per_point,
                               device=device, dtype=torch.float32)
        steps_repeated = steps.repeat(z_nears_base.shape[0], starting_points)

        z_vals = (z_nears_base + steps_repeated).reshape(
            idx, starting_points * samples_per_point)  # [n_images * n_samples, starting_points * samples_per_point]

        return depth_transform.to_world(z_vals, depth_range)

    def get_name(self):
        return self.print_name


class FromIterativeSamplePlacement:
    """
    This takes the iterative sample placement thing and places samples in the center of cells if the cell is active.
    """

    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.num_ray_samples = num_ray_samples
        self.z_step = z_step
        self.noise_amplitude = noise_amplitude
        self.print_name = f"Iter_{self.z_near}_{self.z_far}_{self.num_ray_samples}_{self.__class__.__name__}_{self.z_step}_{self.noise_amplitude}"

    def generate(self, idx, device, **kwargs):
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        sample_placement = kwargs.get('sample_placement', None)
        num_ray_samples = kwargs.get('num_ray_samples', None)
        t_vals = torch.linspace(0., 1., steps=int(sample_placement.shape[-1] + 1), device=device)[0:-1]

        # Use the sample placement booleans to get the corrent z_vals for sampling
        z_vals = ((t_vals + (1.0 / 128) * 0.5) * sample_placement)[sample_placement == 1].reshape(-1, num_ray_samples)

        return depth_transform.to_world(z_vals, depth_range)

    def get_name(self):
        return self.print_name


class FromClassifiedDepth:
    """
    This just samples linearly around multiple reference points
    """
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, config=None, net_idx=-1, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.num_ray_samples = num_ray_samples
        self.z_step = z_step
        self.noise_amplitude = noise_amplitude
        self.net_idx = net_idx
        self.background_value = config.multiDepthIgnoreValue[net_idx]

        self.disc = 128
        if config.multiDepthFeatures:
            self.disc = config.multiDepthFeatures[net_idx]
        self.print_name = f"{self.num_ray_samples}_LSfCD_{self.disc}_{self.noise_amplitude}"
        self.transform = None

        if self.net_idx > 0:
            if config.losses[self.net_idx-1] == "BCEWithLogitsLoss":
                self.transform = torch.sigmoid
            elif config.losses[self.net_idx-1] == "CrossEntropyLoss":
                self.transform = self.softmax
            elif config.losses[self.net_idx-1] == "CrossEntropyLossWeighted":
                self.transform = self.softmaxselect

    def softmax(self, depth):
        return torch.nn.functional.softmax(depth, dim=-1)

    def softmaxselect(self, depth):
        return torch.nn.functional.softmax(depth[..., :self.disc], dim=-1)

    def generate(self, idx, device, **kwargs):
        depth = kwargs.get('depth', None)
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        det = kwargs.get('deterministic_sampling', True)
        depth = depth.detach()

        if self.transform:
            depth = self.transform(depth)

        disc_steps = depth.shape[-1]

        mids_single = torch.linspace(0., 1., disc_steps+1, device=device, dtype=torch.float32)
        mids_all = mids_single.repeat(depth.shape[0], 1)

        z_samples = nerf_sample_pdf(mids_all, depth, self.num_ray_samples+2, det=det)
        z_samples = z_samples[:,1:-1]
        z_samples = z_samples.detach()
        return depth_transform.to_world(z_samples, depth_range)

    def get_name(self):
        return self.print_name


class FromClassifiedDepthAdaptive:
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, config=None, net_idx=-1, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.num_ray_samples = num_ray_samples
        self.z_step = z_step
        self.noise_amplitude = noise_amplitude
        self.net_idx = net_idx
        self.background_value = config.multiDepthIgnoreValue[net_idx]
        self.max_samples_per_ray = num_ray_samples

        self.disc = 128
        if config.multiDepthFeatures:
            self.disc = config.multiDepthFeatures[net_idx]

        self.threshold = config.adaptiveSamplingThreshold
        self.print_name = f"{self.num_ray_samples}_LSfCDA_({self.threshold})_{self.disc}_{self.noise_amplitude}"
        self.transform = None

        self.near_vec = None
        self.far_vec = None

        if self.net_idx > 0:
            if config.losses[self.net_idx-1] == "BCEWithLogitsLoss":
                self.transform = torch.sigmoid
            elif config.losses[self.net_idx-1] == "CrossEntropyLoss":
                self.transform = self.softmax
            elif config.losses[self.net_idx-1] == "CrossEntropyLossWeighted":
                self.transform = self.softmaxselect

    def softmax(self, depth):
        return torch.nn.functional.softmax(depth, dim=-1)

    def softmaxselect(self, depth):
        return torch.nn.functional.softmax(depth[..., :self.disc], dim=-1)

    def generate(self, idx, device, **kwargs):
        depth = kwargs.get('depth', None)
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        depth = depth.detach()

        if self.transform:
            depth = self.transform(depth)

        if self.threshold == 0.0:
            depth_range = kwargs.get('depth_range', None)
            depth_transform = kwargs.get('depth_transform', None)
            t_vals = torch.linspace(0., 1., steps=int(self.num_ray_samples + 1), device=device)[0:-1] + (
                        0.5 / self.num_ray_samples)

            if self.near_vec is None or self.near_vec.shape[0] != idx:
                self.near_vec = torch.ones((idx, 1), device=device) * self.z_near  # [n_images * n_samples, 1]
                self.far_vec = torch.ones((idx, 1), device=device) * self.z_far  # [n_images * n_samples, 1]

            z_vals = self.near_vec * (1. - t_vals) + self.far_vec * t_vals

            return depth_transform.to_world(z_vals, depth_range)

        disc_steps = self.disc
        cell_size = 1.0 / disc_steps
        infinity = float('inf')

        samples, desc_sort_indices = torch.sort(depth, dim=1, descending=True)

        test_samples = torch.zeros(samples.size(), device=desc_sort_indices.device)
        test_samples[samples >= self.threshold] = 1

        # number of samples per ray
        num_samples = torch.sum(test_samples, 1)

        max_samples = self.max_samples_per_ray

        # now the correct cell and the offset to the middle of the cell is set
        # NOTE: we also multiply the offset by 1 or 0, so empty cells still remain at 0
        z_samples = (test_samples[:, :max_samples] * desc_sort_indices[:, :max_samples]) + \
                    (test_samples[:, :max_samples] * .5)

        # now we have the correct depth values (ordered by probability)
        z_samples = z_samples * cell_size

        # this should give us the probabilities per sample
        z_probs = (test_samples[:, :max_samples] * samples[:, :max_samples])

        # fill empty rows with one sample
        z_samples[num_samples == 0, 0] = (desc_sort_indices[num_samples == 0, 0] + 0.5) * cell_size
        z_probs[num_samples == 0, 0] = samples[num_samples == 0, 0]

        # to differentiate inactive cells, we set them to infinity
        z_samples[z_samples == 0] = infinity

        z_samples, z_sort_indices = torch.sort(z_samples, dim=1)
        z_probs = torch.gather(z_probs, 1, z_sort_indices)

        return depth_transform.to_world(z_samples, depth_range), z_probs

    def get_name(self):
        return self.print_name


class FromClassifiedDepthAdaptiveNoDepthRange:
    def __init__(self, z_near, z_far, num_ray_samples, z_step, noise_amplitude, config=None, net_idx=-1, **kwargs):
        self.z_near = z_near
        self.z_far = z_far
        self.num_ray_samples = num_ray_samples
        self.z_step = z_step
        self.noise_amplitude = noise_amplitude
        self.net_idx = net_idx
        self.background_value = config.multiDepthIgnoreValue[net_idx]
        self.max_samples_per_ray = num_ray_samples

        self.disc = 128
        if config.multiDepthFeatures:
            self.disc = config.multiDepthFeatures[net_idx]

        self.threshold = config.adaptiveSamplingThreshold
        self.print_name = f"{self.num_ray_samples}_LSfCDA_({self.threshold})_{self.disc}_{self.noise_amplitude}"
        self.transform = None

        if self.net_idx > 0:
            if config.losses[self.net_idx-1] == "BCEWithLogitsLoss":
                self.transform = torch.sigmoid
            elif config.losses[self.net_idx-1] == "CrossEntropyLoss":
                self.transform = self.softmax
            elif config.losses[self.net_idx-1] == "CrossEntropyLossWeighted":
                self.transform = self.softmaxselect

    def softmax(self, depth):
        return torch.nn.functional.softmax(depth, dim=-1)

    def softmaxselect(self, depth):
        return torch.nn.functional.softmax(depth[..., :self.disc], dim=-1)

    def generate(self, idx, device, **kwargs):
        depth = kwargs.get('depth', None)
        depth_range = kwargs.get('depth_range', None)
        depth_transform = kwargs.get('depth_transform', None)
        depth = depth.detach()

        if self.transform:
            depth = self.transform(depth)

        if self.threshold == 0.0:
            depth_range = kwargs.get('depth_range', None)
            depth_transform = kwargs.get('depth_transform', None)
            t_vals = torch.linspace(0., 1., steps=int(self.num_ray_samples + 1), device=device)[0:-1] + (
                        0.5 / self.num_ray_samples)
            near_vec = torch.ones((idx, 1), device=device) * self.z_near  # [n_images * n_samples, 1]
            far_vec = torch.ones((idx, 1), device=device) * self.z_far  # [n_images * n_samples, 1]
            z_vals = near_vec * (1. - t_vals) + far_vec * t_vals

            return z_vals

        disc_steps = self.disc
        cell_size = 1.0 / disc_steps
        infinity = float('inf')

        samples, desc_sort_indices = torch.sort(depth, dim=1, descending=True)

        test_samples = torch.zeros(samples.size(), device=desc_sort_indices.device)
        test_samples[samples >= self.threshold] = 1

        # number of samples per ray
        num_samples = torch.sum(test_samples, 1)

        max_samples = self.max_samples_per_ray

        # now the correct cell and the offset to the middle of the cell is set
        # NOTE: we also multiply the offset by 1 or 0, so empty cells still remain at 0
        z_samples = (test_samples[:, :max_samples] * desc_sort_indices[:, :max_samples]) + \
                    (test_samples[:, :max_samples] * .5)

        # now we have the correct depth values (ordered by probability)
        z_samples = z_samples * cell_size

        # this should give us the probabilities per sample
        z_probs = (test_samples[:, :max_samples] * samples[:, :max_samples])

        # fill empty rows with one sample
        z_samples[num_samples == 0, 0] = (desc_sort_indices[num_samples == 0, 0] + 0.5) * cell_size
        z_probs[num_samples == 0, 0] = samples.to(torch.float32)[num_samples == 0, 0]

        # to differentiate inactive cells, we set them to infinity
        z_samples[z_samples == 0] = infinity

        z_samples, z_sort_indices = torch.sort(z_samples, dim=1)
        z_probs = torch.gather(z_probs, 1, z_sort_indices)

        return z_samples, z_probs

    def get_name(self):
        return self.print_name
