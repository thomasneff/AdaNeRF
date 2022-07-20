# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch

from features import FeatureSetKeyConstants

class LimitedDepthMSELoss(torch.nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str='mean', config=None, net_idx=-1) -> None:
        super(LimitedDepthMSELoss, self).__init__()
        self.reduction = reduction
        self.mseLoss = torch.nn.MSELoss()
        self.ignore_value = config.multiDepthIgnoreValue[net_idx]

    def forward(self, outputs: torch.Tensor, targets : torch.Tensor) -> torch.Tensor:
        seltargets = torch.where(targets.data < self.ignore_value, targets.data, outputs.data)
        return self.mseLoss(outputs, seltargets)


class MultiDepthLimitedMSELoss(torch.nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str='mean', config=None, net_idx=-1) -> None:
        super(MultiDepthLimitedMSELoss, self).__init__()
        self.reduction = reduction
        self.mseLoss = torch.nn.MSELoss()
        self.ignore_value = config.multiDepthIgnoreValue[net_idx]

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # copy outputs
        outputs_cpy = outputs.clone()
        resort_indices = torch.zeros_like(outputs, dtype=torch.int64)

        # compute minimum distance from predictions to target
        # rep_targets = torch.repeat_interleave(targets, targets.shape[-1], dim=-1)
        for i in range(targets.shape[-1]):
            # sel_targets = rep_targets[:,3*i:3*(i+1)]
            sel_targets = torch.reshape(targets[..., i], (targets.shape[0], 1)).repeat(1, targets.shape[-1])
            diff2 = torch.abs(outputs_cpy - sel_targets)
            ids = torch.argmin(diff2, -1)
            outputs_cpy = outputs_cpy.scatter_(1, torch.reshape(ids, (targets.shape[0], 1)), torch.finfo(outputs.dtype).max)
            # outputs_cpy[..., ids] = torch.finfo(outputs.dtype).max
            resort_indices[..., i] = ids
        outputs_shfled = torch.gather(outputs, 1, resort_indices)

        seltargets = torch.where(targets.data != self.ignore_value, targets.data, outputs_shfled.data)
        return self.mseLoss(outputs_shfled, seltargets)


class MSEPlusWeightAccum(torch.nn.Module):

    def __init__(self, config=None, net_idx=-1) -> None:
        super(MSEPlusWeightAccum, self).__init__()
        self.mseLoss = torch.nn.MSELoss()
        self.asymmetric = True
        self.loss_alpha = config.lossAlpha[net_idx]
        self.loss_beta = config.lossBeta[net_idx]
        self.requires_alpha_beta = True

    def forward(self, outputs: torch.Tensor, targets : torch.Tensor, **kwargs) -> torch.Tensor:
        inference_dict = kwargs.get('inference_dict', None)

        if inference_dict is None:
            raise Exception(f"MSEPlusWeightAccum requires inference_dict argument!")

        weights_sum = torch.sum(inference_dict[FeatureSetKeyConstants.nerf_weights_output], axis=1)

        loss_mse = self.mseLoss(outputs, targets)
        alpha = self.loss_alpha
        beta = self.loss_beta

        # The weights should sum to >= 1.0

        if self.asymmetric:
            weights_sum[weights_sum > 1.0] = 1.0

        loss_weights = self.mseLoss(weights_sum, torch.ones_like(weights_sum))

        return alpha * loss_mse + beta * loss_weights


class NeRFWeightMultiplicationLoss(torch.nn.Module):
    def __init__(self, config=None, net_idx=-1) -> None:
        super(NeRFWeightMultiplicationLoss, self).__init__()
        self.config = config
        self.net_idx = net_idx
        self.l1Loss = torch.nn.L1Loss()
        self.written = False

        self.ones = None
        self.zeros = None

        self.loss_components = config.lossComponents
        self.blend_factors = config.lossComponentBlending

        name = ""
        for i in range(len(config.lossComponents)):
            name = f"{config.lossComponents[i]}_{config.lossComponentBlending[i]}" if i == 0 else \
                f"{name}_{config.lossComponents[i]}_{config.lossComponentBlending[i]}"

        self.weight = config.lossWeights[net_idx]

        self.blending_start = config.lossBlendingStart
        self.blending_interval = config.lossBlendingDuration

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        epoch = kwargs.get("epoch", -1)
        inf_dict = kwargs.get("inference_dict", None)

        nerf_weights = inf_dict[self.net_idx + 1][FeatureSetKeyConstants.nerf_weights_output]
        nerf_alpha = inf_dict[self.net_idx + 1][FeatureSetKeyConstants.nerf_alpha_output]

        final_loss = None

        factor = min(max((epoch - self.blending_start) / self.blending_interval, 0), 1)

        for i, loss_name in enumerate(self.loss_components):
            blend_factor = None
            loss = None

            if loss_name == "One":
                self.ones = torch.ones_like(outputs)  # shape can be different

                loss = self.l1Loss(outputs, self.ones)
                blend_factor = 1.0 - factor * (1.0 - self.blend_factors[i]) if self.blend_factors[i] > 0.0 \
                    else 1.0 - factor

            elif loss_name == "Zero":
                self.zeros = torch.zeros_like(outputs)

                loss = self.l1Loss(outputs, self.zeros)
                blend_factor = factor * self.blend_factors[i] if self.blend_factors[i] > 0.0 else factor

            elif loss_name == "NerfW":
                loss = self.l1Loss(outputs, nerf_weights)
                blend_factor = factor * self.blend_factors[i] if self.blend_factors[i] > 0.0 else factor

            elif loss_name == "NerfA":
                loss = self.l1Loss(outputs, nerf_alpha)
                blend_factor = factor * self.blend_factors[i] if self.blend_factors[i] > 0.0 else factor


            if final_loss is None:
                final_loss = blend_factor * loss
            else:
                final_loss = final_loss + blend_factor * loss

        return final_loss


class DefaultLossWrapper(torch.nn.Module):
    def __init__(self, loss_func, config, net_idx) -> None:
        super(DefaultLossWrapper, self).__init__()
        self.loss_func = loss_func

        name = ""
        for i in range(len(config.lossComponents)):
            name = f"{config.lossComponents[i]}_{config.lossComponentBlending[i]}" if i == 0 else\
                f"{name}_{config.lossComponents[i]}_{config.lossComponentBlending[i]}"
        
        self.weight = config.lossWeights[net_idx]

    def forward(self, outputs: torch.Tensor, targets : torch.Tensor, **kwargs) -> torch.Tensor:
        loss_value = self.loss_func(outputs, targets)

        epoch = kwargs.get("epoch", -1)

        return loss_value


def get_loss_by_name(name, config, net_idx):
    if name == "MSE":
        return DefaultLossWrapper(torch.nn.MSELoss(), config, net_idx)
    if name == "LimitedDepthMSE":
        return DefaultLossWrapper(LimitedDepthMSELoss(config=config, net_idx=net_idx))
    if name == "MultiDepthLimitedMSE":
        return DefaultLossWrapper(MultiDepthLimitedMSELoss(config=config, net_idx=net_idx))
    if name == "MSEPlusWeightAccum":
        return MSEPlusWeightAccum(config=config, net_idx=net_idx)
    if name == "BCEWithLogitsLoss":
        return DefaultLossWrapper(torch.nn.BCEWithLogitsLoss(), config, net_idx)
    if name == "CrossEntropyLoss":
        return DefaultLossWrapper(torch.nn.CrossEntropyLoss(), config, net_idx)
    if name == "CrossEntropyLossWeighted":
        weights = torch.ones(config.multiDepthFeatures[net_idx] + 1, dtype=torch.float32).cuda(device=config.device)
        weights[-1] = 0.
        return DefaultLossWrapper(torch.nn.CrossEntropyLoss(weight=weights), config, net_idx)
    if name == "NeRFWeightMultiplicationLoss":
        return NeRFWeightMultiplicationLoss(config=config, net_idx=net_idx)
    if name.lower() == "none":
        return None
    else:
        raise Exception(f"Loss {name} unknown")
