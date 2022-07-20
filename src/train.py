# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import sys
import csv
import math
import shutil

import numpy as np

from tqdm import tqdm
from util.config import Config
from datasets import create_sample_wrapper
from train_data import TrainConfig
from features import FeatureSetKeyConstants
from plots import render_img, render_video, plot_training_stats
from util.saveimage import save_img, transform_img

import matplotlib
matplotlib.use('Agg')


def validate_batch(train_config: TrainConfig, epoch, train_loss, model_idx=-1):
    accuracies = []
    losses = []
    c_file = train_config.config_file

    val_data_set, _ = train_config.get_data_set_and_loader('val')
    val_data_set.full_images = True

    inference_chunk_size = train_config.config_file.inferenceChunkSize
    dim_w = train_config.dataset_info.w
    dim_h = train_config.dataset_info.h

    validation_images = []

    for i, sample_data in enumerate(tqdm(val_data_set, desc='validating batch', position=0, leave=True)):
        # get sample input data
        sample_data_wrapper = create_sample_wrapper(sample_data, train_config, True)

        inference_imgs = []
        target_img = None
        inference_dict_full = {}

        start_index = 0
        for batch in sample_data_wrapper.batches(inference_chunk_size):
            # inference
            if model_idx == -1:
                outs, inference_dicts = train_config.inference(batch, gradient=False)
                inference_dict = inference_dicts[-1]
            else:
                f_in = train_config.f_in[model_idx]
                model = train_config.models[model_idx]

                prev_outs = []
                for prev_model_idx in range(0, model_idx):
                    prev_outs.append(batch.get_train_target(prev_model_idx))

                inference_dict = f_in.batch(batch.get_batch_input(model_idx), prev_outs=prev_outs)
                x_batch = inference_dict[FeatureSetKeyConstants.input_feature_batch]
                with torch.no_grad():
                    outs = [model(x_batch)]

            if len(inference_imgs) == 0:
                for net_idx in range(len(outs)):
                    inference_imgs.append(torch.zeros((dim_h * dim_w, outs[net_idx].shape[-1]), dtype=torch.float32,
                                                      device=outs[net_idx].device))

                target_img = torch.zeros((dim_h * dim_w, batch.get_train_target(model_idx).shape[-1]),
                                         dtype=torch.float32, device=outs[-1].device)

                for key, value in inference_dict.items():
                    if inference_dict[key] is not None and inference_dict[key].ndim == 2:
                        inference_dict_full[key] = torch.zeros((dim_h * dim_w, value.shape[-1]), device="cpu",
                                                               dtype=torch.float32)

            end_index = min(start_index + train_config.config_file.inferenceChunkSize, dim_w * dim_h)

            for net_idx in range(len(outs)):
                inference_imgs[net_idx][start_index:end_index, :] = outs[net_idx][:inference_chunk_size]

            target_img[start_index:end_index, :] = batch.get_train_target(model_idx)[:inference_chunk_size, :]

            for key, value in inference_dict.items():
                if inference_dict[key] is not None and inference_dict[key].ndim == 2:
                    inference_dict_full[key][start_index:end_index] = (value[:end_index - start_index])

            start_index = end_index

        loss_batch = train_config.losses[model_idx](inference_imgs[-1], target_img, inference_dict=inference_dict_full)
        losses.append(loss_batch)
        diff = abs(inference_imgs[-1] - target_img)
        accuracy = float((diff < 0.001).sum()) / float(diff.shape[0] * diff.shape[1])
        accuracies.append(accuracy)

        # we calculate all the data here, so if we get a better validation loss, we can simply save all these images
        mse = torch.mean(diff ** 2)
        psnr = 10 * torch.log10(1. / mse)
        validation_data = {}
        transformed_images = []

        for net_idx, img in enumerate(inference_imgs):
            transformed_images.append(transform_img(img, train_config.dataset_info))

        validation_data["images"] = transformed_images
        validation_data["psnr"] = psnr.cpu().numpy()
        validation_images.append(validation_data)

    loss = torch.mean(torch.tensor(losses)).item()
    accuracy = torch.mean(torch.tensor(accuracies)).item()

    val_data_set.full_images = False

    print(f"\nvalidation epoch={epoch:<10} loss={loss:.8f} acc={accuracy:.8f}")
    with open(f"{train_config.logDir}logs.txt", "a") as f:
        f.write(f"epoch={epoch} loss={loss:.4f}  acc={accuracy:.8f} train_loss={train_loss:.8f}\r")
    add_header = False
    if not os.path.isfile(f"{train_config.logDir}{c_file.trainStatsName}"):
        add_header = True
    with open(f"{train_config.logDir}{c_file.trainStatsName}", 'a', newline='') as csv_file:
        fieldnames = ['epoch', 'loss', 'accuracy', 'train_loss']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if add_header:
            writer.writeheader()

        writer.writerow({'epoch': f'{epoch}', 'loss': f'{loss}', 'accuracy': f'{accuracy}',
                         'train_loss': f'{train_loss}'})

    plot_training_stats(train_config.logDir, c_file.trainStatsName, 'epoch', 'loss')
    plot_training_stats(train_config.logDir, c_file.trainStatsName, 'epoch', 'train_loss')
    plot_training_stats(train_config.logDir, c_file.trainStatsName, 'epoch', 'accuracy')
    plot_training_stats(train_config.logDir, c_file.trainStatsName, 'epoch', ['loss', 'train_loss', 'accuracy'])
    plot_training_stats(train_config.logDir, c_file.trainStatsName, 'epoch', ['loss', 'train_loss'])

    return loss, validation_images


def pre_train(train_config):
    if train_config.config_file.epochsPretrain is None or len(train_config.config_file.epochsPretrain) == 0:
        return

    c_file = train_config.config_file

    decay_rate = c_file.lrate_decay
    decay_steps = c_file.lrate_decay_steps

    batch_images = c_file.batchImages
    samples = c_file.samples

    if c_file.batchImagesPretrain != -1:
        batch_images = c_file.batchImagesPretrain
    if c_file.samplesPretrain != -1:
        samples = c_file.samplesPretrain

    train_config.train_dataset.num_samples = samples

    # pretrain data_loader can have different batch size
    data_loader = train_config.train_data_loader
    if train_config.pretrain_data_loader is not None:
        data_loader = train_config.pretrain_data_loader

    for model_idx in range(len(train_config.models)):
        epoch_pretrain = train_config.config_file.epochsPretrain[model_idx]
        if train_config.epoch0 >= epoch_pretrain:
            continue

        best_val_loss = sys.float_info.max
        if model_idx < len(train_config.best_valid_loss_pretrain):
            best_val_loss = train_config.best_valid_loss_pretrain[model_idx]

        model = train_config.models[model_idx]
        optim = train_config.optimizers[model_idx]
        criterion = train_config.losses[model_idx]
        f_in = train_config.f_in[model_idx]

        batch_iterator = iter(data_loader)
        num_batches = int(math.ceil(len(train_config.train_dataset) / batch_images))

        tqdm_range = tqdm(range(train_config.epoch0, epoch_pretrain + 1), desc=f"pre-training model {model_idx}")

        for epoch in tqdm_range:
            optim.zero_grad()

            # get sample input data
            batch_samples = next(batch_iterator)
            sample_data = create_sample_wrapper(batch_samples, train_config)

            # we create a new iterator once all elements have been exhausted
            # we get a slight performance bump by creating the iterator here and not in the beginning of the next loop
            if epoch % num_batches == 0:
                batch_iterator = iter(train_config.train_data_loader)

            prev_outs = []
            for prev_model_idx in range(model_idx):
                prev_outs.append(sample_data.get_train_target(prev_model_idx))

            inference_dict = f_in.batch(sample_data.get_batch_input(model_idx), prev_outs=prev_outs)
            x_batch = inference_dict[FeatureSetKeyConstants.input_feature_batch]
            y_batch = sample_data.get_train_target(model_idx)

            if y_batch is not None:
                y_batch = y_batch.reshape(y_batch.shape[0] * samples, -1)

            out = model(x_batch)

            loss = criterion(out, y_batch, inference_dict=inference_dict)
            loss.backward()
            optim.step()

            # Learning rate decay
            new_lrate = c_file.lrate * (decay_rate ** ((train_config.epoch0 + epoch) / decay_steps))
            for param_group in optim.param_groups:
                param_group['lr'] = new_lrate

            if not c_file.nonVerbose:
                tqdm_range.set_description(f"epoch={epoch:<10} loss={loss:.8f}")

            if epoch > 0 and epoch % train_config.config_file.epochsCheckpoint == 0:
                train_config.save_weights(name_suffix=f"{epoch:07d}")

            # debug outputs
            if epoch % c_file.epochsRender == 0 and epoch > 0:
                val_data_set, _ = train_config.get_data_set_and_loader('val')
                val_data_set.full_images = True
                img_samples = create_sample_wrapper(val_data_set[0], train_config, True)
                model_idxs = None if train_config.config_file.preTrained else [model_idx]
                render_img(train_config, img_samples, img_name=f"{epoch:07d}", model_idxs=model_idxs)
                val_data_set.full_images = False

            # debug outputs
            if epoch % c_file.epochsValidate == 0 and epoch > 0:
                # release allocated memory! -> don't do that every single epoch, because of performance impact
                x_batch, y_batch, out = None, None, None
                for optimizer in train_config.optimizers:
                    optimizer.zero_grad()

                with torch.cuda.device(train_config.device):
                    torch.cuda.empty_cache()

                val_loss, _ = validate_batch(train_config, epoch, loss, model_idx)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    with open(f"{train_config.logDir}opt.txt", 'w') as f:
                        f.write(f"Optimal validation loss {best_val_loss} at epoch {epoch}")

                    train_config.save_weights(name_suffix="_opt", model_idx=model_idx)

                del val_loss

        train_config.load_specific_weights(c_file.checkPointName, model_idx)
        train_config.epoch0 = epoch_pretrain

    # restore normal sample count
    train_config.train_dataset.num_samples = c_file.samples
    print("pre-training finished")


def train(train_config):
    loss = None

    best_val_loss = sys.float_info.max if train_config.best_valid_loss is None else train_config.best_valid_loss
    c_file = train_config.config_file

    tqdm_range = tqdm(range(train_config.epoch0, train_config.epochs))

    # creating the iterator starts the workers (if any)
    batch_iterator = iter(train_config.train_data_loader)
    num_batches = int(math.ceil(len(train_config.train_dataset) / c_file.batchImages))

    decay_rate = train_config.config_file.lrate_decay
    decay_steps = train_config.config_file.lrate_decay_steps

    pre_train_epochs = 0

    if train_config.config_file.epochsPretrain is not None and len(train_config.config_file.epochsPretrain) != 0:
        pre_train_epochs = max(train_config.config_file.epochsPretrain)

    for epoch in tqdm_range:
        for optim in train_config.optimizers:
            optim.zero_grad()

        # get sample input data
        samples = next(batch_iterator)
        sample_data = create_sample_wrapper(samples, train_config)

        # we create a new iterator once all elements have been exhausted
        # we get a slight performance bump by creating the iterator here and not in the beginning of the next loop
        if epoch % num_batches == 0:
            batch_iterator = iter(train_config.train_data_loader)
        
        losses = []

        # inference
        with torch.cuda.amp.autocast(enabled=c_file.amp):
            outs, inference_dicts = train_config.inference(sample_data, gradient=True)

        # train net
        for out_idx, criterion in enumerate(train_config.losses):
            if criterion is None or train_config.loss_weights[out_idx] == 0 or \
                    train_config.weights_locked(epoch, out_idx):
                continue

            inference_dict = inference_dicts[out_idx]

            y_batch = sample_data.get_train_target(out_idx)
            if y_batch is not None:
                y_batch = y_batch.reshape(y_batch.shape[0] * c_file.samples, -1)

                if len(y_batch.shape) == 3:
                    y_batch = y_batch[:, 0]

            out = outs[out_idx]

            with torch.cuda.amp.autocast(enabled=c_file.amp):
                loss = criterion(out, y_batch, inference_dict=inference_dicts, epoch=epoch) * train_config.loss_weights[out_idx]
            
            losses.append(loss)
            
            train_config.scaler.scale(loss).backward(retain_graph=out_idx < len(outs) - 1)

        for i, optim in enumerate(train_config.optimizers):
            if not train_config.weights_locked(epoch, i):
                train_config.scaler.step(optim)

            # Learning rate decay
            new_lrate = train_config.config_file.lrate * (
                    decay_rate ** ((epoch - pre_train_epochs) / decay_steps))
            for param_group in optim.param_groups:
                param_group['lr'] = new_lrate

        train_config.scaler.update()

        if not c_file.nonVerbose:
            tqdm_range.set_description(f"epoch={epoch:<10} "
                                       f"losses=[{losses[0]:.8f}{', ' if len(losses) > 1 else ''}{'{:.8f}'.format(losses[1]) if len(losses) > 1 else ''}]")

        # debug outputs
        if epoch % c_file.epochsCheckpoint == 0 and epoch > 0:
            train_config.save_weights(name_suffix=f"{epoch:07d}")

        if epoch % c_file.epochsRender == 0 and epoch > 0:
            val_data_set, _ = train_config.get_data_set_and_loader('val')
            val_data_set.full_images = True
            img_samples = create_sample_wrapper(val_data_set[0], train_config, True)
            render_img(train_config, img_samples, img_name=f"{epoch:07d}")
            val_data_set.full_images = False

        rendered_video = False
        if epoch % c_file.epochsVideo == 0 and epoch > 0 and c_file.epochsVideo >= 0:
            render_video(train_config, vid_name=f"{epoch:07d}")
            rendered_video = True

        if epoch % c_file.epochsValidate == 0 and epoch > 0:
            # release allocated memory! -> don't do that every single epoch, because of performance impact
            batch_features, x_batch, y_batch, out = None, None, None, None
            for optimizer in train_config.optimizers:
                optimizer.zero_grad()

            with torch.cuda.device(train_config.device):
                torch.cuda.empty_cache()

            val_loss = None
            img_data = None
            if c_file.adaptiveSamplingThreshold > 0.0 or \
                    epoch > c_file.lossBlendingStart + c_file.lossBlendingDuration or \
                    c_file.lossBlendingStart > train_config.epochs:
                val_loss, img_data = validate_batch(train_config, epoch, loss)

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                with open(f"{train_config.logDir}opt.txt", 'w') as f:
                    f.write(f"Optimal validation loss {best_val_loss} at epoch {epoch}")

                train_config.save_weights(name_suffix="_opt")

                valid_dir = os.path.join(train_config.logDir, "opt", "val")
                os.makedirs(valid_dir, exist_ok=True)

                print("\n")
                psnrs = []
                for i, data in enumerate(img_data):
                    psnrs.append(data["psnr"])
                    print(f"Render all img psnr {i} {psnrs[i]}")

                    for net_index, img in enumerate(data["images"]):
                        save_img(img, train_config.dataset_info, os.path.join(valid_dir, f"_{net_index}_{i}.png"), False)

                print(f"Average PSNR: {np.array(psnrs).mean()}")

                if not rendered_video and c_file.epochsVideo >= 0:
                    render_video(train_config, vid_name="_opt")
                elif rendered_video:
                    # Simply copy over the existing one, because we already rendered a video
                    for net_idx in range(len(train_config.models)):
                        shutil.copy(os.path.join(c_file.logDir, f"{epoch:07d}_{net_idx}.mp4"),
                                    os.path.join(c_file.logDir, f"_opt_{net_idx}.mp4"))

            # release allocated memory
            del val_loss
            del img_data

            with torch.cuda.device(train_config.device):
                torch.cuda.empty_cache()

    del batch_iterator


# just to avoid globals
def main():
    config = Config.init()
    train_config = TrainConfig()
    train_config.initialize(config)

    print(f"Training config: {train_config.logDir.split('/')[-2]} ({config.config})")
    train_config.load_latest_weights()
    pre_train(train_config)
    train(train_config)

    if config.performEvaluation:
        from evaluate import evaluate
        evaluations = ["complexity", "images", "flip", "psnr", "output_images"]

        train_config.load_specific_weights(config.checkPointName)

        evaluate(train_config, None, evaluations)

    # deleting the loaders and sets here takes care of any remaining workers,
    # which otherwise would need to tome out first
    train_config.delete_data_sets_and_laoders()


if __name__ == '__main__':
    main()
