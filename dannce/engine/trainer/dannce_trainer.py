import csv
import os
from datetime import datetime

import imageio
import numpy as np
import torch
from tqdm import tqdm

from dannce.engine.trainer.base_trainer import BaseTrainer
from dannce.engine.trainer.train_utils import (
    LossHelper,
    MetricHelper,
    prepare_batch,
    save_2d_reprojection_visualizations,
)
from dannce.engine.utils.augmentation import construct_augmented_batch
from dannce.engine.utils.image import norm_im


class DANNCETrainer(BaseTrainer):
    """
    Trainer class for DANNCE base networks.
    """

    def __init__(
        self,
        device,
        train_dataloader,
        valid_dataloader,
        lr_scheduler=None,
        visualize_batch=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.loss = LossHelper(self.params, checkpoint_dir=self.checkpoint_dir, logger=self.logger)
        
        # Disable verbose debug log initialization
        self.metrics = MetricHelper(self.params)
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler

        self.visualize_batch = visualize_batch

        self.split = False  # self.params.get("social_joint_training", False)

        # whether each batch only contains transformed versions of one single instance
        self.aug_batch = self.params.get("batch_augmentation", False)
        self.aug_bs = self.params.get("batch_aug_size", None)
        self.per_batch_sample = self.params["batch_size"]

        # set up csv file for tracking training and validation stats
        stats_file = open(
            os.path.join(self.checkpoint_dir, "training.csv"), "w", newline=""
        )
        stats_writer = csv.writer(stats_file)
        self.stats_keys = [*self.loss.names, *self.metrics.names]
        self.train_stats_keys = ["train_" + k for k in self.stats_keys]
        self.valid_stats_keys = ["val_" + k for k in self.stats_keys]
        stats_writer.writerow(["Epoch", *self.train_stats_keys, *self.valid_stats_keys])
        stats_file.close()

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            # open csv
            stats_file = open(
                os.path.join(self.params["dannce_train_dir"], "training.csv"),
                "a",
                newline="",
            )
            stats_writer = csv.writer(stats_file)
            stats = [epoch]
            # train
            train_stats = self._train_epoch(epoch)

            for k in self.stats_keys:
                stats.append(train_stats[k])

            result_msg = f"Epoch[{epoch}/{self.epochs}]\n" + "".join(
                f"train_{k}: {val:.4f}\n" for k, val in train_stats.items()
            )

            # validation
            valid_stats = self._valid_epoch(epoch)

            for k in self.stats_keys:
                stats.append(valid_stats[k])

            result_msg = result_msg + "".join(
                f"val_{k}: {val:.4f}\n" for k, val in valid_stats.items()
            )
            self.logger.info(result_msg)

            # Update learning rate scheduler with validation loss
            if self.lr_scheduler is not None:
                # For ReduceLROnPlateau, use validation loss
                scheduler_type = type(self.lr_scheduler).__name__
                if scheduler_type == "ReduceLROnPlateau":
                    # Use 2D validation loss if available, otherwise use main loss
                    val_loss_2d_key = next((k for k in valid_stats.keys() if k.endswith('_2d')), None)
                    if val_loss_2d_key:
                        self.lr_scheduler.step(valid_stats[val_loss_2d_key])
                        self.logger.info(f"LR Scheduler step with {val_loss_2d_key}: {valid_stats[val_loss_2d_key]:.4f}")
                    else:
                        # Fallback to first available loss
                        val_loss = next(iter(valid_stats.values()))
                        self.lr_scheduler.step(val_loss)
                        self.logger.info(f"LR Scheduler step with validation loss: {val_loss:.4f}")
                else:
                    # For other schedulers (StepLR, etc.), just step
                    self.lr_scheduler.step()
                    
                # Log current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Current learning rate: {current_lr:.2e}")

            # write stats to csv
            stats_writer.writerow(stats)
            stats_file.close()

            # write stats to tensorboard
            for k, v in zip(
                [*self.train_stats_keys, *self.valid_stats_keys], stats[1:]
            ):
                self.writer.add_scalar(k, v, epoch)

            # save checkpoints after each save period or at the end of training
            self._save_checkpoint(epoch)

    def _forward(self, epoch, batch, train=True):
        volumes, grid_centers, keypoints_3d_gt, aux, keypoints_2d_gt, batch_debug_info, sample_ids = prepare_batch(batch, self.device)

        if self.visualize_batch:
            self.visualize(epoch, volumes)
            return

        if train and self.aug_batch:
            volumes, grid_centers, aux = construct_augmented_batch(
                volumes.permute(0, 2, 3, 4, 1),
                grid_centers,
                aux=aux if aux is None else aux.permute(0, 2, 3, 4, 1),
                copies_per_sample=self.aug_bs // self.per_batch_sample,
            )
            volumes = volumes.permute(0, 4, 1, 2, 3)
            aux = aux if aux is None else aux.permute(0, 4, 1, 2, 3)
            keypoints_3d_gt = keypoints_3d_gt.repeat(self.aug_bs, 1, 1)
            if keypoints_2d_gt is not None:
                keypoints_2d_gt = keypoints_2d_gt.repeat(self.aug_bs, 1, 1)

        keypoints_3d_pred, heatmaps, _ = self.model(volumes, grid_centers)

        keypoints_3d_gt, keypoints_3d_pred, heatmaps = self._split_data(
            keypoints_3d_gt, keypoints_3d_pred, heatmaps
        )

        return (
            keypoints_3d_gt,
            keypoints_3d_pred,
            heatmaps,
            grid_centers,
            aux,
            keypoints_2d_gt,
            self.train_dataloader.dataset.cameras,
            sample_ids,
        )

    def _train_epoch(self, epoch):
        self.model.train()
        
        # Set epoch for debug logging
        self.loss.set_epoch(epoch)

        # Anomaly detection disabled - inplace operations have been eliminated
        # with torch.autograd.set_detect_anomaly(True):
        epoch_loss_dict, epoch_metric_dict = {}, {}
        pbar = tqdm(self.train_dataloader)
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            (
                keypoints_3d_gt,
                keypoints_3d_pred,
                heatmaps,
                grid_centers,
                aux,
                keypoints_2d_gt,
                cameras,
                sample_ids,
            ) = self._forward(epoch, batch)

            total_loss, loss_dict = self.loss.compute_loss(
                keypoints_3d_gt,
                keypoints_3d_pred,
                heatmaps,
                grid_centers,
                aux,
                keypoints_2d_gt=keypoints_2d_gt,
                cameras=cameras,
                sample_ids=sample_ids,
            )

            # Optional concise grad print - disable by default to reduce noise
            # Build concise oneline status with 2D loss if present
            keys_sorted = sorted(loss_dict.keys())
            two_d_keys = [k for k in keys_sorted if k.endswith('_2d')]
            three_d_keys = [k for k in keys_sorted if not k.endswith('_2d')]
            parts = []
            # Prefer 2D loss visibility
            for k in two_d_keys:
                parts.append(f"{k}:{loss_dict[k]:.4f}")
            # Include a small subset of other losses
            for k in three_d_keys[:2]:
                parts.append(f"{k}:{loss_dict[k]:.4f}")
            result = f"Epoch[{epoch}/{self.epochs}] " + " ".join(parts)
            pbar.set_description(result)
            # Remove duplicate print - progress bar already shows this info

            # Optional: report grad norm every batch (compact)
            # Note: keypoints_3d_pred is not a leaf tensor and doesn't have .grad
            # Gradient access removed to prevent autograd warnings and errors

            total_loss.backward()
            
            # Apply gradient clipping if specified
            clip_threshold = self.params.get('gradient_clip_norm')
            if clip_threshold is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    clip_threshold
                )
                # Log if gradient was clipped (only for high values to avoid spam)
                if grad_norm > clip_threshold * 2:
                    print(f"   ⚡ Gradient norm clipped: {grad_norm:.2f} -> {clip_threshold}", flush=True)
            
            self.optimizer.step()

            epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

            # Save 2D reprojection visualizations for first 3 batches per epoch
            try:
                train_on_2d = self.params.get("train_on_2d", False)
                
                if epoch == self.start_epoch and batch_idx < 3 and train_on_2d:
                    with torch.no_grad():
                        vols_for_vis = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else None
                        
                        # Get sample_id directly from the dataset partition
                        sample_id = self.train_dataloader.dataset.train_sample_ids[batch_idx]

                        save_2d_reprojection_visualizations(
                            epoch=epoch,
                            batch_idx=batch_idx,
                            volumes=vols_for_vis,
                            kpts_pred=keypoints_3d_pred,
                            keypoints_2d_gt=keypoints_2d_gt,
                            cameras=cameras,
                            params=self.params,
                            checkpoint_dir=self.checkpoint_dir,
                            dataset=self.train_dataloader.dataset,
                            batch=batch,  # Pass original batch for accessing images
                            sample_id=sample_id, # Pass single sample ID
                        )
                else:
                    if not train_on_2d:
                        pass
                    elif epoch != self.start_epoch:
                        pass
                    elif batch_idx >= 3:
                        pass
            except Exception as e:
                print(f"❌ VIS ERROR: {e}")
                import traceback
                traceback.print_exc()

            if len(self.metrics.names) != 0:
                metric_dict = self.metrics.evaluate(
                    keypoints_3d_pred.detach().cpu().numpy(),
                    keypoints_3d_gt.clone().cpu().numpy(),
                )
                epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)

        # Note: LR scheduler step is called after validation in train() method

        epoch_loss_dict, epoch_metric_dict = (
            self._average(epoch_loss_dict),
            self._average(epoch_metric_dict),
        )
        return {**epoch_loss_dict, **epoch_metric_dict}

    def _valid_epoch(self, epoch):
        self.model.eval()
        
        # Set epoch for debug logging
        self.loss.set_epoch(epoch)

        epoch_loss_dict = {}
        epoch_metric_dict = {}

        pbar = tqdm(self.valid_dataloader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                (
                    keypoints_3d_gt,
                    keypoints_3d_pred,
                    heatmaps,
                    grid_centers,
                    aux,
                    keypoints_2d_gt,
                    cameras,
                    sample_ids,
                ) = self._forward(epoch, batch, False)

                _, loss_dict = self.loss.compute_loss(
                    keypoints_3d_gt,
                    keypoints_3d_pred,
                    heatmaps,
                    grid_centers,
                    aux,
                    keypoints_2d_gt=keypoints_2d_gt,
                    cameras=cameras,
                    sample_ids=sample_ids,
                )
                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                # Optionally visualize first 3 validation batches as well
                try:
                    if epoch == self.start_epoch and batch_idx < 3 and self.params.get("train_on_2d", False):
                        vols_for_vis = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else None
                        
                        # Get sample_id directly from the dataset partition
                        sample_id = self.valid_dataloader.dataset.valid_sample_ids[batch_idx]
                        
                        save_2d_reprojection_visualizations(
                            epoch=epoch,
                            batch_idx=batch_idx,
                            volumes=vols_for_vis,
                            kpts_pred=keypoints_3d_pred,
                            keypoints_2d_gt=keypoints_2d_gt,
                            cameras=cameras,
                            params=self.params,
                            checkpoint_dir=self.checkpoint_dir,
                            dataset=self.valid_dataloader.dataset,
                            batch=batch,
                            sample_id=sample_id, # Pass single sample ID
                        )
                except Exception:
                    pass

                if len(self.metrics.names) != 0:
                    metric_dict = self.metrics.evaluate(
                        keypoints_3d_pred.detach().cpu().numpy(),
                        keypoints_3d_gt.clone().cpu().numpy(),
                    )
                    epoch_metric_dict = self._update_step(
                        epoch_metric_dict, metric_dict
                    )

        epoch_loss_dict, epoch_metric_dict = (
            self._average(epoch_loss_dict),
            self._average(epoch_metric_dict),
        )
        return {**epoch_loss_dict, **epoch_metric_dict}

    def _split_data(self, keypoints_3d_gt, keypoints_3d_pred, heatmaps):
        if not self.split:
            return keypoints_3d_gt, keypoints_3d_pred, heatmaps

        keypoints_3d_gt = keypoints_3d_gt.reshape(
            *keypoints_3d_gt.shape[:2], 2, -1
        ).permute(0, 2, 1, 3)
        keypoints_3d_gt = keypoints_3d_gt.reshape(-1, *keypoints_3d_gt.shape[2:])
        keypoints_3d_pred = keypoints_3d_pred.reshape(
            *keypoints_3d_pred.shape[:2], 2, -1
        ).permute(0, 2, 1, 3)
        keypoints_3d_pred = keypoints_3d_pred.reshape(-1, *keypoints_3d_pred.shape[2:])
        heatmaps = heatmaps.reshape(heatmaps.shape[0], 2, -1, *heatmaps.shape[2:])
        heatmaps = heatmaps.reshape(-1, *heatmaps.shape[2:])

        return keypoints_3d_gt, keypoints_3d_pred, heatmaps

    def _update_step(self, epoch_dict, step_dict):
        if len(epoch_dict) == 0:
            for k, v in step_dict.items():
                epoch_dict[k] = [v]
        else:
            for k, v in step_dict.items():
                epoch_dict[k].append(v)
        return epoch_dict

    def _average(self, epoch_dict):
        for k, v in epoch_dict.items():
            valid_num = sum([item > 0 for item in v])
            epoch_dict[k] = sum(v) / valid_num if valid_num > 0 else 0.0
        return epoch_dict

    def _rewrite_csv(self):
        stats_file = open(
            os.path.join(self.params["dannce_train_dir"], "training.csv"),
            "w",
            newline="",
        )
        stats_writer = csv.writer(stats_file)
        stats_writer.writerow(["Epoch", *self.train_stats_keys, *self.valid_stats_keys])
        stats_file.close()

    def _add_loss_attr(self, names):
        self.stats_keys = names + self.stats_keys
        self.train_stats_keys = [f"train_{k}" for k in names] + self.train_stats_keys
        self.valid_stats_keys = [f"val_{k}" for k in names] + self.valid_stats_keys

        self._rewrite_csv()

    def _del_loss_attr(self, names):
        for name in names:
            self.stats_keys.remove(name)
            self.train_stats_keys.remove(f"train_{name}")
            self.valid_stats_keys.remove(f"val_{name}")

        self._rewrite_csv()

    def visualize(self, epoch, volumes):
        tifdir = os.path.join(
            self.params["dannce_train_dir"], "debug_volumes", f"epoch{epoch}"
        )
        if not os.path.exists(tifdir):
            os.makedirs(tifdir)
        print("Dump training volumes to {}".format(tifdir))
        volumes = volumes.clone().detach().cpu().permute(0, 2, 3, 4, 1).numpy()
        for i in range(volumes.shape[0]):
            for j in range(volumes.shape[-1] // self.params["chan_num"]):
                im = volumes[
                    i,
                    :,
                    :,
                    :,
                    j * self.params["chan_num"] : (j + 1) * self.params["chan_num"],
                ]
                im = norm_im(im) * 255
                im = im.astype("uint8")
                of = os.path.join(tifdir, f"sample{i}" + "_cam" + str(j) + ".tif",)
                imageio.mimwrite(of, np.transpose(im, [2, 0, 1, 3]))
