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
    build_visibility_camera_features,
    get_visibility_camera_order,
    prepare_batch,
    save_2d_reprojection_visualizations,
    visibility_dict_to_tensor,
    visibility_tensor_to_dict,
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
        self.learned_visibility_enabled = bool(
            self.params.get("learned_visibility_enabled", False)
        )
        self.learned_visibility_bce_enabled = self.learned_visibility_enabled
        self.learned_visibility_bce_stop_below = self.params.get(
            "learned_visibility_stop_bce_below", None
        )
        self.learned_visibility_bce_stop_metric = str(
            self.params.get("learned_visibility_stop_bce_metric", "val")
        ).lower()
        self.learned_visibility_bce_stop_after_epoch = int(
            self.params.get("learned_visibility_stop_bce_after_epoch", 1)
        )
        self.learned_visibility_bce_stopped_epoch = None

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

    def _build_visibility_camera_features(
        self,
        sample_ids,
        visibility_2d_gt,
        cameras,
        dtype,
    ):
        if not self.learned_visibility_enabled or visibility_2d_gt is None:
            return None

        camera_order = get_visibility_camera_order(visibility_2d_gt)
        if len(camera_order) == 0:
            return None

        return build_visibility_camera_features(
            sample_ids,
            cameras,
            camera_order,
            device=self.device,
            dtype=dtype,
        )

    def _predict_with_optional_visibility(
        self,
        volumes,
        grid_centers,
        visibility_camera_features=None,
    ):
        if self.learned_visibility_enabled:
            keypoints_3d_pred, heatmaps, aux_outputs = self.model.predict_with_aux(
                volumes,
                grid_centers,
                visibility_camera_features=visibility_camera_features,
            )
            return keypoints_3d_pred, heatmaps, aux_outputs.get("visibility_logits")

        keypoints_3d_pred, heatmaps, _ = self.model(volumes, grid_centers)
        return keypoints_3d_pred, heatmaps, None

    def _compute_visibility_terms(self, epoch, visibility_2d_gt, visibility_logits):
        if not self.learned_visibility_enabled:
            return visibility_2d_gt, None, {}

        zero_loss = torch.zeros((), device=self.device)
        metric = {"VisibilityBCE": 0.0}

        if visibility_2d_gt is None:
            return visibility_2d_gt, zero_loss, metric

        camera_order = get_visibility_camera_order(visibility_2d_gt)
        if len(camera_order) == 0:
            return visibility_2d_gt, zero_loss, metric

        if visibility_logits is None:
            raise RuntimeError(
                "learned_visibility_enabled=True but the model did not return visibility logits."
            )

        effective_visibility = visibility_2d_gt
        warmup_epochs = int(self.params.get("learned_visibility_warmup_epochs", 1))
        mode = self.params.get("exclude_occluded_2d_mode", "capsule_raycast")
        if mode == "learned" and epoch > warmup_epochs:
            threshold = float(self.params.get("learned_visibility_threshold", 0.5))
            predicted_visibility = torch.sigmoid(visibility_logits) >= threshold
            effective_visibility = visibility_tensor_to_dict(
                predicted_visibility, camera_order
            )

        if not self.learned_visibility_bce_enabled:
            return effective_visibility, zero_loss, metric

        target_tensor = visibility_dict_to_tensor(
            visibility_2d_gt,
            camera_order,
            device=visibility_logits.device,
            dtype=visibility_logits.dtype,
        )
        if target_tensor is None:
            return effective_visibility, zero_loss, metric

        if visibility_logits.shape != target_tensor.shape:
            raise ValueError(
                "Visibility logits shape mismatch: "
                f"got {tuple(visibility_logits.shape)}, expected {tuple(target_tensor.shape)}."
            )

        visibility_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            visibility_logits, target_tensor
        )
        visibility_loss = visibility_loss * float(
            self.params.get("learned_visibility_loss_weight", 1.0)
        )
        metric["VisibilityBCE"] = visibility_loss.detach().cpu().item()

        return effective_visibility, visibility_loss, metric

    def _maybe_disable_visibility_bce(self, epoch, train_stats, valid_stats):
        if not self.learned_visibility_enabled or not self.learned_visibility_bce_enabled:
            return

        threshold = self.learned_visibility_bce_stop_below
        if threshold is None or epoch < self.learned_visibility_bce_stop_after_epoch:
            return

        metric_source = (
            valid_stats if self.learned_visibility_bce_stop_metric == "val" else train_stats
        )
        metric_prefix = (
            "val" if self.learned_visibility_bce_stop_metric == "val" else "train"
        )
        current_bce = metric_source.get("VisibilityBCE")
        if current_bce is None:
            return

        if float(current_bce) <= float(threshold):
            self.learned_visibility_bce_enabled = False
            self.learned_visibility_bce_stopped_epoch = epoch
            self.logger.info(
                "Disabling visibility BCE supervision after epoch "
                f"{epoch}: {metric_prefix}_VisibilityBCE={float(current_bce):.4f} "
                f"<= {float(threshold):.4f}."
            )

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

            # Optional early-stopping condition on validation 3D MPJPE.
            # This is useful for long runs that should stop once a target accuracy is reached.
            stop_thr = self.params.get("stop_when_val_euclidean_distance_3D_below", None)
            should_stop = False
            if stop_thr is not None:
                try:
                    stop_thr = float(stop_thr)
                    val_mpjpe = valid_stats.get("euclidean_distance_3D", None)
                    if val_mpjpe is not None and float(val_mpjpe) <= stop_thr:
                        should_stop = True
                except Exception:
                    # If parsing fails, ignore the stop request rather than crashing training.
                    should_stop = False

            result_msg = result_msg + "".join(
                f"val_{k}: {val:.4f}\n" for k, val in valid_stats.items()
            )
            self._maybe_disable_visibility_bce(epoch, train_stats, valid_stats)
            self.logger.info(result_msg)

            # Update learning rate scheduler with validation loss
            if self.lr_scheduler is not None:
                # For ReduceLROnPlateau, use validation loss
                scheduler_type = type(self.lr_scheduler).__name__
                if scheduler_type == "ReduceLROnPlateau":
                    # Allow explicit monitor override, otherwise prefer 3D metric for training decisions.
                    monitor_key = self.params.get("lr_scheduler_monitor")
                    if monitor_key in valid_stats:
                        chosen_key = monitor_key
                    elif "euclidean_distance_3D" in valid_stats:
                        chosen_key = "euclidean_distance_3D"
                    else:
                        # Backward-compatible fallback for runs without 3D metric logging.
                        val_loss_2d_key = next(
                            (k for k in valid_stats.keys() if k.endswith("_2d")),
                            None,
                        )
                        chosen_key = val_loss_2d_key or next(iter(valid_stats.keys()))

                    monitor_val = valid_stats[chosen_key]
                    self.lr_scheduler.step(monitor_val)
                    self.logger.info(
                        f"LR Scheduler step with {chosen_key}: {monitor_val:.4f}"
                    )
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

            if should_stop:
                self.logger.info(
                    f"Early stopping at epoch {epoch}: val_euclidean_distance_3D <= {stop_thr}"
                )
                break

    def _forward(self, epoch, batch, train=True):
        (
            volumes,
            grid_centers,
            keypoints_3d_gt,
            aux,
            keypoints_2d_gt,
            visibility_2d_gt,
            batch_debug_info,
            sample_ids,
        ) = prepare_batch(batch, self.device)

        if self.visualize_batch:
            self.visualize(epoch, volumes)
            return

        cameras = (
            self.train_dataloader.dataset.cameras
            if train
            else self.valid_dataloader.dataset.cameras
        )

        if train and self.aug_batch:
            copies_per_sample = self.aug_bs // self.per_batch_sample
            volumes, grid_centers, aux = construct_augmented_batch(
                volumes.permute(0, 2, 3, 4, 1),
                grid_centers,
                aux=aux if aux is None else aux.permute(0, 2, 3, 4, 1),
                copies_per_sample=copies_per_sample,
            )
            volumes = volumes.permute(0, 4, 1, 2, 3)
            aux = aux if aux is None else aux.permute(0, 4, 1, 2, 3)
            keypoints_3d_gt = keypoints_3d_gt.repeat_interleave(
                copies_per_sample, dim=0
            )
            if sample_ids is not None:
                sample_ids = [
                    sample_id
                    for sample_id in sample_ids
                    for _ in range(copies_per_sample)
                ]
            if keypoints_2d_gt is not None:
                if isinstance(keypoints_2d_gt, dict):
                    keypoints_2d_gt = {
                        cam_name: cam_data.repeat(self.aug_bs, 1, 1)
                        for cam_name, cam_data in keypoints_2d_gt.items()
                    }
                else:
                    keypoints_2d_gt = keypoints_2d_gt.repeat(self.aug_bs, 1, 1)
            if visibility_2d_gt is not None:
                if isinstance(visibility_2d_gt, dict):
                    visibility_2d_gt = {
                        cam_name: cam_data.repeat(self.aug_bs, 1)
                        for cam_name, cam_data in visibility_2d_gt.items()
                    }
                else:
                    visibility_2d_gt = visibility_2d_gt.repeat(self.aug_bs, 1)

        visibility_camera_features = self._build_visibility_camera_features(
            sample_ids,
            visibility_2d_gt,
            cameras,
            volumes.dtype,
        )
        keypoints_3d_pred, heatmaps, visibility_logits = (
            self._predict_with_optional_visibility(
                volumes,
                grid_centers,
                visibility_camera_features=visibility_camera_features,
            )
        )

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
            visibility_2d_gt,
            visibility_logits,
            cameras,
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
                visibility_2d_gt,
                visibility_logits,
                cameras,
                sample_ids,
            ) = self._forward(epoch, batch)

            visibility_mask, visibility_loss, visibility_metrics = (
                self._compute_visibility_terms(
                    epoch, visibility_2d_gt, visibility_logits
                )
            )

            total_loss, loss_dict = self.loss.compute_loss(
                keypoints_3d_gt,
                keypoints_3d_pred,
                heatmaps,
                grid_centers,
                aux,
                keypoints_2d_gt=keypoints_2d_gt,
                visibility_2d_gt=visibility_mask,
                cameras=cameras,
                sample_ids=sample_ids,
            )
            if visibility_loss is not None:
                total_loss = total_loss + visibility_loss
                loss_dict.update(visibility_metrics)

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
                # Intentionally avoid per-batch logging here; frequent stdout writes
                # slow long runs with large numbers of training steps.
            
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
                    visibility_2d_gt,
                    visibility_logits,
                    cameras,
                    sample_ids,
                ) = self._forward(epoch, batch, False)

                visibility_mask, _, visibility_metrics = self._compute_visibility_terms(
                    epoch, visibility_2d_gt, visibility_logits
                )

                _, loss_dict = self.loss.compute_loss(
                    keypoints_3d_gt,
                    keypoints_3d_pred,
                    heatmaps,
                    grid_centers,
                    aux,
                    keypoints_2d_gt=keypoints_2d_gt,
                    visibility_2d_gt=visibility_mask,
                    cameras=cameras,
                    sample_ids=sample_ids,
                )
                loss_dict.update(visibility_metrics)
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
            n_prev = len(next(iter(epoch_dict.values()))) if len(epoch_dict) > 0 else 0
            for k in list(epoch_dict.keys()):
                if k not in step_dict:
                    epoch_dict[k].append(0.0)
            for k, v in step_dict.items():
                if k not in epoch_dict:
                    epoch_dict[k] = [0.0] * n_prev
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
