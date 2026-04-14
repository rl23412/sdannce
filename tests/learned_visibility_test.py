import shutil
import unittest
from pathlib import Path

import torch
import yaml

import dannce.config as config
from dannce import _param_defaults_dannce
from dannce.engine.models.nets import DANNCE
from dannce.engine.models.posegcn.nets import PoseGCN
from dannce.engine.trainer.dannce_trainer import DANNCETrainer
from dannce.engine.trainer.train_utils import (
    build_visibility_camera_features,
)
from dannce.engine.utils.vis import save_visibility_overlay_previews
from tools.preview_real_learned_visibility import build_label_subset


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DANNCE_CONFIG = REPO_ROOT / "configs" / "dannce_rat_config.yaml"
LABEL3D_FILE = REPO_ROOT / "vid1" / "vid1_Label3D_dannce.mat"
VIDEO_DIR = REPO_ROOT / "vid1" / "videos"
SMOKE_ROOT = REPO_ROOT / "tests" / "_learned_visibility_smoke"


def _make_grid_centers(batch_size=2, nvox=8):
    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(-1.0, 1.0, nvox),
            torch.linspace(-1.0, 1.0, nvox),
            torch.linspace(-1.0, 1.0, nvox),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    return coords.unsqueeze(0).repeat(batch_size, 1, 1)


def _make_visibility_camera_features(batch_size=2, n_views=2):
    camera_features = torch.randn(batch_size, n_views, 6)
    camera_features[..., 3:6] = torch.nn.functional.normalize(
        camera_features[..., 3:6],
        dim=-1,
    )
    return camera_features


class LearnedVisibilityUnitTest(unittest.TestCase):
    def test_dannce_visibility_head_shape(self):
        model = DANNCE(
            input_channels=6,
            output_channels=23,
            input_shape=8,
            compressed=True,
            visibility_num_views=2,
            visibility_hidden_dim=32,
        )
        volumes = torch.randn(2, 6, 8, 8, 8)
        grid_centers = _make_grid_centers(batch_size=2, nvox=8)
        camera_features = _make_visibility_camera_features(batch_size=2, n_views=2)

        coords, heatmaps, aux_outputs = model.predict_with_aux(
            volumes,
            grid_centers,
            visibility_camera_features=camera_features,
        )

        self.assertEqual(coords.shape, (2, 3, 23))
        self.assertEqual(heatmaps.shape[:2], (2, 23))
        self.assertEqual(aux_outputs["visibility_logits"].shape, (2, 2, 23))

    def test_sdannce_visibility_head_shape(self):
        pose_generator = DANNCE(
            input_channels=6,
            output_channels=23,
            input_shape=8,
            compressed=True,
            visibility_num_views=2,
            visibility_hidden_dim=32,
        )
        params = {"n_channels_out": 23, "skeleton": "rat23", "temporal_chunk_size": 1}
        graph_cfg = {
            "n_instances": 1,
            "use_features": False,
            "use_residual": False,
            "hidden_dim": 64,
            "n_layers": 1,
        }
        model = PoseGCN(params, graph_cfg, pose_generator)
        volumes = torch.randn(2, 6, 8, 8, 8)
        grid_centers = _make_grid_centers(batch_size=2, nvox=8)
        camera_features = _make_visibility_camera_features(batch_size=2, n_views=2)

        init_poses, final_poses, heatmaps, aux_outputs = model.predict_with_aux(
            volumes,
            grid_centers,
            visibility_camera_features=camera_features,
        )

        self.assertEqual(init_poses.shape, (2, 3, 23))
        self.assertEqual(final_poses.shape, (2, 3, 23))
        self.assertEqual(heatmaps.shape[:2], (2, 23))
        self.assertEqual(aux_outputs["visibility_logits"].shape, (2, 2, 23))

    def test_setup_train_rejects_learned_visibility_without_2d_training(self):
        params = {
            "camnames": ["Camera1", "Camera2"],
            "viddir": str(VIDEO_DIR),
            "exp": [{"label3d_file": str(LABEL3D_FILE), "viddir": str(VIDEO_DIR)}],
            "vmin": -120,
            "vmax": 120,
            "nvox": 16,
            "n_views": 2,
            "dataset": "rat7m",
            "n_channels_out": 20,
            "train_mode": "new",
            "loss": {"L1Loss": {"loss_weight": 1.0}},
            "exclude_occluded_2d": True,
            "exclude_occluded_2d_mode": "learned",
            "learned_visibility_enabled": True,
            "train_on_2d": False,
            "n_instances": 1,
            "multi_gpu_train": False,
            "graph_cfg": None,
            "use_silhouette_in_volume": False,
            "use_silhouette": False,
            "batch_size": 1,
            "chan_num": 3,
            "cam3_train": False,
            "channel_combo": None,
            "expval": True,
            "mono": False,
            "unlabeled_fraction": None,
            "dataset_args": None,
            "random_seed": 1,
            "crop_height": [0, 1200],
            "crop_width": [0, 1920],
        }

        with self.assertRaises(ValueError):
            config.setup_train(params)


class _DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


class LearnedVisibilityScheduleTest(unittest.TestCase):
    def test_train_on_2d_defaults_to_geometry_only_learned_visibility(self):
        params = {
            "camnames": ["Camera1", "Camera2"],
            "exp": [{"camnames": ["Camera1", "Camera2"]}],
            "vmin": -120,
            "vmax": 120,
            "nvox": 16,
            "train_on_2d": True,
            "train_on_2d_use_default_learned_visibility": True,
            "exclude_occluded_2d": False,
            "exclude_occluded_2d_mode": "capsule_raycast",
            "learned_visibility_enabled": False,
            "learned_visibility_warmup_epochs": 1,
            "exclude_occluded_2d_reproj_only_from_stored3d": False,
        }

        config.check_config(params, dannce_net=True, prediction=False)

        self.assertTrue(params["exclude_occluded_2d"])
        self.assertEqual(params["exclude_occluded_2d_mode"], "learned")
        self.assertTrue(params["learned_visibility_enabled"])
        self.assertEqual(params["learned_visibility_warmup_epochs"], 5)
        self.assertTrue(params["exclude_occluded_2d_reproj_only_from_stored3d"])

    def test_train_on_2d_can_opt_out_of_default_learned_visibility(self):
        params = {
            "camnames": ["Camera1", "Camera2"],
            "exp": [{"camnames": ["Camera1", "Camera2"]}],
            "vmin": -120,
            "vmax": 120,
            "nvox": 16,
            "train_on_2d": True,
            "train_on_2d_use_default_learned_visibility": False,
            "exclude_occluded_2d": False,
            "exclude_occluded_2d_mode": "capsule_raycast",
            "learned_visibility_enabled": False,
            "learned_visibility_warmup_epochs": 1,
            "exclude_occluded_2d_reproj_only_from_stored3d": False,
        }

        config.check_config(params, dannce_net=True, prediction=False)

        self.assertFalse(params["exclude_occluded_2d"])
        self.assertEqual(params["exclude_occluded_2d_mode"], "capsule_raycast")
        self.assertFalse(params["learned_visibility_enabled"])
        self.assertEqual(params["learned_visibility_warmup_epochs"], 1)
        self.assertFalse(params["exclude_occluded_2d_reproj_only_from_stored3d"])

    def test_disable_visibility_bce_after_threshold(self):
        trainer = object.__new__(DANNCETrainer)
        trainer.learned_visibility_enabled = True
        trainer.learned_visibility_bce_enabled = True
        trainer.learned_visibility_bce_stop_below = 0.2
        trainer.learned_visibility_bce_stop_metric = "val"
        trainer.learned_visibility_bce_stop_after_epoch = 2
        trainer.learned_visibility_bce_stopped_epoch = None
        trainer.logger = _DummyLogger()

        trainer._maybe_disable_visibility_bce(
            epoch=1,
            train_stats={"VisibilityBCE": 0.18},
            valid_stats={"VisibilityBCE": 0.18},
        )
        self.assertTrue(trainer.learned_visibility_bce_enabled)
        self.assertIsNone(trainer.learned_visibility_bce_stopped_epoch)

        trainer._maybe_disable_visibility_bce(
            epoch=2,
            train_stats={"VisibilityBCE": 0.25},
            valid_stats={"VisibilityBCE": 0.19},
        )
        self.assertFalse(trainer.learned_visibility_bce_enabled)
        self.assertEqual(trainer.learned_visibility_bce_stopped_epoch, 2)
        self.assertTrue(any("Disabling visibility BCE supervision" in m for m in trainer.logger.messages))

    def test_update_step_handles_missing_loss_keys(self):
        trainer = object.__new__(DANNCETrainer)
        epoch_dict = trainer._update_step({}, {"VisibilityBCE": 0.3})
        epoch_dict = trainer._update_step(
            epoch_dict,
            {"CharbonnierLoss_2d": 4.2, "VisibilityBCE": 0.2},
        )
        epoch_dict = trainer._update_step(epoch_dict, {"VisibilityBCE": 0.1})

        self.assertEqual(epoch_dict["VisibilityBCE"], [0.3, 0.2, 0.1])
        self.assertEqual(epoch_dict["CharbonnierLoss_2d"], [0.0, 4.2, 0.0])


class LearnedVisibilitySmokeTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = SMOKE_ROOT
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_io_override(self):
        io_params = {
            "viddir": "videos",
            "exp": [
                {
                    "label3d_file": str(LABEL3D_FILE),
                    "viddir": str(VIDEO_DIR),
                }
            ]
        }
        io_path = self.output_dir / "io_smoke.yaml"
        io_path.write_text(yaml.safe_dump(io_params), encoding="utf-8")
        return io_path

    def _build_train_params(self):
        io_override = self._build_io_override()
        params = {
            **_param_defaults_dannce,
            **config.build_params(
            str(BASE_DANNCE_CONFIG),
            dannce_net=True,
            io_config_override=str(io_override),
            ),
        }
        params["viddir"] = "videos"
        params["camnames"] = ["Camera1", "Camera2"]
        params["n_channels_out"] = 14
        params["new_n_channels_out"] = 14
        params["n_views"] = 2
        params["nvox"] = 16
        params["sigma"] = 4
        params["vmin"] = -80
        params["vmax"] = 80
        params["gpu_id"] = "cpu"
        params["batch_size"] = 1
        params["epochs"] = 1
        params["save_period"] = 1
        params["num_train_per_exp"] = 2
        params["num_validation_per_exp"] = 1
        params["COM_augmentation"] = False
        params["rand_view_replace"] = False
        params["n_rand_views"] = 0
        params["com_fromlabels"] = True
        params["train_on_2d"] = True
        params["exclude_occluded_2d"] = True
        params["exclude_occluded_2d_mode"] = "learned"
        params["learned_visibility_enabled"] = True
        params["learned_visibility_warmup_epochs"] = 0
        params["learned_visibility_loss_weight"] = 0.2
        params["learned_visibility_hidden_dim"] = 64
        params["learned_visibility_threshold"] = 0.5
        params["dannce_train_dir"] = str(self.output_dir / "DANNCE" / "train_learned_visibility")
        params["use_npy"] = False
        params = config.infer_params(params, dannce_net=True, prediction=False)
        config.check_config(params, dannce_net=True, prediction=False)
        return params

    def test_dannce_smoke_generates_visibility_previews(self):
        params = self._build_train_params()
        sample_ids = ["0_41498"]
        labels_2d, _labels_3d, cameras, camera_order = build_label_subset(
            LABEL3D_FILE,
            sample_ids,
            params,
            str(VIDEO_DIR),
        )
        model = DANNCE(
            input_channels=6,
            output_channels=params["n_channels_out"],
            input_shape=params["nvox"],
            compressed=True,
            visibility_num_views=len(camera_order),
            visibility_hidden_dim=64,
        )
        model.eval()
        volumes = torch.randn(1, 6, params["nvox"], params["nvox"], params["nvox"])
        grids = _make_grid_centers(batch_size=1, nvox=params["nvox"])

        with torch.no_grad():
            camera_features = build_visibility_camera_features(
                sample_ids,
                cameras,
                camera_order,
                device=volumes.device,
                dtype=volumes.dtype,
            )
            _coords, _heatmaps, aux_outputs = model.predict_with_aux(
                volumes,
                grids,
                visibility_camera_features=camera_features,
            )

        preview_dir = Path(params["dannce_train_dir"]) / "visibility_preview"
        preview_paths = save_visibility_overlay_previews(
            params=params,
            labels_2d=labels_2d,
            sample_ids=sample_ids,
            visibility_logits=aux_outputs["visibility_logits"],
            visibility_targets=labels_2d[sample_ids[0]]["visibility"],
            output_dir=str(preview_dir),
            camera_order=camera_order,
            threshold=params["learned_visibility_threshold"],
            max_samples=1,
        )

        self.assertGreater(len(preview_paths), 0)
        for preview_path in preview_paths:
            self.assertTrue(Path(preview_path).exists())


if __name__ == "__main__":
    unittest.main()
