import unittest

import numpy as np

from dannce.engine.utils.visibility import (
    _project_point_to_camera,
    populate_occlusion_visibility,
)


def _make_camera(camera_center):
    camera_center = np.asarray(camera_center, dtype=np.float64).reshape(1, 3)
    return {
        "K": np.array(
            [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [640.0, 512.0, 1.0]],
            dtype=np.float64,
        ),
        "R": np.eye(3, dtype=np.float64),
        "t": -camera_center,
        "RDistort": np.zeros(3, dtype=np.float64),
        "TDistort": np.zeros(2, dtype=np.float64),
    }


class VisibilityOcclusionTest(unittest.TestCase):
    def setUp(self):
        self.params = {
            "exclude_occluded_2d": True,
            "exclude_occluded_2d_mode": "capsule_raycast",
            "exclude_occluded_2d_reproj_guard_px": 5.0,
            "exclude_occluded_2d_min_support_views": 2,
            "skeleton": "rat23",
        }

    def test_raycast_masks_only_blocked_camera(self):
        sample_id = "0_sample"
        cameras = {
            "0_cam0": _make_camera([0.0, 0.0, 0.0]),
            "0_cam1": _make_camera([300.0, 0.0, 0.0]),
        }

        pose3d = np.full((3, 23), np.nan, dtype=np.float64)
        pose3d[:, 0] = np.array([0.0, 0.0, 1000.0])
        pose3d[:, 1] = np.array([0.0, 0.0, 500.0])

        cam_data = {}
        for cam_name, cam_params in cameras.items():
            cam_points = np.full((2, 23), np.nan, dtype=np.float64)
            cam_points[:, 0] = _project_point_to_camera(pose3d[:, 0], cam_params)
            cam_data[cam_name] = cam_points

        datadict = {sample_id: {"data": cam_data, "frames": {}}}
        datadict_3d = {sample_id: pose3d}

        populate_occlusion_visibility(
            datadict=datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=[sample_id],
            params=self.params,
        )

        visibility = datadict[sample_id]["visibility"]
        self.assertFalse(visibility["0_cam0"][0])
        self.assertTrue(visibility["0_cam1"][0])

    def test_insufficient_support_keeps_finite_2d_point(self):
        sample_id = "0_sample"
        cameras = {"0_cam0": _make_camera([0.0, 0.0, 0.0])}

        pose3d = np.full((3, 23), np.nan, dtype=np.float64)
        cam_points = np.full((2, 23), np.nan, dtype=np.float64)
        cam_points[:, 0] = np.array([640.0, 512.0], dtype=np.float64)

        datadict = {sample_id: {"data": {"0_cam0": cam_points}, "frames": {}}}
        datadict_3d = {sample_id: pose3d}

        populate_occlusion_visibility(
            datadict=datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=[sample_id],
            params=self.params,
        )

        visibility = datadict[sample_id]["visibility"]
        self.assertTrue(visibility["0_cam0"][0])

    def test_temporal_reference_masks_inconsistent_point_without_multiview_support(self):
        cameras = {"0_cam0": _make_camera([0.0, 0.0, 0.0])}
        sample_ids = ["0_99", "0_100", "0_101"]

        pose_prev = np.full((3, 23), np.nan, dtype=np.float64)
        pose_prev[:, 0] = np.array([0.0, 0.0, 1000.0], dtype=np.float64)
        pose_next = np.full((3, 23), np.nan, dtype=np.float64)
        pose_next[:, 0] = np.array([0.0, 0.0, 1000.0], dtype=np.float64)
        pose_mid = np.full((3, 23), np.nan, dtype=np.float64)

        datadict_3d = {
            "0_99": pose_prev,
            "0_100": pose_mid,
            "0_101": pose_next,
        }

        good_point = _project_point_to_camera(pose_prev[:, 0], cameras["0_cam0"])
        bad_point = good_point + np.array([60.0, 0.0], dtype=np.float64)

        datadict = {
            "0_99": {"data": {"0_cam0": np.full((2, 23), np.nan, dtype=np.float64)}, "frames": {"0_cam0": 99}},
            "0_100": {"data": {"0_cam0": np.full((2, 23), np.nan, dtype=np.float64)}, "frames": {"0_cam0": 100}},
            "0_101": {"data": {"0_cam0": np.full((2, 23), np.nan, dtype=np.float64)}, "frames": {"0_cam0": 101}},
        }
        datadict["0_99"]["data"]["0_cam0"][:, 0] = good_point
        datadict["0_100"]["data"]["0_cam0"][:, 0] = bad_point
        datadict["0_101"]["data"]["0_cam0"][:, 0] = good_point

        temporal_params = {
            **self.params,
            "exclude_occluded_2d_temporal_reference": True,
            "exclude_occluded_2d_temporal_window": 2,
        }

        populate_occlusion_visibility(
            datadict=datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=sample_ids,
            params=temporal_params,
        )

        self.assertFalse(datadict["0_100"]["visibility"]["0_cam0"][0])

    def test_reproj_only_from_stored3d_keeps_point_when_only_temporal_reference_disagrees(self):
        cameras = {"0_cam0": _make_camera([0.0, 0.0, 0.0])}
        sample_ids = ["0_99", "0_100", "0_101"]

        pose_prev = np.full((3, 23), np.nan, dtype=np.float64)
        pose_prev[:, 0] = np.array([0.0, 0.0, 1000.0], dtype=np.float64)
        pose_next = np.full((3, 23), np.nan, dtype=np.float64)
        pose_next[:, 0] = np.array([0.0, 0.0, 1000.0], dtype=np.float64)
        pose_mid = np.full((3, 23), np.nan, dtype=np.float64)

        datadict_3d = {
            "0_99": pose_prev,
            "0_100": pose_mid,
            "0_101": pose_next,
        }

        good_point = _project_point_to_camera(pose_prev[:, 0], cameras["0_cam0"])
        bad_point = good_point + np.array([60.0, 0.0], dtype=np.float64)

        datadict = {
            "0_99": {"data": {"0_cam0": np.full((2, 23), np.nan, dtype=np.float64)}, "frames": {"0_cam0": 99}},
            "0_100": {"data": {"0_cam0": np.full((2, 23), np.nan, dtype=np.float64)}, "frames": {"0_cam0": 100}},
            "0_101": {"data": {"0_cam0": np.full((2, 23), np.nan, dtype=np.float64)}, "frames": {"0_cam0": 101}},
        }
        datadict["0_99"]["data"]["0_cam0"][:, 0] = good_point
        datadict["0_100"]["data"]["0_cam0"][:, 0] = bad_point
        datadict["0_101"]["data"]["0_cam0"][:, 0] = good_point

        baseline_params = {
            **self.params,
            "exclude_occluded_2d_temporal_jump_filter": False,
            "exclude_occluded_2d_temporal_reference": True,
            "exclude_occluded_2d_temporal_window": 2,
        }
        populate_occlusion_visibility(
            datadict=datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=sample_ids,
            params=baseline_params,
        )
        self.assertFalse(datadict["0_100"]["visibility"]["0_cam0"][0])

        kept_datadict = {
            "0_99": {"data": {"0_cam0": np.full((2, 23), np.nan, dtype=np.float64)}, "frames": {"0_cam0": 99}},
            "0_100": {"data": {"0_cam0": np.full((2, 23), np.nan, dtype=np.float64)}, "frames": {"0_cam0": 100}},
            "0_101": {"data": {"0_cam0": np.full((2, 23), np.nan, dtype=np.float64)}, "frames": {"0_cam0": 101}},
        }
        kept_datadict["0_99"]["data"]["0_cam0"][:, 0] = good_point
        kept_datadict["0_100"]["data"]["0_cam0"][:, 0] = bad_point
        kept_datadict["0_101"]["data"]["0_cam0"][:, 0] = good_point
        stored3d_only_params = {
            **baseline_params,
            "exclude_occluded_2d_reproj_only_from_stored3d": True,
        }
        populate_occlusion_visibility(
            datadict=kept_datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=sample_ids,
            params=stored3d_only_params,
        )
        self.assertTrue(kept_datadict["0_100"]["visibility"]["0_cam0"][0])

    def test_temporal_2d_jump_filter_masks_outlier_even_if_stored3d_matches(self):
        cameras = {"0_cam0": _make_camera([0.0, 0.0, 0.0])}
        sample_ids = ["0_99", "0_100", "0_101"]

        pose_prev = np.full((3, 23), np.nan, dtype=np.float64)
        pose_prev[:, 0] = np.array([0.0, 0.0, 1000.0], dtype=np.float64)
        pose_next = np.full((3, 23), np.nan, dtype=np.float64)
        pose_next[:, 0] = np.array([0.0, 0.0, 1000.0], dtype=np.float64)
        pose_mid = np.full((3, 23), np.nan, dtype=np.float64)
        pose_mid[:, 0] = np.array([60.0, 0.0, 1000.0], dtype=np.float64)

        datadict_3d = {
            "0_99": pose_prev,
            "0_100": pose_mid,
            "0_101": pose_next,
        }

        good_point = _project_point_to_camera(pose_prev[:, 0], cameras["0_cam0"])
        bad_point = _project_point_to_camera(pose_mid[:, 0], cameras["0_cam0"])

        datadict = {
            "0_99": {"data": {"0_cam0": np.full((2, 23), np.nan, dtype=np.float64)}, "frames": {"0_cam0": 99}},
            "0_100": {"data": {"0_cam0": np.full((2, 23), np.nan, dtype=np.float64)}, "frames": {"0_cam0": 100}},
            "0_101": {"data": {"0_cam0": np.full((2, 23), np.nan, dtype=np.float64)}, "frames": {"0_cam0": 101}},
        }
        datadict["0_99"]["data"]["0_cam0"][:, 0] = good_point
        datadict["0_100"]["data"]["0_cam0"][:, 0] = bad_point
        datadict["0_101"]["data"]["0_cam0"][:, 0] = good_point

        jump_params = {
            **self.params,
            "exclude_occluded_2d_temporal_jump_filter": True,
            "exclude_occluded_2d_temporal_jump_window": 2,
            "exclude_occluded_2d_temporal_jump_thresh_px": 20.0,
        }

        populate_occlusion_visibility(
            datadict=datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=sample_ids,
            params=jump_params,
        )

        self.assertFalse(datadict["0_100"]["visibility"]["0_cam0"][0])

    def test_bone_prior_rescue_keeps_plausible_joint_despite_bad_reprojection(self):
        sample_ids = ["0_99", "0_100", "0_101"]
        cameras = {"0_cam0": _make_camera([0.0, 0.0, 0.0])}

        pose_good = np.full((3, 19), np.nan, dtype=np.float64)
        pose_good[:, 16] = np.array([-100.0, 0.0, 1000.0], dtype=np.float64)
        pose_good[:, 17] = np.array([0.0, 50.0, 1000.0], dtype=np.float64)
        pose_good[:, 18] = np.array([100.0, 0.0, 1000.0], dtype=np.float64)

        pose_bad = pose_good.copy()
        pose_bad[:, 17] = np.array([60.0, 50.0, 1000.0], dtype=np.float64)

        datadict_3d = {
            "0_99": pose_good,
            "0_100": pose_bad,
            "0_101": pose_good,
        }

        current_points = np.full((2, 19), np.nan, dtype=np.float64)
        for joint_idx in (16, 17, 18):
            current_points[:, joint_idx] = _project_point_to_camera(
                pose_good[:, joint_idx], cameras["0_cam0"]
            )

        datadict = {
            sample_id: {
                "data": {"0_cam0": current_points.copy()},
                "frames": {"0_cam0": frame_idx},
            }
            for sample_id, frame_idx in zip(sample_ids, (99, 100, 101))
        }

        base_params = {
            **self.params,
            "skeleton": "mouse19",
            "exclude_occluded_2d_temporal_jump_filter": False,
        }

        populate_occlusion_visibility(
            datadict=datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=sample_ids,
            params=base_params,
        )
        self.assertFalse(datadict["0_100"]["visibility"]["0_cam0"][17])

        rescue_datadict = {
            sample_id: {
                "data": {"0_cam0": current_points.copy()},
                "frames": {"0_cam0": frame_idx},
            }
            for sample_id, frame_idx in zip(sample_ids, (99, 100, 101))
        }
        rescue_params = {
            **base_params,
            "exclude_occluded_2d_bone_prior_rescue": True,
            "exclude_occluded_2d_bone_prior_max_z": 2.0,
            "exclude_occluded_2d_bone_prior_min_support_bones": 2,
        }

        populate_occlusion_visibility(
            datadict=rescue_datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=sample_ids,
            params=rescue_params,
        )
        self.assertTrue(rescue_datadict["0_100"]["visibility"]["0_cam0"][17])

    def test_com_filter_masks_joint_with_bad_anchor_relative_position(self):
        sample_ids = ["0_99", "0_100", "0_101"]
        cameras = {"0_cam0": _make_camera([0.0, 0.0, 0.0])}

        datadict_3d = {
            sample_id: np.full((3, 19), np.nan, dtype=np.float64)
            for sample_id in sample_ids
        }

        datadict = {
            sample_id: {
                "data": {"0_cam0": np.full((2, 19), np.nan, dtype=np.float64)},
                "frames": {"0_cam0": frame_idx},
            }
            for sample_id, frame_idx in zip(sample_ids, (99, 100, 101))
        }

        # Joint 17 barely moves in absolute image coordinates, so the old jump
        # filter would keep it. The sample COM shifts strongly at frame 100.
        for sample_id in sample_ids:
            datadict[sample_id]["data"]["0_cam0"][:, 17] = np.array(
                [650.0, 512.0], dtype=np.float64
            )

        com3d_dict = {
            "0_99": np.array([0.0, 0.0, 1000.0], dtype=np.float64),
            "0_100": np.array([120.0, 0.0, 1000.0], dtype=np.float64),
            "0_101": np.array([0.0, 0.0, 1000.0], dtype=np.float64),
        }

        base_params = {
            **self.params,
            "skeleton": "mouse19",
            "exclude_occluded_2d_temporal_jump_filter": False,
        }
        populate_occlusion_visibility(
            datadict=datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=sample_ids,
            params=base_params,
            com3d_dict=com3d_dict,
        )
        self.assertTrue(datadict["0_100"]["visibility"]["0_cam0"][17])

        filtered_datadict = {
            sample_id: {
                "data": {"0_cam0": datadict[sample_id]["data"]["0_cam0"].copy()},
                "frames": {"0_cam0": frame_idx},
            }
            for sample_id, frame_idx in zip(sample_ids, (99, 100, 101))
        }
        com_params = {
            **base_params,
            "exclude_occluded_2d_com_filter": True,
            "exclude_occluded_2d_com_filter_window": 2,
            "exclude_occluded_2d_com_filter_thresh_px": 50.0,
        }
        populate_occlusion_visibility(
            datadict=filtered_datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=sample_ids,
            params=com_params,
            com3d_dict=com3d_dict,
        )
        self.assertFalse(filtered_datadict["0_100"]["visibility"]["0_cam0"][17])

    def test_learned_teacher_masks_far_side_joint_through_torso(self):
        sample_ids = ["0_100", "0_101", "0_102"]
        cameras = {"0_cam0": _make_camera([0.0, 300.0, 0.0])}

        def _make_pose():
            pose = np.full((3, 19), np.nan, dtype=np.float64)
            pose[:, 3] = np.array([40.0, 0.0, 1010.0], dtype=np.float64)   # NeckB
            pose[:, 4] = np.array([15.0, 0.0, 1005.0], dtype=np.float64)   # SpineF
            pose[:, 5] = np.array([-15.0, 0.0, 995.0], dtype=np.float64)   # SpineM
            pose[:, 6] = np.array([-45.0, 0.0, 990.0], dtype=np.float64)   # Tail(base)
            pose[:, 7] = np.array([10.0, 35.0, 1002.0], dtype=np.float64)  # ForShdL
            pose[:, 10] = np.array([10.0, -35.0, 1002.0], dtype=np.float64)  # ForeShdR
            pose[:, 13] = np.array([-15.0, 30.0, 998.0], dtype=np.float64)  # HindShdL
            pose[:, 16] = np.array([-15.0, -30.0, 998.0], dtype=np.float64)  # HindShdR
            pose[:, 15] = np.array([-20.0, 80.0, 980.0], dtype=np.float64)  # HindpawL
            pose[:, 18] = np.array([-20.0, -80.0, 980.0], dtype=np.float64)  # HindpawR
            return pose

        datadict_3d = {sample_id: _make_pose() for sample_id in sample_ids}
        datadict = {}
        for sample_id in sample_ids:
            points_2d = np.full((2, 19), np.nan, dtype=np.float64)
            points_2d[:, 15] = _project_point_to_camera(
                datadict_3d[sample_id][:, 15], cameras["0_cam0"]
            )
            points_2d[:, 18] = _project_point_to_camera(
                datadict_3d[sample_id][:, 18], cameras["0_cam0"]
            )
            datadict[sample_id] = {
                "data": {"0_cam0": points_2d},
                "frames": {"0_cam0": int(sample_id.split("_")[1])},
            }

        learned_params = {
            **self.params,
            "skeleton": "mouse19",
            "exclude_occluded_2d_mode": "learned",
            "exclude_occluded_2d_temporal_jump_filter": False,
            "exclude_occluded_2d_reproj_guard_px": 1000.0,
        }
        populate_occlusion_visibility(
            datadict=datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=sample_ids,
            params=learned_params,
        )

        self.assertTrue(datadict["0_100"]["visibility"]["0_cam0"][15])
        self.assertFalse(datadict["0_100"]["visibility"]["0_cam0"][18])

    def test_learned_teacher_keeps_near_side_joint(self):
        sample_id = "0_100"
        cameras = {"0_cam0": _make_camera([0.0, -300.0, 0.0])}

        pose = np.full((3, 19), np.nan, dtype=np.float64)
        pose[:, 3] = np.array([40.0, 0.0, 1010.0], dtype=np.float64)
        pose[:, 4] = np.array([15.0, 0.0, 1005.0], dtype=np.float64)
        pose[:, 5] = np.array([-15.0, 0.0, 995.0], dtype=np.float64)
        pose[:, 6] = np.array([-45.0, 0.0, 990.0], dtype=np.float64)
        pose[:, 7] = np.array([10.0, 35.0, 1002.0], dtype=np.float64)
        pose[:, 10] = np.array([10.0, -35.0, 1002.0], dtype=np.float64)
        pose[:, 13] = np.array([-15.0, 30.0, 998.0], dtype=np.float64)
        pose[:, 16] = np.array([-15.0, -30.0, 998.0], dtype=np.float64)
        pose[:, 18] = np.array([-20.0, -80.0, 980.0], dtype=np.float64)

        points_2d = np.full((2, 19), np.nan, dtype=np.float64)
        points_2d[:, 18] = _project_point_to_camera(pose[:, 18], cameras["0_cam0"])
        datadict = {
            sample_id: {
                "data": {"0_cam0": points_2d},
                "frames": {"0_cam0": 100},
            }
        }
        datadict_3d = {sample_id: pose}

        learned_params = {
            **self.params,
            "skeleton": "mouse19",
            "exclude_occluded_2d_mode": "learned",
            "exclude_occluded_2d_temporal_jump_filter": False,
            "exclude_occluded_2d_reproj_guard_px": 1000.0,
        }
        populate_occlusion_visibility(
            datadict=datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=[sample_id],
            params=learned_params,
        )

        self.assertTrue(datadict[sample_id]["visibility"]["0_cam0"][18])

    def test_body_side_learned_alias_masks_far_side_joint(self):
        sample_id = "0_100"
        cameras = {"0_cam0": _make_camera([0.0, 300.0, 0.0])}

        pose = np.full((3, 19), np.nan, dtype=np.float64)
        pose[:, 3] = np.array([40.0, 0.0, 1010.0], dtype=np.float64)
        pose[:, 4] = np.array([15.0, 0.0, 1005.0], dtype=np.float64)
        pose[:, 5] = np.array([-15.0, 0.0, 995.0], dtype=np.float64)
        pose[:, 6] = np.array([-45.0, 0.0, 990.0], dtype=np.float64)
        pose[:, 7] = np.array([10.0, 35.0, 1002.0], dtype=np.float64)
        pose[:, 10] = np.array([10.0, -35.0, 1002.0], dtype=np.float64)
        pose[:, 13] = np.array([-15.0, 30.0, 998.0], dtype=np.float64)
        pose[:, 16] = np.array([-15.0, -30.0, 998.0], dtype=np.float64)
        pose[:, 15] = np.array([-20.0, 80.0, 980.0], dtype=np.float64)
        pose[:, 18] = np.array([-20.0, -80.0, 980.0], dtype=np.float64)

        points_2d = np.full((2, 19), np.nan, dtype=np.float64)
        points_2d[:, 15] = _project_point_to_camera(pose[:, 15], cameras["0_cam0"])
        points_2d[:, 18] = _project_point_to_camera(pose[:, 18], cameras["0_cam0"])
        datadict = {
            sample_id: {
                "data": {"0_cam0": points_2d},
                "frames": {"0_cam0": 100},
            }
        }
        datadict_3d = {sample_id: pose}

        learned_params = {
            **self.params,
            "skeleton": "mouse19",
            "exclude_occluded_2d_mode": "body_side_learned",
            "exclude_occluded_2d_temporal_jump_filter": False,
            "exclude_occluded_2d_reproj_guard_px": 1000.0,
        }
        populate_occlusion_visibility(
            datadict=datadict,
            datadict_3d=datadict_3d,
            cameras=cameras,
            sample_ids=[sample_id],
            params=learned_params,
        )

        self.assertFalse(datadict[sample_id]["visibility"]["0_cam0"][18])


if __name__ == "__main__":
    unittest.main()
