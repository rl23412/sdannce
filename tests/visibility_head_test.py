import unittest

import torch

from dannce.engine.models.visibility_head import VisibilityHead


class VisibilityHeadNanTest(unittest.TestCase):
    def test_forward_handles_missing_joint_coords(self):
        model = VisibilityHead(
            num_joints=4,
            num_views=3,
            hidden_dim=16,
            edges=[(0, 1), (1, 2), (2, 3)],
        )
        joint_coords = torch.tensor(
            [
                [
                    [0.0, 1.0, float("nan"), 2.0],
                    [0.0, 0.5, float("nan"), -1.0],
                    [1.0, -0.5, float("nan"), 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        camera_features = torch.tensor(
            [
                [
                    [0.0, 0.0, -5.0, 0.0, 0.0, 1.0],
                    [5.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                    [-5.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )

        logits = model(joint_coords, camera_features)

        self.assertEqual(tuple(logits.shape), (1, 3, 4))
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
