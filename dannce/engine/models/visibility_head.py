from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisibilityHead(nn.Module):
    """Predict per-view visibility logits from pose and camera geometry only."""

    def __init__(
        self,
        num_joints: int,
        num_views: int,
        hidden_dim: int = 128,
        edges: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.num_joints = int(num_joints)
        self.num_views = int(num_views)
        self.hidden_dim = int(hidden_dim)
        self.pose_encoder = nn.Sequential(
            nn.Linear(self.num_joints * 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.local_feature_dim = 9
        self.joint_view_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.local_feature_dim + 11, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self._register_neighbors(edges)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _register_neighbors(self, edges: list[tuple[int, int]] | None):
        adjacency = [[] for _ in range(self.num_joints)]
        for edge in edges or []:
            if len(edge) < 2:
                continue
            joint_a, joint_b = int(edge[0]), int(edge[1])
            if not (0 <= joint_a < self.num_joints and 0 <= joint_b < self.num_joints):
                continue
            adjacency[joint_a].append(joint_b)
            adjacency[joint_b].append(joint_a)

        max_neighbors = max((len(neighbors) for neighbors in adjacency), default=0)
        if max_neighbors == 0:
            neighbor_index = torch.full((self.num_joints, 1), -1, dtype=torch.long)
            neighbor_mask = torch.zeros((self.num_joints, 1), dtype=torch.bool)
        else:
            neighbor_index = torch.full(
                (self.num_joints, max_neighbors),
                -1,
                dtype=torch.long,
            )
            neighbor_mask = torch.zeros(
                (self.num_joints, max_neighbors),
                dtype=torch.bool,
            )
            for joint_idx, neighbors in enumerate(adjacency):
                for neighbor_slot, neighbor_idx in enumerate(neighbors):
                    neighbor_index[joint_idx, neighbor_slot] = neighbor_idx
                    neighbor_mask[joint_idx, neighbor_slot] = True

        self.register_buffer("neighbor_index", neighbor_index, persistent=False)
        self.register_buffer("neighbor_mask", neighbor_mask, persistent=False)

    def forward(
        self,
        joint_coords: torch.Tensor | None,
        camera_features: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if joint_coords is None or camera_features is None:
            return None
        if joint_coords.ndim != 3 or joint_coords.shape[1] != 3:
            raise ValueError(
                "joint_coords must have shape [B, 3, J] for pose-only visibility."
            )
        if camera_features.ndim != 3 or camera_features.shape[-1] != 6:
            raise ValueError(
                "camera_features must have shape [B, V, 6] containing camera centers and forward vectors."
            )
        if joint_coords.shape[2] != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joints but received {joint_coords.shape[2]}."
            )
        if camera_features.shape[1] != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} camera views but received {camera_features.shape[1]}."
            )

        coords = joint_coords.transpose(1, 2).contiguous()
        valid_mask = torch.isfinite(coords).all(dim=-1, keepdim=True)
        safe_coords = torch.where(valid_mask, coords, torch.zeros_like(coords))
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pose_center = safe_coords.sum(dim=1, keepdim=True) / valid_counts
        centered_coords = torch.where(
            valid_mask,
            safe_coords - pose_center,
            torch.zeros_like(safe_coords),
        )
        pose_scale = centered_coords.norm(dim=-1)
        pose_scale = pose_scale.masked_fill(~valid_mask.squeeze(-1), 0.0)
        pose_scale = pose_scale.amax(dim=1, keepdim=True)
        pose_scale = pose_scale.clamp_min(1e-6).unsqueeze(-1)
        normalized_coords = centered_coords / pose_scale

        pose_embedding = self.pose_encoder(
            normalized_coords.reshape(normalized_coords.shape[0], -1)
        )

        local_pose_features = self._build_local_pose_features(
            normalized_coords,
            valid_mask,
        )

        camera_centers = camera_features[..., :3]
        camera_forward = F.normalize(camera_features[..., 3:6], dim=-1, eps=1e-6)
        centered_camera = camera_centers - pose_center
        normalized_camera = centered_camera / pose_scale
        camera_direction = F.normalize(normalized_camera, dim=-1, eps=1e-6)

        rays = normalized_coords.unsqueeze(1) - normalized_camera.unsqueeze(2)
        ray_distance = rays.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ray_direction = rays / ray_distance
        view_alignment = (
            ray_direction * camera_forward.unsqueeze(2)
        ).sum(dim=-1, keepdim=True)

        joint_view_features = torch.cat(
            (
                local_pose_features.unsqueeze(1).expand(-1, self.num_views, -1, -1),
                camera_direction.unsqueeze(2).expand(-1, -1, self.num_joints, -1),
                camera_forward.unsqueeze(2).expand(-1, -1, self.num_joints, -1),
                ray_direction,
                torch.log1p(ray_distance),
                view_alignment,
                pose_embedding.unsqueeze(1).unsqueeze(2).expand(
                    -1, self.num_views, self.num_joints, -1
                ),
            ),
            dim=-1,
        )
        logits = self.joint_view_mlp(joint_view_features).squeeze(-1)
        return logits

    def _build_local_pose_features(
        self,
        normalized_coords: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        joint_radius = normalized_coords.norm(dim=-1, keepdim=True)
        joint_radius = torch.where(valid_mask, joint_radius, torch.zeros_like(joint_radius))
        if self.neighbor_mask.any():
            neighbor_index = self.neighbor_index.clamp_min(0)
            neighbor_coords = normalized_coords[:, neighbor_index, :]
            joint_coords = normalized_coords.unsqueeze(2)
            neighbor_mask = self.neighbor_mask.unsqueeze(0).unsqueeze(-1)
            neighbor_valid = valid_mask[:, neighbor_index, :]
            effective_mask = neighbor_mask & neighbor_valid & valid_mask.unsqueeze(2)
            neighbor_offsets = (neighbor_coords - joint_coords) * neighbor_mask
            neighbor_offsets = neighbor_offsets * effective_mask
            neighbor_counts = effective_mask.sum(dim=2).clamp_min(1.0)
            mean_neighbor_offset = neighbor_offsets.sum(dim=2) / neighbor_counts
            neighbor_extent = (
                neighbor_offsets.norm(dim=-1) * effective_mask.squeeze(-1)
            ).amax(dim=2, keepdim=True)
        else:
            mean_neighbor_offset = torch.zeros_like(normalized_coords)
            neighbor_extent = torch.zeros_like(joint_radius)

        return torch.cat(
            (
                normalized_coords,
                mean_neighbor_offset,
                joint_radius,
                neighbor_extent,
                valid_mask.float(),
            ),
            dim=-1,
        )
