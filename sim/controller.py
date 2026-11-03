"""Geometric (Lee et al. 2010) velocity controller that turns velocity commands into a body wrench.

Commands per env: (v_forward, v_left, z_ref, yaw_rate) with the horizontal velocity expressed in the drone's
yaw frame. Output: collective thrust along body +Z and a body torque, applied to the base link as an external
wrench (the Crazyflie USD has spinning props but no thrust model of its own).
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import matrix_from_quat

GRAVITY = 9.81


def _vee(m: torch.Tensor) -> torch.Tensor:
    return torch.stack([m[:, 2, 1], m[:, 0, 2], m[:, 1, 0]], dim=-1)


class VelocityController:
    def __init__(
        self,
        mass: torch.Tensor,
        inertia: torch.Tensor,
        k_vel: float = 2.5,
        k_alt: float = 2.0,
        k_rot: float = 150.0,
        k_omega: float = 22.0,
        max_tilt_deg: float = 30.0,
    ):
        """`mass` (N,), `inertia` (N, 3, 3) of the whole vehicle in the body frame."""
        self.mass, self.inertia = mass, inertia
        self.k_vel, self.k_alt, self.k_rot, self.k_omega = k_vel, k_alt, k_rot, k_omega
        self.max_horizontal_acc = GRAVITY * torch.tan(torch.deg2rad(torch.tensor(max_tilt_deg))).item()
        self.yaw_ref: torch.Tensor | None = None

    def reset(self, quat_w: torch.Tensor, env_ids: torch.Tensor | None = None):
        yaw = self._yaw(matrix_from_quat(quat_w))
        if self.yaw_ref is None or env_ids is None:
            self.yaw_ref = yaw.clone()
        else:
            self.yaw_ref[env_ids] = yaw[env_ids]

    @staticmethod
    def _yaw(rot: torch.Tensor) -> torch.Tensor:
        return torch.atan2(rot[:, 1, 0], rot[:, 0, 0])

    def __call__(
        self,
        cmd: torch.Tensor,
        pos_w: torch.Tensor,
        quat_w: torch.Tensor,
        lin_vel_w: torch.Tensor,
        ang_vel_b: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (force_b, torque_b), each (N, 3)."""
        rot = matrix_from_quat(quat_w)
        if self.yaw_ref is None:
            self.reset(quat_w)
        self.yaw_ref = self.yaw_ref + cmd[:, 3] * dt
        cy, sy = torch.cos(self.yaw_ref), torch.sin(self.yaw_ref)

        # Velocity -> desired acceleration (world frame).
        v_des = torch.stack(
            [cy * cmd[:, 0] - sy * cmd[:, 1], sy * cmd[:, 0] + cy * cmd[:, 1], self.k_alt * (cmd[:, 2] - pos_w[:, 2])],
            dim=-1,
        )
        acc = self.k_vel * (v_des - lin_vel_w)
        horiz = acc[:, :2]
        scale = (self.max_horizontal_acc / horiz.norm(dim=-1).clamp(min=1e-6)).clamp(max=1.0)
        acc = torch.cat([horiz * scale[:, None], acc[:, 2:].clamp(-0.5 * GRAVITY, GRAVITY)], dim=-1)
        force_w = self.mass[:, None] * (acc + torch.tensor([0.0, 0.0, GRAVITY], device=acc.device))

        # Desired attitude: thrust axis along force_w, heading yaw_ref.
        b3 = force_w / force_w.norm(dim=-1, keepdim=True)
        b1c = torch.stack([cy, sy, torch.zeros_like(cy)], dim=-1)
        b2 = torch.linalg.cross(b3, b1c)
        b2 = b2 / b2.norm(dim=-1, keepdim=True)
        rot_des = torch.stack([torch.linalg.cross(b2, b3), b2, b3], dim=-1)

        thrust = (force_w * rot[:, :, 2]).sum(-1).clamp(min=0.0)
        e_rot = 0.5 * _vee(rot_des.transpose(1, 2) @ rot - rot.transpose(1, 2) @ rot_des)
        omega_des = torch.zeros_like(ang_vel_b)
        omega_des[:, 2] = cmd[:, 3]
        e_omega = ang_vel_b - omega_des
        j_omega = (self.inertia @ ang_vel_b[..., None]).squeeze(-1)
        torque = (self.inertia @ (-self.k_rot * e_rot - self.k_omega * e_omega)[..., None]).squeeze(-1)
        torque = torque + torch.linalg.cross(ang_vel_b, j_omega)

        force_b = torch.zeros_like(force_w)
        force_b[:, 2] = thrust
        return force_b, torque
