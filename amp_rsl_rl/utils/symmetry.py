# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities to describe and apply kinematic symmetries.

The symmetry helpers exposed here allow users to describe how left/right (or any
other mirrored) joints of a robot relate to each other. They produce efficient
permutation and sign tensors that can be reused across the AMP pipeline for
augmentation and observation manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SymmetryTransform:
    """Permutation and sign information to mirror joint-space quantities.

    Attributes
    ----------
    joint_permutation:
        Long tensor of shape ``(num_joints,)`` describing how indices must be
        permuted to obtain the mirrored order. The tensor maps the original index
        ``i`` to ``joint_permutation[i]``.
    joint_sign:
        Float tensor of shape ``(num_joints,)`` encoding sign flips to apply after
        the permutation. Use ``+1`` to keep the value and ``-1`` to change sign.
    base_lin_sign:
        Float tensor of shape ``(3,)`` with sign flips for the (x, y, z) base linear
        velocity components.
    base_ang_sign:
        Float tensor of shape ``(3,)`` with sign flips for the (roll, pitch, yaw)
        base angular velocity components.
    """

    joint_permutation: torch.Tensor
    joint_sign: torch.Tensor
    base_lin_sign: torch.Tensor
    base_ang_sign: torch.Tensor

    def to(self, device: torch.device | str) -> "SymmetryTransform":
        """Returns a copy of the transform moved to ``device``."""

        dev = torch.device(device)
        return SymmetryTransform(
            joint_permutation=self.joint_permutation.to(dev),
            joint_sign=self.joint_sign.to(dev),
            base_lin_sign=self.base_lin_sign.to(dev),
            base_ang_sign=self.base_ang_sign.to(dev),
        )


@dataclass(frozen=True)
class SymmetrySpec:
    """Declarative description of the robot's mirrored joints.

    Parameters
    ----------
    joint_pairs:
        Iterable of ``(left, right)`` tuples specifying pairs of mirrored joints.
        Each joint name must appear at most once across all tuples.
    center_joints:
        Optional iterable listing joints that lie on the mirror plane (e.g., torso
        joints). These joints keep their position during mirroring. Defaults to an
        empty tuple.
    joint_sign_overrides:
        Optional mapping ``joint_name -> sign`` that allows forcing a ``+1`` or
        ``-1`` multiplier for specific joints. This is useful for revolute joints
        whose axes reverse direction when swapped between sides.
    base_linear_sign:
        Tuple of three multipliers for mirrored base linear velocity components. A
        sagittal-plane mirror typically keeps ``v_x`` and ``v_z`` while flipping
        ``v_y`` (default: ``(1.0, -1.0, 1.0)``).
    base_angular_sign:
        Tuple of three multipliers for mirrored base angular velocity components.
        Defaults to ``(1.0, -1.0, -1.0)``, assuming roll and yaw change sign when
        reflected across the sagittal plane.
    allow_unmapped:
        When ``True`` (default), joints that do not belong to ``joint_pairs`` nor
        ``center_joints`` keep their original index. Setting it to ``False`` raises
        ``ValueError`` if such joints are encountered.
    """

    joint_pairs: Sequence[Tuple[str, str]]
    center_joints: Sequence[str] = field(default_factory=tuple)
    joint_sign_overrides: Mapping[str, float] = field(default_factory=dict)
    base_linear_sign: Tuple[float, float, float] = (1.0, -1.0, 1.0)
    base_angular_sign: Tuple[float, float, float] = (1.0, -1.0, -1.0)
    allow_unmapped: bool = True

    def build_transform(self, joint_names: Sequence[str]) -> SymmetryTransform:
        """Create a :class:`SymmetryTransform` for ``joint_names``.

        Parameters
        ----------
        joint_names:
            Canonical ordering of the joints used throughout the observations and
            motion datasets.

        Returns
        -------
        SymmetryTransform
            Ready-to-use permutation and sign tensors that can be applied to joint
            position, velocity, or torque vectors that follow ``joint_names``.
        """

        joint_to_index: MutableMapping[str, int] = {
            name: i for i, name in enumerate(joint_names)
        }

        permutation = torch.arange(len(joint_names), dtype=torch.long)
        joint_sign = torch.ones(len(joint_names), dtype=torch.float32)

        # Validate uniqueness of joint usage.
        seen: set[str] = set()
        for left, right in self.joint_pairs:
            if left == right:
                raise ValueError(
                    f"Joint pair ({left}, {right}) cannot contain identical entries."
                )
            if left in seen or right in seen:
                raise ValueError(
                    f"Joint '{left if left in seen else right}' appears in multiple symmetry pairs."
                )
            seen.add(left)
            seen.add(right)

            if left not in joint_to_index:
                if self.allow_unmapped:
                    continue
                raise ValueError(
                    f"Joint '{left}' from symmetry pairs missing in provided joint list."
                )
            if right not in joint_to_index:
                if self.allow_unmapped:
                    continue
                raise ValueError(
                    f"Joint '{right}' from symmetry pairs missing in provided joint list."
                )

            left_idx = joint_to_index[left]
            right_idx = joint_to_index[right]
            permutation[left_idx] = right_idx
            permutation[right_idx] = left_idx

        center_set = set(self.center_joints)
        if len(center_set) != len(tuple(self.center_joints)):
            raise ValueError("`center_joints` must not contain duplicates.")

        for joint in center_set:
            if joint not in joint_to_index and not self.allow_unmapped:
                raise ValueError(
                    f"Center joint '{joint}' missing in provided joint list."
                )

        # Apply joint-level sign overrides.
        for joint_name, sign in self.joint_sign_overrides.items():
            if abs(sign) != 1.0:
                raise ValueError(
                    f"Sign override for joint '{joint_name}' must be either +1 or -1, got {sign}."
                )
            if joint_name not in joint_to_index:
                if self.allow_unmapped:
                    continue
                raise ValueError(
                    f"Joint '{joint_name}' lacks an index in the provided joint list."
                )
            idx = joint_to_index[joint_name]
            joint_sign[idx] = float(sign)

        # Ensure center joints keep their original indices.
        for joint in center_set:
            if joint in joint_to_index:
                idx = joint_to_index[joint]
                permutation[idx] = idx

        missing: set[str] = set(joint_names) - seen - center_set
        if not self.allow_unmapped:
            if missing:
                raise ValueError(
                    "The following joints are neither paired nor centered: "
                    + ", ".join(sorted(missing))
                )
        else:
            # Unmapped joints keep their position and default sign (already set).
            pass

        base_lin_sign = torch.tensor(self.base_linear_sign, dtype=torch.float32)
        base_ang_sign = torch.tensor(self.base_angular_sign, dtype=torch.float32)

        return SymmetryTransform(
            joint_permutation=permutation,
            joint_sign=joint_sign,
            base_lin_sign=base_lin_sign,
            base_ang_sign=base_ang_sign,
        )


def apply_joint_symmetry(
    joint_tensor: torch.Tensor,
    transform: SymmetryTransform,
) -> torch.Tensor:
    """Mirror ``joint_tensor`` using ``transform``.

    Parameters
    ----------
    joint_tensor:
        Tensor of shape ``(..., num_joints)`` following the ordering used to build
        ``transform``. The function mirrors the last dimension.
    transform:
        :class:`SymmetryTransform` built for the desired joint ordering.

    Returns
    -------
    torch.Tensor
        Mirrored tensor with the same shape as ``joint_tensor``.
    """

    mirrored = torch.index_select(
        joint_tensor, dim=-1, index=transform.joint_permutation
    )
    return mirrored * transform.joint_sign


def apply_base_symmetry(
    base_linear: torch.Tensor,
    base_angular: torch.Tensor,
    transform: SymmetryTransform,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply base velocity symmetry flips.

    Parameters
    ----------
    base_linear:
        Tensor with shape ``(..., 3)`` representing linear velocity components.
    base_angular:
        Tensor with shape ``(..., 3)`` representing angular velocity components.
    transform:
        Symmetry transform returned by :meth:`SymmetrySpec.build_transform`.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Mirrored base linear and angular velocities.
    """

    mirrored_linear = base_linear * transform.base_lin_sign
    mirrored_angular = base_angular * transform.base_ang_sign
    return mirrored_linear, mirrored_angular


def mirror_amp_observation(
    observation: torch.Tensor, transform: SymmetryTransform
) -> torch.Tensor:
    """Mirror an AMP observation vector following the canonical layout.

    The observation is expected to have the structure:

    ``[joint_pos, joint_vel, base_lin_vel, base_ang_vel]``

    Parameters
    ----------
    observation:
        Tensor whose final dimension follows the structure above.
    transform:
        Symmetry transform returned by :class:`SymmetrySpec`.

    Returns
    -------
    torch.Tensor
        Mirrored observation with the same shape as the input.
    """

    joint_dim = transform.joint_permutation.numel()
    expected_dim = 2 * joint_dim + 6
    if observation.shape[-1] != expected_dim:
        raise ValueError(
            f"AMP observation last dimension must be {expected_dim}, got {observation.shape[-1]}"
        )
    joint_pos = observation[..., :joint_dim]
    joint_vel = observation[..., joint_dim : 2 * joint_dim]
    base_lin = observation[..., 2 * joint_dim : 2 * joint_dim + 3]
    base_ang = observation[..., 2 * joint_dim + 3 : 2 * joint_dim + 6]

    mirrored_pos = apply_joint_symmetry(joint_pos, transform)
    mirrored_vel = apply_joint_symmetry(joint_vel, transform)
    mirrored_lin, mirrored_ang = apply_base_symmetry(base_lin, base_ang, transform)

    return torch.cat((mirrored_pos, mirrored_vel, mirrored_lin, mirrored_ang), dim=-1)


def mirror_amp_transition(
    state: torch.Tensor, next_state: torch.Tensor, transform: SymmetryTransform
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror a pair of AMP observations representing consecutive states."""

    return (
        mirror_amp_observation(state, transform),
        mirror_amp_observation(next_state, transform),
    )
