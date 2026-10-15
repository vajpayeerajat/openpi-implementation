"""Policy transforms for the Unitree G1 + Dex3 hands (upper body, 28 DoF).

Matches the LeRobot v2.1 dataset at `splitted_lerobot_and_school/{train,val,test}`:

* `observation.state` / `action` are 28-dim absolute joint positions, ordered
  `[L_arm 7, L_hand 7, R_arm 7, R_hand 7]` (see `G1_JOINTS`). The dataset was already
  sliced from the 43-dim whole-body vector (indices 15..42), so no slicing is needed for
  training. At serve time the robot may still send the full 43-dim vector; both are accepted.
* Cameras: `ego_view` (head), `ego_left` / `ego_right` (wrists), 640x480 @ 20 fps.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms

G1_DIM = 28

# Upper body inside the 43-dim whole-body vector (legs 0..11 and waist 12..14 are dropped).
_WHOLE_BODY_DIM = 43
_UPPER_BODY = slice(15, 43)

# Order of the 28 dims. The robot must apply the policy outputs to these joints in this order.
G1_JOINTS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
)  # fmt: skip

# Arm joints of the 28 dims. They interleave as [arm, hand, arm, hand], so this is NOT make_bool_mask(14, -14).
ARM_MASK = transforms.make_bool_mask(7, -7, 7, -7)

# Delta on the arms, absolute on the hands.
DELTA_ACTION_MASK = ARM_MASK


def make_g1_dex3_example() -> dict:
    """Creates a random input example for the G1 Dex3 policy, in the format the robot client sends."""
    return {
        "observation/state": np.zeros((G1_DIM,), dtype=np.float32),
        "observation/ego_view": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/ego_left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/ego_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "Pick up the pink octopus and place it inside the orange basket.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _upper_body(x, name: str) -> np.ndarray:
    """Returns the 28 trained dims from either a 28-dim or a 43-dim whole-body vector."""
    x = np.asarray(x)
    if x.shape[-1] == G1_DIM:
        return x
    if x.shape[-1] == _WHOLE_BODY_DIM:
        return x[..., _UPPER_BODY]
    raise ValueError(f"{name}: expected last dim {G1_DIM} or {_WHOLE_BODY_DIM}, got {x.shape[-1]}")


@dataclasses.dataclass(frozen=True)
class G1Dex3Inputs(transforms.DataTransformFn):
    """Inputs for the G1 Dex3 policy.

    Expected inputs:
    - observation/ego_view, observation/ego_left, observation/ego_right: [H, W, C] or [C, H, W] images
    - observation/state: [28] (or [43] whole body)
    - actions: [action_horizon, 28] (training only)
    """

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": _upper_body(data["observation/state"], "state"),
            "image": {
                "base_0_rgb": _parse_image(data["observation/ego_view"]),
                "left_wrist_0_rgb": _parse_image(data["observation/ego_left"]),
                "right_wrist_0_rgb": _parse_image(data["observation/ego_right"]),
            },
            # All three cameras exist for every frame of every episode.
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training. They are padded to the model action dim
        # later by the model transforms.
        if "actions" in data:
            inputs["actions"] = _upper_body(data["actions"], "actions")

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Dex3Outputs(transforms.DataTransformFn):
    """Outputs for the G1 Dex3 policy: [action_horizon, 28] absolute joint targets in `G1_JOINTS` order."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :G1_DIM])}
