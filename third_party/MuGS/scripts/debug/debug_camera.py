#!/usr/bin/env python3
"""Debug camera view and object positions"""

import mujoco
import json
import numpy as np
from pathlib import Path

# Load scene
scene_path = Path("scenes/first_person_kitchen.xml")
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

# Load camera 117 params
with open("data/pretrained/kitchen/cameras.json") as f:
    cams = json.load(f)
cam117 = cams[117]

print("Camera 117:")
print(f"  Position: {cam117['position']}")

# Get camera transform
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
mujoco.mj_forward(model, data)

print(f"\nMuJoCo wrist_cam:")
print(f"  ID: {cam_id}")
print(f"  Position: {data.cam_xpos[cam_id]}")

# Get all object positions
print("\nObject positions:")
for name in ['mug', 'bowl', 'plate', 'apple']:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    pos = data.xpos[body_id]
    print(f"  {name:6s}: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")

# Calculate distances from camera
cam_pos = np.array(cam117['position'])
print(f"\nDistances from Camera 117:")
for name in ['mug', 'bowl', 'plate', 'apple']:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    pos = data.xpos[body_id]
    dist = np.linalg.norm(pos - cam_pos)
    print(f"  {name:6s}: {dist:.3f}m")

# Camera direction
rot_mat = np.array(cam117['rotation'])
forward = -rot_mat[:, 2]  # Camera -Z is forward
print(f"\nCamera forward direction: [{forward[0]:.3f}, {forward[1]:.3f}, {forward[2]:.3f}]")

# Check if objects are in front of camera
print(f"\nObjects in front of camera?")
for name in ['mug', 'bowl', 'plate', 'apple']:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    pos = data.xpos[body_id]
    to_obj = pos - cam_pos
    dot = np.dot(to_obj, forward)
    in_front = "YES" if dot > 0 else "NO"
    angle = np.arccos(np.clip(np.dot(to_obj / np.linalg.norm(to_obj), forward), -1, 1)) * 180 / np.pi
    print(f"  {name:6s}: {in_front:3s}  (angle={angle:.1f}°, dot={dot:.3f})")
