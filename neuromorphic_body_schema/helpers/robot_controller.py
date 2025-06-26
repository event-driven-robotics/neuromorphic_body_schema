"""
robot_controller.py

Author: Simon F. Muller-Cleve
Affiliation: Istituto Italiano di Tecnologia (IIT)
Department: Event-Driven Perception for Robotics (EDPR)
Date: 29.04.2025

Description:
This module provides functionality for controlling the iCub robot's joints in a MuJoCo simulation. 
It includes functions to update joint positions by setting control targets for specified joints.

Functions:
- update_joint_positions: Updates the joint positions in the MuJoCo model by setting control targets.

"""

import mujoco


def get_joints(data, model, joint: dict):
    """
    Args:
      data: Mujoco.MjData
      model: Mujoco model attribute
      joint: a dict contains the controlled joints' name


    """
    joint_poses = {"joints": [], "pose": []}

    for joint_name in joint.keys():
        joint_pos = data.joint(joint_name).qpos[0]
        joint_val = data.joint(joint_name).qvel[0]
        joint_poses["joints"].append(joint_pos)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wrist_1")
    joint_poses["pose"] = data.xpos[body_id]
    return joint_poses


def check_joints(data, joint: dict):
    """
    check if reaches the target joints

    Args:
       data: mujoco data attribute
       joint: a dict contains the controlled joints' name

    """
    angle_tolerance = [0.03]*len(joint.keys())
    error = []
    for joint_name, target_pos in joint.items():
        error.append(abs(data.joint(joint_name).qpos[0]-target_pos))

    if all(err < tol for err, tol in zip(error, angle_tolerance)):
        return True
    else:
        return False


def update_joint_positions(data, joint_positions):
    """
    Updates the joint positions in the MuJoCo model by setting control targets for specified joints.

    Args:
        data (mujoco.MjData): The MuJoCo data object containing actuator controls.
        joint_positions (dict): A dictionary with joint names as keys and target positions as values.

    Returns:
        None
    """

    for joint_name, target_position in joint_positions.items():
        joint = data.actuator(joint_name)
        joint.ctrl[0] = target_position
