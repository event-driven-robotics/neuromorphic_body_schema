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

import logging

import mujoco


def get_joints(data, model, joint: dict, end_effector: str = "l_wrist_1"):
    """
    Retrieves the current positions of specified joints and the pose of the end-effector.

    Args:
        data: MuJoCo MjData object.
        model: MuJoCo model object.
        joint: dict containing the controlled joint names as keys.
        end_effector (str): Name of the end-effector body (default: "l_wrist_1").

    Returns:
        dict: {
            "joints": [positions of specified joints],
            "pose": position of the end-effector (np.ndarray, shape (3,))
        }
    """
    joint_poses = {
        "joints": [data.joint(joint_name).qpos[0] for joint_name in joint.keys()],
        "pose": data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector)]
    }
    return joint_poses


def check_joints(data, joint: dict, angle_tolerance: float = 0.1) -> bool:
    """
    Checks if all specified joints have reached their target positions within a given tolerance.

    Args:
        data: MuJoCo data object.
        joint: dict mapping joint names to target positions.
        angle_tolerance (float): Allowed absolute error for each joint (default: 0.1).

    Returns:
        bool: True if all joints are within tolerance, False otherwise.
    """
    errors = [data.joint(joint_name).qpos[0] -
              target_pos for joint_name, target_pos in joint.items()]
    logging.info(f"Joint errors: {errors}")
    return all(abs(err) < angle_tolerance for err in errors)


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
        data.actuator(joint_name).ctrl[0] = target_position
