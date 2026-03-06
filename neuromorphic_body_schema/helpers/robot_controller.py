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
        data (mujoco.MjData): MuJoCo data object containing joint states.
        model (mujoco.MjModel): MuJoCo model object.
        joint (dict): Dictionary containing the controlled joint names as keys.
        end_effector (str): Name of the end-effector body. Default is "l_wrist_1".

    Returns:
        dict: A dictionary containing:
            - "joints" (list): List of current positions for the specified joints.
            - "pose" (np.ndarray): 3D position of the end-effector.
    """
    joint_poses = {
        "joints": [data.joint(joint_name).qpos[0] for joint_name in joint.keys()],
        "pose": data.xpos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector)
        ],
    }
    return joint_poses


def check_joints(data, joint: dict, angle_tolerance: float = 0.1) -> bool:
    """
    Checks if all specified joints have reached their target positions within a given tolerance.

    Args:
        data (mujoco.MjData): MuJoCo data object containing joint states.
        joint (dict): Dictionary mapping joint names (str) to target positions (float).
        angle_tolerance (float): Allowed absolute error for each joint in radians. Default is 0.1.

    Returns:
        bool: True if all joints are within tolerance, False otherwise.
    """
    errors = [
        data.joint(joint_name).qpos[0] - target_pos
        for joint_name, target_pos in joint.items()
    ]
    logging.info(f"Joint errors: {errors}")
    return all(abs(err) < angle_tolerance for err in errors)


def update_joint_positions(data, joint_positions):
    """
    Updates the joint positions in the MuJoCo simulation by setting control targets for specified actuators.

    This function sets the control values for actuators corresponding to each joint,
    which will be used by the controller to move the joints to the target positions.

    Args:
        data (mujoco.MjData): The MuJoCo data object containing actuator controls.
        joint_positions (dict): A dictionary mapping joint names (str) to target positions (float).
    """
    for joint_name, target_position in joint_positions.items():
        data.actuator(joint_name).ctrl[0] = target_position
