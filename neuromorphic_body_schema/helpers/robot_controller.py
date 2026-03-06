"""
robot_controller.py

Author: Simon F. Muller-Cleve
Affiliation: Istituto Italiano di Tecnologia (IIT)
Department: Event-Driven Perception for Robotics (EDPR)
Date: 29.04.2025

Description:
This module provides functionality for controlling the iCub robot's joints in a MuJoCo simulation.
It includes functions to update joint positions, reset simulations, and perform inverse kinematics calculations.

Functions:
- get_joints: Retrieves current positions of specified joints and end-effector pose.
- check_joints: Checks if joints have reached their target positions.
- update_joint_positions: Updates joint positions by setting control targets.
- reset_simulation: Resets the simulation to a given keyframe.
- ik_calculation: Performs inverse kinematics calculation using the provided solver.

"""

import logging
from typing import TYPE_CHECKING, Any, Optional

import mujoco
import numpy as np

if TYPE_CHECKING:
    from .ik_solver import Ik_solver

MjData = Any
MjModel = Any
mj_name2id = getattr(mujoco, "mj_name2id")
mjtObj = getattr(mujoco, "mjtObj")
mj_forward = getattr(mujoco, "mj_forward")


def get_joints(
    data: MjData, model: MjModel, joint: dict, end_effector: str
) -> dict:
    """
    Retrieves the current positions of specified joints and the pose of the end-effector.

    Args:
        data (mujoco.MjData): MuJoCo data object containing joint states.
        model (mujoco.MjModel): MuJoCo model object.
        joint (dict): Dictionary containing the controlled joint names as keys.
        end_effector (str): Name of the end-effector body.

    Returns:
        dict: A dictionary containing:
            - "joints" (list): List of current positions for the specified joints.
            - "pose" (np.ndarray): 3D position of the end-effector.
    """
    joint_poses = {
        "joints": [data.joint(joint_name).qpos[0] for joint_name in joint.keys()],
        "pose": data.xpos[mj_name2id(model, mjtObj.mjOBJ_BODY, end_effector)],
    }
    return joint_poses


def check_joints(
    data: MjData, joint: dict, angle_tolerance: float = 0.1
) -> bool:
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


def update_joint_positions(data: MjData, joint_positions: dict) -> None:
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


def reset_simulation(
    keyframe: np.ndarray, data: MjData, model: MjModel
) -> None:
    """
    Resets the simulation to the given keyframe configuration.

    Args:
        keyframe (np.ndarray): Joint positions to reset to (must match data.qpos shape).
        data (mujoco.MjData): The MuJoCo data object.
        model (mujoco.MjModel): The MuJoCo model object.

    Returns:
        None

    Raises:
        AssertionError: If keyframe shape does not match data.qpos shape.
    """
    assert (
        keyframe.shape == data.qpos.shape
    ), "Keyframe shape does not match qpos shape."
    data.qpos[:] = keyframe
    mj_forward(model, data)


def ik_calculation(
    ik_solver: "Ik_solver",
    target_pos: np.ndarray,
    target_ori: np.ndarray,
    joint_names: list,
) -> Optional[dict]:
    """
    Performs inverse kinematics calculation and returns joint positions.

    Runs the IK solver to compute joint positions that achieve the desired
    end-effector pose (position and orientation).

    Args:
        ik_solver (Ik_solver): An instance of the IK solver with an ik_step method.
        target_pos (np.ndarray): Target end-effector position [x, y, z].
        target_ori (np.ndarray): Target end-effector orientation (quaternion or rotation matrix).
        joint_names (list): List of joint names in the same order as the solver output.

    Returns:
        Optional[dict]: Dictionary mapping joint names to positions {joint_name: position}
                       if successful, None if IK fails.
    """
    try:
        q_arm = ik_solver.ik_step(target_pos, target_ori)
        joint_pose = dict(zip(joint_names, q_arm))
        logging.info(f"Solution found: {joint_pose}")
        return joint_pose
    except ValueError as e:
        logging.error(f"IK failed: {e}")
        return None

