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
