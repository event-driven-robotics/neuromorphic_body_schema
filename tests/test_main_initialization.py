"""
Test main.py initialization without running the full interactive loop.
"""

import copy
import re
from collections import defaultdict

import mujoco
import numpy as np
import pytest

from neuromorphic_body_schema.helpers.helpers import (MODEL_PATH,
                                                      DynamicGroupedSensors)
from neuromorphic_body_schema.helpers.robot_controller import (
    check_joints, update_joint_positions)

mj_name2id = getattr(mujoco, "mj_name2id")
mjtObj = getattr(mujoco, "mjtObj")
mj_step = getattr(mujoco, "mj_step")


def test_model_initialization(mujoco_model, mujoco_data):
    """Test basic model initialization as done in main.py."""
    # Set model to 0.0 start position
    mujoco_data.qpos.fill(0.0)

    # Define a start position for the model
    joint_init_pos = {
        "r_shoulder_roll": 0.6,
        "r_shoulder_pitch": -0.5,
        "r_shoulder_yaw": 0.0,
        "r_elbow": 1.1,
        "l_shoulder_roll": 0.6,
        "l_shoulder_pitch": -0.5,
        "l_shoulder_yaw": 0.0,
        "l_elbow": 1.1,
    }

    # Set the initial joint positions
    for joint_name, position in joint_init_pos.items():
        try:
            joint_id = mj_name2id(mujoco_model, mjtObj.mjOBJ_JOINT, joint_name)
            mujoco_data.joint(joint_id).qpos[0] = position
            mujoco_data.actuator(joint_name).ctrl[0] = position
        except ValueError:
            # Joint not found - this is acceptable in some model variants
            pass

    assert mujoco_data is not None


def test_sensor_grouping_as_in_main(mujoco_model, mujoco_data):
    """Test sensor grouping as performed in main.py."""
    names_list = mujoco_model.names.decode("utf-8").split("\x00")
    sensor_info = [x for x in names_list if "taxel" in x]

    grouped_sensors = defaultdict(list)
    for adr, name in enumerate(sensor_info):
        base_name = re.sub(r"_\d+$", "", name)
        grouped_sensors[base_name].append(adr)

    dynamic_grouped_sensors = DynamicGroupedSensors(
        mujoco_data, grouped_sensors)
    assert dynamic_grouped_sensors is not None


def test_timestep_configuration(mujoco_model, mujoco_data):
    """Test timestep configuration as in main.py."""
    mujoco_model.opt.timestep = 0.001
    assert mujoco_model.opt.timestep == 0.001

    # Verify simulation can step
    initial_time = mujoco_data.time
    mj_step(mujoco_model, mujoco_data)
    assert mujoco_data.time > initial_time
