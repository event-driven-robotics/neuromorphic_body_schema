"""
main.py

Author: Simon F. Muller-Cleve
Affiliation: Istituto Italiano di Tecnologia (IIT)
Department: Event-Driven Perception for Robotics (EDPR)
Date: 29.04.2025

Description:
This script initializes and runs a neuromorphic simulation of the iCub robot using MuJoCo.
It integrates event-based camera, proprioception, and skin sensors, and provides visualization
options for each sensory modality.

Modules:
- ICubEyes: Simulates event-based camera functionality.
- ICubProprioception: Simulates proprioceptive spiking events.
- ICubSkin: Simulates tactile skin events.
- DynamicGroupedSensors: Provides dynamic access to grouped sensor data.
- update_joint_positions: Updates joint positions in the MuJoCo model.
- init_POV: Configures the viewer's point of view.

Usage:
Run this script to start the simulation and visualize the sensory data.

"""

# Add parent directory to path for direct script execution
import sys
from pathlib import Path

# Get the project root directory (parent of neuromorphic_body_schema)
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import copy
import logging
import math
import re
import threading
from collections import defaultdict

import mujoco
import numpy as np
from neuromorphic_body_schema.helpers.ed_cam import ICubEyes
from neuromorphic_body_schema.helpers.ed_prop import ICubProprioception
from neuromorphic_body_schema.helpers.ed_skin import ICubSkin
from neuromorphic_body_schema.helpers.helpers import (
    MODEL_PATH,
    DynamicGroupedSensors,
    init_POV,
)
from neuromorphic_body_schema.helpers.ik_solver import Ik_solver
from neuromorphic_body_schema.helpers.robot_controller import (
    check_joints,
    ik_calculation,
    reset_simulation,
    update_joint_positions,
)
from mujoco import viewer

# from helpers.ik_solver_fede import qpos_from_site_pose

DEBUG = False  # use to visualize the triangles
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# MuJoCo is a C-extension module and some symbols are not visible to static analyzers.
# Resolve once via getattr so runtime behavior stays identical while Pylance can type-check calls.
MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")
mj_name2id = getattr(mujoco, "mj_name2id")
mjtObj = getattr(mujoco, "mjtObj")
mj_step = getattr(mujoco, "mj_step")

CAMERA_MODE = "frame_based"  # "event_driven" or "frame_based"
CAM_TO_USE = "all"  # "left", "right", or "all"
VISUALIZE_CAMERA_FEED = False
# setting this to true also activates the event-based camera if not already activated, since we need it to show the feed
if CAMERA_MODE == "event_driven":
    VISUALIZE_ED_CAMERA_FEED = True
else:
    VISUALIZE_ED_CAMERA_FEED = False

# NOTE: Skin can be visualized when toggling Site group 1-5 in the MuJoCo viewer
SKIN_MODE = "frame_based"  # "event_driven" or "frame_based"
SKIN_PART = "all"  # see helpers SKIN_PARTS for the list of possible skin parts
# ["r_hand", "r_forearm", "r_upper_arm", "torso", "l_hand", "l_forearm", "l_upper_arm", "r_upper_leg", "r_lower_leg", "l_upper_leg", "l_lower_leg"]
VISUALIZE_SKIN_FEED = False
VISUALIZE_ED_SKIN_FEED = False

PROPRIOCEPTION_MODE = "event_driven"  # "event_driven" or "frame_based"
VISUALIZE_PROPRIOCEPTION_FEED = False


if __name__ == "__main__":
    #############################
    ### setting everything up ###
    #############################

    viewer_closed_event = threading.Event()

    # Load the MuJoCo model and create a simulation
    model = MjModel.from_xml_path(MODEL_PATH)
    data = MjData(model)
    # set model to 0.0 start position
    data.qpos.fill(0.0)
    # define a start position for the model
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
    # let's set the initial joint positions and actuator controls
    for joint_name, position in joint_init_pos.items():
        try:
            joint_id = mj_name2id(model, mjtObj.mjOBJ_JOINT, joint_name)
            data.joint(joint_id).qpos[0] = position
            data.actuator(joint_name).ctrl[0] = position
        except ValueError:
            logging.warning(f"Joint {joint_name} not found in the model.")
    print("Model loaded")

    # Set the time step duration to 0.001 seconds (1 milliseconds)
    model.opt.timestep = 0.001  # sec

    # prepare the mapping from skin to body parts
    names_list = model.names.decode("utf-8").split("\x00")
    sensor_info = [x for x in names_list if "taxel" in x]

    # Extract base names and group sensor addresses by base names
    grouped_sensors = defaultdict(list)
    for adr, name in enumerate(sensor_info):
        base_name = re.sub(r"_\d+$", "", name)
        grouped_sensors[base_name].append(adr)

    if DEBUG:
        for key, value in grouped_sensors.items():
            print(key, len(value))

    dynamic_grouped_sensors = DynamicGroupedSensors(data, grouped_sensors)

    joint_dict_prop = {
        "r_shoulder_roll": {
            "position_max_freq": 1000,  # Hz
            "velocity_max_freq": 1000,
            "load_max_freq": 1000,
            "limits_max_freq": 1000,
        },
        "l_shoulder_roll": {
            "position_max_freq": 1000,
            "velocity_max_freq": 1000,
            "load_max_freq": 1000,
            "limits_max_freq": 1000,
        },
        # 'r_pinky': {
        #     'position_max_freq': 1000,  # Hz
        #     'velocity_max_freq': 1000,
        #     'load_max_freq': 1000,
        #     'limits_max_freq': 1000,
        # },
        # 'l_pinky': {
        #     'position_max_freq': 1000,
        #     'velocity_max_freq': 1000,
        #     'load_max_freq': 1000,
        #     'limits_max_freq': 1000,
        # },
    }

    # # Define parameters for the sine wave
    # frequencies = [1.0, 1.0, 1.0, 1.0, 1.0]
    # min_max_pos = np.zeros((len(joints), 2))
    # for i, joint in enumerate(joints):
    #     min_max_pos[i] = model.actuator(joint).ctrlrange
    #     # to ensure we move close to the contact position
    #     if 'pinky' in joint:
    #         min_max_pos[i][0] = min_max_pos[i][1]*0.98

    ############################
    ### Start the simulation ###
    ############################

    with viewer.launch_passive(model, data) as sim_viewer:
        # Disable the left and right panels
        # sim_viewer.options.gui_left = False
        # sim_viewer.options.gui_right = False

        init_POV(sim_viewer)

        sim_time = data.time

        skin_object = ICubSkin(
            sim_time,
            dynamic_grouped_sensors,
            skin=SKIN_PART,
            skin_mode=SKIN_MODE,
            show_raw_feed=VISUALIZE_SKIN_FEED,
            show_ed_feed=VISUALIZE_ED_SKIN_FEED,
            DEBUG=DEBUG
        )
        eye_camera_object = ICubEyes(sim_time, model, data, eye=CAM_TO_USE, camera_mode=CAMERA_MODE, show_raw_feed=VISUALIZE_CAMERA_FEED, show_ed_feed=VISUALIZE_ED_CAMERA_FEED, DEBUG=DEBUG)
        # proprioception_object = ICubProprioception(
        #     model,
        #     joint_dict_prop,
        #     show_proprioception=VISUALIZE_PROPRIOCEPTION_FEED,
        #     DEBUG=DEBUG,
        # )

        # Valid kinematic links shoulde be predefined
        joint_names = [
            "l_shoulder_pitch",
            "l_shoulder_roll",
            "l_shoulder_yaw",
            "l_elbow",
            "l_wrist_prosup",
        ]
        end_effector_name = "l_forearm"

        # IK with Quaternion seems more robust for Icub
        # should copy the data for forward knimeatics, otherwise the ik will update the model directly
        data_copy = copy.deepcopy(data)
        ik_solver = Ik_solver(model, data_copy, joint_names,
                              end_effector_name, "quat")

        # Sequential Reaching task

        target_pos = np.array([
            [-0.09736548, -0.20864483, 0.94718375],
            [-0.13067764, -0.25348467, 1.12211061],
        ])
        target_ori = np.array([
            [-0.35741511, 0.26772824, 0.02986113, 0.89425071],
            [-0.18286161, -0.0885009, 0.46619002, 0.8610436],
        ])

        caculated, reached, finished = False, False, False
        q_arm: dict[str, float] | None = None

        count = 0
        while sim_viewer.is_running():
            # print(sim_time)
            mj_step(model, data)  # Step the simulation
            sim_viewer.sync()
            # sim_time_ns = data.time*1E9  # ns
            # new_joint_pos = {}
            # for (min_max, frequency, joint) in zip(min_max_pos, frequencies, joints):
            #     joint_position = min_max[0] + (min_max[1] - min_max[0]) * 0.5 * (
            #         1 + math.sin(2 * math.pi * frequency * data.time))
            #     new_joint_pos[joint] = joint_position
            #     # Update joint positions
            #     if DEBUG:
            #         print(joint, joint_position)
            # update_joint_positions(data, {joint: joint_position})

            eye_cam_events = eye_camera_object.update_camera(
                data.time * 1e9
            )  # expects ns; returns dict {"left": events, "right": events}

            skin_events = skin_object.update_skin(
                data.time * 1e9)  # expects ns

            # proprioception_events = proprioception_object.update_proprioception(
            #     time=data.time, data=data
            # )  # expects seconds

            # if count >= len(target_pos):
            #     finished = True
            # if not finished:
            #     if not caculated:
            #         q_arm = ik_calculation(
            #             ik_solver, target_pos[count], target_ori[count], joint_names
            #         )
            #         caculated = True
            #         if not q_arm:
            #             finished = True
            #             logging.info(
            #                 f"Solution not found, task terminated at {count}th goal"
            #             )
            #             continue

            #     # seems like the mujoco can not achieve the joints in one loop, so keep checking and control the joints
            #     if q_arm is not None and caculated and not check_joints(data, q_arm):
            #         # currently PD controller for the joints
            #         update_joint_positions(data, q_arm)
            #     else:
            #         logging.info("Goal reached")
            #         caculated = False
            #         count += 1

            pass
