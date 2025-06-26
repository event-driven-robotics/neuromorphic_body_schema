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


import logging
import math
import os
import random
import re
import threading
from collections import defaultdict

import mujoco
import numpy as np
from helpers.ed_cam import ICubEyes
from helpers.ed_prop import ICubProprioception
from helpers.ed_skin import ICubSkin
from helpers.helpers import MODEL_PATH, DynamicGroupedSensors, init_POV
from helpers.robot_controller import update_joint_positions, get_joints, check_joints
from mujoco import viewer

from helpers.ik_solver_fede import qpos_from_site_pose

DEBUG = False  # use to visualize the triangles
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

VISUALIZE_CAMERA_FEED = True
VISUALIZE_ED_CAMERA_FEED = False
VISUALIZE_SKIN = False
VISUALIZE_PROPRIOCEPTION_FEED = False


if __name__ == '__main__':
    #############################
    ### setting everything up ###
    #############################

    viewer_closed_event = threading.Event()

    r_eye_camera_name = 'r_eye_camera'
    l_eye_camera_name = 'l_eye_camera'

    # Load the MuJoCo model and create a simulation
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    print("Model loaded")

    # Set the time step duration to 0.001 seconds (1 milliseconds)
    model.opt.timestep = 0.001  # sec

    # prepare the mapping from skin to body parts
    names_list = model.names.decode('utf-8').split('\x00')
    sensor_info = [x for x in names_list if "taxel" in x]

    # Extract base names and group sensor addresses by base names
    grouped_sensors = defaultdict(list)
    for adr, name in enumerate(sensor_info):
        base_name = re.sub(r'_\d+$', '', name)
        grouped_sensors[base_name].append(adr)

    if DEBUG:
        for key, value in grouped_sensors.items():
            print(key, len(value))

    dynamic_grouped_sensors = DynamicGroupedSensors(data, grouped_sensors)
    # set robot to any wanted start position
    init_position = {
        'l_shoulder_pitch':-0.587,
         'l_shoulder_roll':1.13,
        'l_shoulder_yaw':0.43, 
        'l_elbow':1
    }
    sim_time = data.time
    while not check_joints(data,init_position):
        mujoco.mj_step(model,data)
        update_joint_positions(data,init_position)  
   
    print("keyframe reached")
    
    keyframe=data.qpos.copy()
    
    l_shoulder_pitcg=[-0.8,-1.28]
    l_shoulder_roll=[1.13,0.3]
    l_shoulder_yaw=[0.43,1.07]
    l_elbow=[1,1.3]
    
    
    
    init_position={
        'l_shoulder_pitch':np.random.uniform(low=-0.8,high=-1.28),
        'l_shoulder_roll':np.random.uniform(low=0.3,high=1.13),
        'l_shoulder_yaw':np.random.uniform(low=0.43,high=1.07),   
        'l_elbow':np.random.uniform(low=1,high=1.3)
    }

    joint_dict_prop = {
        'r_shoulder_roll': {
            'position_max_freq': 1000,  # Hz
            'velocity_max_freq': 1000,
            'load_max_freq': 1000,
            'limits_max_freq': 1000,
        },
        'l_shoulder_roll': {
            'position_max_freq': 1000,
            'velocity_max_freq': 1000,
            'load_max_freq': 1000,
            'limits_max_freq': 1000,
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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Disable the left and right panels
        # viewer.options.gui_left = False
        # viewer.options.gui_right = False

        init_POV(viewer)

        sim_time = data.time

        skin_object = ICubSkin(
            sim_time, dynamic_grouped_sensors, show_skin=VISUALIZE_SKIN, DEBUG=DEBUG)
        r_eye_camera_object = ICubEyes(sim_time, model, data, r_eye_camera_name,
                                       show_raw_feed=VISUALIZE_CAMERA_FEED, show_ed_feed=VISUALIZE_ED_CAMERA_FEED, DEBUG=DEBUG)
        l_eye_camera_object = ICubEyes(sim_time, model, data, l_eye_camera_name,
                                       show_raw_feed=VISUALIZE_CAMERA_FEED, show_ed_feed=VISUALIZE_ED_CAMERA_FEED, DEBUG=DEBUG)
        proprioception_object = ICubProprioception(
            model, joint_dict_prop, show_proprioception=VISUALIZE_PROPRIOCEPTION_FEED, DEBUG=DEBUG)

        # let's try the IK solver and see what it does
        r_hand_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, 'r_hand_index_taxel_5')
        r_hand_target_pose = [-0.12940918,  0.31780352,
                              0.43955828]  # data.site_xpos[r_hand_site]

        # get the target orientation as a quaternion
        # dtype = data.qpos.dtype
        # site_xmat = data.site_xmat[r_hand_site]
        # r_hand_target_orientation = np.empty(4, dtype=dtype)
        # mujoco.mju_mat2Quat(r_hand_target_orientation, site_xmat)

        r_hand_target_ori = [0.11580792, -0.46872792, -
                             0.25499586,  0.83777072]  # quaternion [x, y, z, w]
        # r_hand_ori = data.site_xmat[r_hand_site]
        # joints_to_control = ["r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw", "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow"]
        joints_to_control = ["r_shoulder_roll"]
        ik_result = qpos_from_site_pose(model=model, data=data, site_name='r_hand_index_taxel_5', target_pos=r_hand_target_pose,
                                        target_quat=None, joint_names=joints_to_control, tol=1e-8, max_steps=500, inplace=True)
        print("ik_result", ik_result)

        while viewer.is_running():
            # print(sim_time)
            mujoco.mj_step(model, data)  # Step the simulation
            viewer.sync()
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

            r_eye_cam_events = r_eye_camera_object.update_camera(
                data.time*1E9)  # expects ns
            l_eye_cam_events = l_eye_camera_object.update_camera(
                data.time*1E9)  # expects ns

            skin_events = skin_object.update_skin(data.time*1E9)  # expects ns

            proprioception_events = proprioception_object.update_proprioception(
                time=data.time, data=data)  # expects seconds

            pass
