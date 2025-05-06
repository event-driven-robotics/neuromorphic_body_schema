"""
ik_test.py

Author: Ruidong Ma
Affiliation: Sheffield Hallam University
Date: 01.05.2025

Description: 
This script contains a example usage of the ik solver. It contains a sequention reaching task for two target end-effector poses.
Currently, the mujoco's PD controller is used to control the joints to reach the target joint configurations.

"""


import logging
import re
from collections import defaultdict
import os
import sys
import mujoco.viewer
current_dir = os.path.dirname(os.path.abspath(__file__))

# get the package root path
main_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
module_dir = os.path.join(main_dir)
sys.path.append(module_dir)

import mujoco
import numpy as np
from helpers.ed_cam import ICubEyes
from helpers.ed_prop import ICubProprioception
from helpers.ed_skin import ICubSkin
from helpers.helpers import MODEL_PATH, DynamicGroupedSensors, init_POV
from mujoco import viewer

from neuromorphic_body_schema.helpers.robot_controller import \
    update_joint_positions,get_joints,check_joints
    
    

DEBUG = False  # use to visualize the triangles
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

VISUALIZE_CAMERA_FEED = True
VISUALIZE_ED_CAMERA_FEED = True
VISUALIZE_SKIN = False
VISUALIZE_PROPRIOCEPTION_FEED = False
def reset(keyframe,data,model):
    data.qpos[:]=keyframe
    mujoco.mj_forward(model,data)
    
import random    
import os    

        
        
def ik_caculation(ik_solver,target_pos,target_ori,joint_names):
    try:
        q_arm=ik_solver.ik_step(target_pos,target_ori)
        joint_pose={joint_name:pose for joint_name,pose in zip(joint_names,q_arm)}
        logging.info(f"Soultion found:{joint_pose}")
        return joint_pose                            
    except ValueError as e:
        logging.info(e)
        return None
            
        
    
    
if __name__ == '__main__':
    #############################
    ### setting everything up ###
    #############################

    

    camera_name = 'front_cam'

    # Load the MuJoCo model and create a simulation
    model_path="./neuromorphic_body_schema/models/icub_v2_full_body.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
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
    
    # init example motion
    joints = ['r_shoulder_roll', 'l_shoulder_roll']  # , 'r_pinky', 'l_pinky'
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
        }
     
    }

   
    ############################
    ### Start the simulation ###
    ############################
    
    import time
    
    time.sleep(0.2)
    sim_time = data.time    
    count=0    
    joints=[]
    
    
    camera_object = ICubEyes(sim_time, model, data, camera_name,
                                 show_raw_feed=VISUALIZE_CAMERA_FEED, show_ed_feed=VISUALIZE_ED_CAMERA_FEED, DEBUG=DEBUG)
    
    
    from ik_solver import *
    
     ## Valid kinematic links shoulde be predefined 
    joint_name=["l_shoulder_pitch","l_shoulder_roll","l_shoulder_yaw","l_elbow","l_wrist_prosup"] 
    end_name="l_forearm"    
    
    ###  IK with Quaternion seems more robust for Icub 
    data_copy=copy.deepcopy(data) # should copy the data for forward knimeatics, otherwise the ik will update the model directly 
    ik_solver=Ik_solver(model,data_copy,joint_name,end_name,"quat")
    
    ##Sequential Reaching task 
    
    target_pos=[[-0.16273532, -0.23288355,  1.20810485],[-0.13067764, -0.25348467,  1.12211061]]
    target_ori=[[-0.38835096,  0.17977812, -0.02433204,  0.90347734],[-0.18286161, -0.0885009,   0.46619002 , 0.8610436 ]]
    
   
    
    
    caculated,reached,finished=False,False,False
    
    count=0
    while True:
            
            mujoco.mj_step(model, data)  # Step the simulation
            rgb,event = camera_object.update_camera(
                    data.time*1E9)  # expects ns
            if count>=len(target_pos):
                finished=True
            if not finished:
                if not caculated:
                    q_arm=ik_caculation(ik_solver,target_pos[count],target_ori[count],joint_name)
                    caculated=True
                    if not q_arm:
                        finished=True
                        logging.info(f"Solution not found, task terminated at {count}th goal")
                        continue
                        
                 #seems like the mujoco can not achieve the joints in one loop, so keep checking and control the joints
                if caculated and not check_joints(data,q_arm): 
                        update_joint_positions(data,q_arm)  #currently PD controller for the joints
                else:   
                    logging.info("Goal reached")
                    caculated=False
                    count+=1
