import logging
import math
import re
import threading
from collections import defaultdict

import mujoco
import numpy as np
from draw_pads import fingertip3L, fingertip3R, palmL, palmR, triangle_10pad
from mujoco import viewer


from ed_cam import ICubEyes
from ed_skin import ICubSkin
from ed_prop import ICubProprioception
from helpers import MODEL_PATH, DynamicGroupedSensors, init_POV
from robot_controller import update_joint_positions

DEBUG = False  # use to visualize the triangles
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

VISUALIZE_CAMERA_FEED = False
VISUALIZE_ED_CAMERA_FEED = False
VISUALIZE_SKIN = False
VISUALIZE_PROP_FEED = False




if __name__ == '__main__':
    #############################
    ### setting everything up ###
    #############################

    viewer_closed_event = threading.Event()

    camera_name = 'head_cam'

    # Load the MuJoCo model and create a simulation
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    print("Model loaded")

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
        'r_shoulder_roll': 0.5,
        'l_shoulder_roll': 0.5
    }
    update_joint_positions(data, init_position)

    # init example motion
    # joints = ['r_index_proximal', 'r_index_distal', 'r_middle_proximal', 'r_middle_distal']
    joints = ['r_shoulder_roll', 'l_shoulder_roll']
    joint_dict_prop = {
        'r_shoulder_roll': {
            'position_max_freq': 0.001,
            'velocity_max_freq': 0.001,
            'load_max_freq': 0.001,
            'limits_max_freq': 0.001,
        },
        'l_shoulder_roll': {
            'position_max_freq': 0.001,
            'velocity_max_freq': 0.001,
            'load_max_freq': 0.001,
            'limits_max_freq': 0.001,
        },
    }

    # Define parameters for the sine wave
    frequencies = [0.005, 0.001]
    min_max_pos = np.zeros((len(joints), 2))
    for i, joint in enumerate(joints):
        min_max_pos[i] = model.actuator(joint).ctrlrange*0.8
        # to ensure we move close to the contact position
        if 'pinky' in joint:
            min_max_pos[i][0] = min_max_pos[i][1]

    skin_initialized = False
    esim_cam = None
    proprioception_initialized = False

    ############################
    ### Start the simulation ###
    ############################

    with mujoco.viewer.launch_passive(model, data) as viewer:

        init_POV(viewer)

        sim_time = 0
        timestep = 0
        
        skin_object = ICubSkin(sim_time, dynamic_grouped_sensors, show_skin=VISUALIZE_SKIN, DEBUG=DEBUG)
        camera_object = ICubEyes(sim_time, model, data, camera_name, show_raw_feed=VISUALIZE_CAMERA_FEED, show_ed_feed=VISUALIZE_ED_CAMERA_FEED, DEBUG=DEBUG)
        proprioception_object = ICubProprioception(model, joint_dict_prop, show_proprioception=VISUALIZE_PROP_FEED, DEBUG=DEBUG)

        while viewer.is_running():
            #print(sim_time)
            mujoco.mj_step(model, data)  # Step the simulation
            viewer.sync()
            sim_time += 1000  # 1 ms
            timestep += 1
            if timestep == 53:
                pass
            print('timestep', timestep)

            for (min_max, frequency, joint) in zip(min_max_pos, frequencies, joints):
                scaled_time = sim_time / 1000
                joint_position = min_max[0] + (min_max[1] - min_max[0]) * 0.5 * (1 + math.sin(2 * math.pi * frequency * scaled_time))
                # Update joint positions
                if DEBUG:
                    print(joint, joint_position)
                update_joint_positions(data, {joint: joint_position})

            cam_events = camera_object.update_camera(sim_time)

            skin_events = skin_object.update_skin(sim_time)
            
            prop_events = proprioception_object.update_proprioception(time=sim_time, data=data)  
                
            pass


            
        # cv2.destroyAllWindows()




