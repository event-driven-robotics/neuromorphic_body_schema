import mujoco
from mujoco import viewer
import cv2
# import mediapy as media
import threading
import logging
# import mujoco.viewer
import mujoco.viewer_test
import numpy as np
import time

import xml.etree.ElementTree as ET  # importa tutte funzioni che servono per manipolare files xml (grazie Fede <3)
# import threading
# import numpy as np


full_body = True
DEBUG = False

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


if full_body:
    if DEBUG:
        model_path = './models/icub_mk2_right_hand_only_contact_sensor.xml'  # right hand only
    else:
        model_path = './neuromorphic_body_schema/models/icub_v2_full_body_contact_sensors.xml'  # full iCub
else:
    if DEBUG:
        model_path = './models/icub_mk2_right_hand_only_contact_sensor.xml'  # right hand only
    else:
        # right hand only
        model_path = './neuromorphic_body_schema/models/icub_mk2_right_hand_only_contact_sensor.xml'




# Load the MuJoCo model and create a simulation
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)


print('init done')

'''
This code below is to know the int corresponding to a object type.
In case a quicker response is needed please consult https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjmodel.h 
and count literally the number of the object you need in the list under mjtObj_ 

import sys
DEFAULT_ENCODING = sys.getdefaultencoding()

def to_binary_string(s):
  """Convert text string to binary."""
  if isinstance(s, bytes):
    return s
  return s.encode(DEFAULT_ENCODING)


type_id = mujoco.mju_str2Type(to_binary_string('join'))

print(type_id)
'''


'''
How does mujoco.mj_name2id(a, b, c) work
a = model ( mujoco.MjModel.from_xml_path(model_path), or  mujoco.physics.model.from_xml_path(model_path))
b = name, string with the name we need based on the xml file (ex. "torso_pitch")
c = int, object type -> to each object type is associated a int value
    use code above if in doubt

'''
print(mujoco.mj_name2id(model,3, "torso_pitch"))

print(mujoco.mj_name2id(model,7, "front_cam"))

#print(mujoco.viewer._Simulate)
#print(viewer.Handle(cam="front_cam"))

#print(mujoco.mjtCamera(3))
#print(mujoco.MjvCamera())
#print(mujoco.MjvScene)



'''
def move_joint_to_max(model, data, viewer, joint_name, duration):
    """
    Moves a joint from its minimum value to its maximum value over a specified duration.

    Args:
        model: The MuJoCo model object.
        sim: The MuJoCo simulation object.
        joint_name: The name of the joint to move.
        duration: The time duration over which to move the joint.
    """

    viewer.launch(model, data)

    # Find the joint ID using the joint name
    joint_id = mujoco.mj_name2id(model, 3, joint_name)

    # Get joint limits
    joint_min = model.jnt_range[joint_id][0]
    joint_max = model.jnt_range[joint_id][1]

    # Calculate the total number of simulation steps
    timestep = model.opt.timestep
    n_steps = int(duration / timestep)

    # Create a linear trajectory from min to max
    trajectory = np.linspace(joint_min, joint_max, n_steps)

    # Run the simulation and apply the joint positions
    for position in trajectory:
        data.qpos[joint_id] = position
        print(position)
        mujoco.mj_step(model, data)  # Step the simulation
        viewer.render()
        time.sleep(timestep)  # Optional: add a delay to visualize the motion
'''

# Find the joint ID using the joint name
joint_id = mujoco.mj_name2id(model, 3, "torso_pitch")

# Get joint limits
joint_min = model.jnt_range[joint_id][0]
joint_max = model.jnt_range[joint_id][1]

# Calculate the total number of simulation steps
timestep = model.opt.timestep
n_steps = int(10 / timestep)

# Create a linear trajectory from min to max
trajectory = np.linspace(joint_min, joint_max, n_steps)

#viewer.launch(model, data)



with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    i=0
    viewer.cam.azimuth = -4.5
    viewer.cam.distance = 2
    viewer.cam.elevation = -16
    viewer.cam.lookat = np.array([0, -0.25, 1])
    
    while viewer.is_running():

        step_start = time.time()

        # data.qpos[joint_id] = trajectory[i]
        # #data.ctrl[0] += 0.01
        # print(trajectory[i])
        mujoco.mj_step(model, data)  # Step the simulation
        #viewer.render()
        # with viewer.lock():
        #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        i+=1
        time.sleep(timestep)  # Optional: add a delay to visualize the motion

        renderer.update_scene(data, camera = "front_cam")

# Example usage:
# move_joint_to_max(model, data, viewer,'torso_pitch', duration=10.0)
