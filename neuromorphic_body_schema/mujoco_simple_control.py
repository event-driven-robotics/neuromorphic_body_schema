import mujoco
from mujoco import viewer
import logging
import mujoco.viewer
import mujoco.viewer_test
import numpy as np
import time

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

logging.info('init done')

torso_pitch_model = model.actuator('torso_pitch')
torso_pitch_data = data.actuator('torso_pitch')

# Create a linear trajectory from min to max
trajectory = np.linspace(
    torso_pitch_model.ctrlrange[0], torso_pitch_model.ctrlrange[1], 50)

with mujoco.viewer.launch_passive(model, data) as viewer:

    start = time.time()
    i = 0
    viewer.cam.azimuth = -4.5
    viewer.cam.distance = 2
    viewer.cam.elevation = -16
    viewer.cam.lookat = np.array([0, -0.25, 1])
    step = 1
    while viewer.is_running():
        step_start = time.time()
        try:
            torso_pitch_data.ctrl[0] = trajectory[i]
        except IndexError:
            logging.info('Trajectory done')
            step *= -1
        mujoco.mj_step(model, data)  # Step the simulation
        viewer.sync()
        i += step
        time.sleep(0.1)  # Optional: add a delay to visualize the motion

        renderer.update_scene(data, camera="front_cam")
