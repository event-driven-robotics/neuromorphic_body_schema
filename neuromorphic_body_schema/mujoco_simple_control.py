import mujoco
from mujoco import viewer
import logging
import mujoco.viewer
import mujoco.viewer_test
import numpy as np
import time

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

model_path = './neuromorphic_body_schema/models/icub_v2_full_body_contact_sensors.xml'  # full iCub

# DEBUG
# model_path = './neuromorphic_body_schema/neuromorphic_body_schema/models/icub_v2_full_body_contact_sensors.xml'  # full iCub

# Load the MuJoCo model and create a simulation
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)


def init_POV(viewer):
    """
    Use:
    Inside the loop:
    with mujoco.viewer.launch_passive(model, data) as viewer:
    """
    viewer.cam.azimuth = -4.5
    viewer.cam.distance = 2
    viewer.cam.elevation = -16
    viewer.cam.lookat = np.array([0, -0.25, 1])

    return viewer

logging.info('init done')

# torso_pitch_model = model.actuator('torso_pitch')
# torso_pitch_data = data.actuator('torso_pitch')

# # Create a linear trajectory from min to max
# trajectory = np.linspace(
#     torso_pitch_model.ctrlrange[0], torso_pitch_model.ctrlrange[1], 50)


# code "wave hello"
l_shoulder_roll_model = model.actuator('l_shoulder_roll')
l_shoulder_roll_data = data.actuator('l_shoulder_roll')

l_shoulder_yaw_model = model.actuator('l_shoulder_yaw')
l_shoulder_yaw_data = data.actuator('l_shoulder_yaw')

l_elbow_model = model.actuator('l_elbow')
l_elbow_data = data.actuator('l_elbow')
l_elbow_range = np.linspace(
                        (-l_elbow_model.ctrlrange[0]+l_elbow_model.ctrlrange[1])/2, 
                        (-l_elbow_model.ctrlrange[0]+l_elbow_model.ctrlrange[1])/2, 
                        10)

l_wrist_prosup_model = model.actuator('l_wrist_prosup')
l_wrist_prosup_data = data.actuator('l_wrist_prosup')


with mujoco.viewer.launch_passive(model, data) as viewer:

    init_POV(viewer)
    
    start = time.time()
    i = 0
    
    step = 1
    while viewer.is_running():
        step_start = time.time()
        try:
            # do your stuff that modifies the data
            # torso_pitch_data.ctrl[0] = trajectory[i]
            if l_shoulder_roll_data.ctrl[0]< (l_shoulder_roll_model.ctrlrange[0]+l_shoulder_roll_model.ctrlrange[1])/2:
                l_shoulder_roll_data.ctrl[0] += 0.01
            if l_shoulder_yaw_data.ctrl[0]> l_shoulder_yaw_model.ctrlrange[0]:
                l_shoulder_yaw_data.ctrl[0] -= 0.01
            if l_wrist_prosup_data.ctrl[0]> -0.9:
                l_wrist_prosup_data.ctrl[0] -= 0.01
            l_elbow_data.ctrl[0] = (-l_elbow_model.ctrlrange[0]+l_elbow_model.ctrlrange[1])*np.sin(i/50)

        except IndexError:
            logging.info('Trajectory done')
            step *= -1
        mujoco.mj_step(model, data)  # Step the simulation
        viewer.sync()
        i += step
        time.sleep(0.5)  # Optional: add a delay to visualize the motion

        # renderer.update_scene(data, camera="front_cam")




