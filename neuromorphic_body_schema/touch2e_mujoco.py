import mujoco
from mujoco import viewer
import cv2
import threading
import logging
import numpy as np
import time


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def detect_contact(model, data):
    while True:
        print("Gonna show a contact info soon my friend. Be patient!")
        # for link in range(len(model.body_pos)):
        # print(model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_hand_thumb_3")])

        # base frame
        # print(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link"))

        # for geom1, geom2 in zip(data.contact.geom1, data.contact.geom2):
        #     geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, geom1)
        #     geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, geom2)
            
        #     print(geom1_name, geom2_name)
        print('\n\n')

# model_path = 'neuromorphic_body_schema/models/icub_right_hand_position_actuators_actuate_hands_contact_sensor.xml'  # right hand only
model_path = 'neuromorphic_body_schema/models/icub_position_actuators_actuate_hands_contact_sensors.xml'  # full iCub
# Load the MuJoCo model and create a simulation
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# threading.Thread(target=detect_contact, args=(model, data, )).start()
viewer.launch(model, data)
