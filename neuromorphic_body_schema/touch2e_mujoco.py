import mujoco
from mujoco import viewer
import cv2
import threading
import logging
import numpy as np
import time


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def detect_contact(model, data):
    ''' Loop to monitor contact information'''
    while True:
        # print("Gonna show a contact info soon my friend. Be patient!")
        # print(sensor_data)
        sensor_data = data.sensordata
        # only output info when non-zero contact detected
        if np.sum(sensor_data) > 0.0:
            taxel = np.where(sensor_data != 0.0)[0]  # get non-zero taxels IDs
            values = sensor_data[taxel]  # get non-zero taxel readings
            sensor_data = np.column_stack((taxel, values))  # here we fuse the taxel id and sensor reading in a 2d array
            print("Sensor readings:")
            print(sensor_data)
            print('\n')

# set model path
model_path = './neuromorphic_body_schema/models/icub_v2_full_body_contact_sensors.xml'  # full iCub
# model_path = 'neuromorphic_body_schema/models/icub_v2_right_hand_mk2_contact_sensor.xml'  # right hand only
# DEBUG #
# model_path = './models/icub_v2_full_body_contact_sensors.xml'  # full iCub
# model_path = './models/icub_v2_right_hand_mk2_contact_sensor.xml'  # right hand only

# Load the MuJoCo model and create a simulation
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

threading.Thread(target=detect_contact, args=(model, data)).start()
viewer.launch(model, data)
