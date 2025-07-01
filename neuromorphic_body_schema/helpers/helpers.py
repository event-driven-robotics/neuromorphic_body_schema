import numpy as np

# set model path
# MODEL_PATH = './neuromorphic_body_schema/models/icub_v2_full_body_contact_sensors.xml'  # full iCub
# TRIANGLE_INI_PATH = "../icub-main/app/skinGui/conf/skinGui"
# FIG_PATH = "./figures"

# DEBUG
MODEL_PATH = './models/icub_v2_full_body_contact_sensors.xml'  # full iCub
TRIANGLE_INI_PATH = "../../icub-main/app/skinGui/conf/skinGui"
FIG_PATH = "../figures"

### SKIN ###
SKIN_PARTS = ["r_hand", "r_forearm", "r_upper_arm",
              "torso",
              "l_hand", "l_forearm", "l_upper_arm",
              "r_upper_leg", "r_lower_leg",
              "l_upper_leg", "l_lower_leg"]

TRIANGLE_FILES = ['right_hand_V2_2', 'right_forearm_V2', 'right_arm_V2_7',
                  'torso',
                  'left_hand_V2_2', 'left_forearm_V2', 'left_arm_V2_7',
                  'right_leg_upper', 'right_leg_lower',
                  'left_leg_upper', 'left_leg_lower']

KEY_MAPPING = {
    "right_leg_upper": "r_upper_leg_taxel",
    "left_leg_upper": "l_upper_leg_taxel",
    "right_leg_lower": "r_lower_leg_taxel",
    "left_leg_lower": "l_lower_leg_taxel",
    "r_hand": [
        "r_palm_taxel", "r_hand_thumb_taxel", "r_hand_index_taxel", "r_hand_middle_taxel", "r_hand_ring_taxel", "r_hand_pinky_taxel"],
    "torso": "torso_taxel",
    "right_forearm_V2": "r_forearm_taxel",
    "left_forearm_V2": "l_forearm_taxel",
    "right_arm_V2_7": "r_upper_arm_taxel",
    "left_arm_V2_7": "l_upper_arm_taxel",
    "l_hand": [
        "l_palm_taxel", "l_hand_thumb_taxel", "l_hand_index_taxel", "l_hand_middle_taxel", "l_hand_ring_taxel", "l_hand_pinky_taxel"],
}


class DynamicGroupedSensors:
    """
    Provides dynamic access to grouped sensor data from a simulation.

    This class allows accessing specific sensor groups dynamically by mapping keys to sensor data indices.

    Attributes:
        data (mujoco.MjData): The MuJoCo data object containing sensor data.
        grouped_sensors (dict): A dictionary mapping sensor group names to their corresponding indices in the sensor data.
    """

    def __init__(self, data, grouped_sensors):
        """
        Initializes the DynamicGroupedSensors class with sensor data and group mappings.

        Args:
            data (mujoco.MjData): The MuJoCo data object containing sensor data.
            grouped_sensors (dict): A dictionary mapping sensor group names to their corresponding indices in the sensor data.
        """

        self.data = data
        self.grouped_sensors = grouped_sensors

    def __getitem__(self, key):
        """
        Retrieves sensor data for a specific group.

        Args:
            key (str): The name of the sensor group to retrieve.

        Returns:
            np.array: The sensor data corresponding to the specified group.
        """

        adrs = self.grouped_sensors[key]
        return self.data.sensordata[adrs]


### VIEWER ###
def init_POV(viewer):
    """
    Initializes the point of view (POV) for the MuJoCo viewer.

    This function sets the camera's azimuth, distance, elevation, and look-at point to predefined values 
    for a consistent and clear view of the simulation.

    Args:
        viewer (mujoco.viewer.Viewer): The MuJoCo viewer instance to configure.

    Returns:
        mujoco.viewer.Viewer: The configured viewer instance with the updated camera settings.
    """

    viewer.cam.azimuth = -4.5
    viewer.cam.distance = 2
    viewer.cam.elevation = -16
    viewer.cam.lookat = np.array([0, -0.25, 1])

    return viewer


### PROPRIOCEPTION ###
TIME_WINDOW = 10  # seconds
TICK_HEIGHT = 20  # pixels
MARGIN = 5  # pixels
HEIGHT = 8 * (TICK_HEIGHT + MARGIN)
WIDTH = 500  # pixels
