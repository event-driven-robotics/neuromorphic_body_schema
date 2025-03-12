import os

import numpy as np
from draw_pads import triangle_10pad, fingertip2R, fingertip2L
# ini files from [...]/icub-main/app/skinGui/conf/positions/*.ini
skin_parts = ["left_arm", "left_forearm_V2", "left_hand_V2_1", "left_leg_lower", "left_leg_upper",
              "torso", "right_arm", "right_forearm_V2", "right_hand_V2_1", "right_leg_lower", "right_leg_upper", "palm"]
MUJOCO_MODEL = './models/icub_v2_full_body.xml'
TAXEL_INI_PATH = "../../icub-main/app/skinGui/conf/positions"
TRIANGLE_INI_PATH = "../../icub-main/app/skinGui/conf/skinGui"

mujoco_model_out = './models/icub_v2_full_body_contact_sensors.xml'


def read_calibration_data(file_path: str) -> np.array:
    """
    Reads calibration data from a given file path.

    Args:
        file_path (str): The path to the calibration data file.

    Returns:
        np.array: A numpy array containing the calibration data.
    """
    calibration = []
    start_found = False
    with open(file_path, 'r') as file:
        for line in file:
            if start_found:
                if not line.strip():
                    continue
                pos_nrm = list(map(float, line.split()))
                calibration.append(pos_nrm)
            if "[calibration]" in line:
                start_found = True
    return np.array(calibration)


def validate_taxel_data(calibration: np.array) -> list:
    """
    Validates and extracts taxel data from the calibration array.

    Args:
        calibration (np.array): A numpy array containing the calibration data.

    Returns:
        list: A list of tuples, each containing the position, normal, and index of a valid taxel.
    """
    taxels = []
    size = len(calibration)
    for i in range(1, size):
        taxel_data = calibration[i]
        pos_nrm = taxel_data[:6]
        pos = pos_nrm[:3]
        nrm = pos_nrm[3:]
        if np.linalg.norm(nrm) != 0 or np.linalg.norm(pos) != 0:
            taxels.append((pos, nrm, i - 1))
    return taxels


def read_triangle_data(file_path: str) -> np.array:
    """
    Reads triangle data from a given file path.

    Args:
        file_path (str): The path to the triangle data file.

    Returns:
        np.array: A numpy array containing the triangle data. First two entries are x and y followed by orientation and triangle index.
    """
    triangles = []
    start_found = False
    read_header = False
    with open(file_path, 'r') as file:
        for line in file:
            if start_found and not read_header:
                if not line.strip() or "# left lower" in line or 'rightupperarm_lower' in line:
                    continue
                entry = line.split()
                config_type = entry[0]
                triangle = list(map(float, entry[1:]))
                triangles.append((np.array(triangle[1:]), int(triangle[0])))
            if read_header:
                read_header = False
                continue
            if "[SENSORS]" in line:
                start_found = True
                read_header = True
    return config_type, triangles


def include_skin_to_mujoco_model(mujoco_model, path_to_skin, skin_parts):
    # read the taxel positions from the skin configuration file
    ini_files_taxels = os.listdir(path_to_skin)
    # only keep the files listed in skin_parts
    ini_files_taxels = [f for f in ini_files_taxels if f.split('.')[
        0] in skin_parts]
    # read the triangle positions from the skin configuration file
    ini_files_triangles = os.listdir(TRIANGLE_INI_PATH)
    # only keep the files listed in skin_parts
    ini_files_triangles = [f for f in ini_files_triangles if f.split('.')[
        0] in skin_parts]
    for taxels_ini, triangles_ini in zip(ini_files_taxels, ini_files_triangles):
        calibration = read_calibration_data(f"{path_to_skin}/{taxels_ini}")
        taxels = validate_taxel_data(calibration)
        config_type, triangles = read_triangle_data(f"{TRIANGLE_INI_PATH}/{triangles_ini}")
        print(config_type)
        dX = []
        dY = []
        dXv = []
        dYv = []
        dXmin = []
        dYmin = []
        dXmax = []
        dYmax = []
        dXc = []
        dYc = []
        for triangle in triangles:
            cx, cy, th, gain, layout_num = triangle[0][0], triangle[0][1], triangle[0][2], triangle[0][3], triangle[0][4]
            # triangle_10pad, fingertip2R, fingertip2L
            if config_type == "triangle_10pad":
                to_draw = triangle_10pad(cx=cx, cy=cy, th=th, gain=gain, layout_num=layout_num)
            elif config_type == "fingertip2R":
                to_draw = fingertip2R(cx=cx, cy=cy, th=th, gain=gain, layout_num=layout_num)
            elif config_type == "fingertip2L":
                to_draw = fingertip2L(cx=cx, cy=cy, th=th, gain=gain, layout_num=layout_num)
            else:
                print("Unknown config type")
            dX.append(to_draw[0])
            dY.append(to_draw[1])
            dXv.append(to_draw[2])
            dYv.append(to_draw[3])
            dXmin.append(to_draw[4])
            dYmin.append(to_draw[5])
            dXmax.append(to_draw[6])
            dYmax.append(to_draw[7])
            dXc.append(to_draw[8])
            dYc.append(to_draw[9])
            pass


if __name__ == "__main__":
    include_skin_to_mujoco_model(MUJOCO_MODEL, TAXEL_INI_PATH, skin_parts)
    pass
