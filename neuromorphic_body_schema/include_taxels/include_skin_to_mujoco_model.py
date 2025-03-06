import os

import numpy as np
from draw_pads import fingertip2L, fingertip2R, triangle_10pad

# ini files from [...]/icub-main/app/skinGui/conf/positions/*.ini
skin_parts = ["left_arm", "left_forearm_V2", "left_hand_V2_1", "left_leg_lower", "left_leg_upper",
              "torso", "right_arm", "right_forearm_V2", "right_hand_V2_1", "right_leg_lower", "right_leg_upper"]
MOJOCO_SKIN_PARTS = ["r_upper_leg", "r_lower_leg", "l_upper_leg", "l_lower_leg", "chest", "r_shoulder_3", "r_forearm", "r_hand_dh_frame", "r_hand_thumb_3", "r_hand_index_3", "r_hand_middle_3", "r_hand_ring_3",
                     "r_hand_little_3", "l_shoulder_3", "l_forearm", "l_hand_dh_frame", "l_hand_thumb_3", "l_hand_index_3", "l_hand_middle_3", "l_hand_ring_3", "l_hand_little_3"]

### DEBUG ###
# MUJOCO_MODEL = './models/icub_v2_full_body.xml'  # DEBUG
# TAXEL_INI_PATH = "../../icub-main/app/skinGui/conf/positions"  # DEBUG
# mujoco_model_out = './models/icub_v2_full_body_contact_sensors_automated.xml'  # DEBUG

MUJOCO_MODEL = './neuromorphic_body_schema/models/icub_v2_full_body.xml'
TAXEL_INI_PATH = "../icub-main/app/skinGui/conf/positions"
mujoco_model_out = './neuromorphic_body_schema/models/icub_v2_full_body_contact_sensors_automated.xml'

# place to implement the taxels is after reading:
# <body name=MOJOCO_SKIN_PARTS[*] pos="0 0 -0.145825"> followed by <geom [...]/>

# sensor must be implemented with:
# <site name="l_lower_leg_taxel_*" size="0.005" pos="0.06 0 0.01" rgba="0 1 0 0.0"/>

# after </tendon> we need to include the taxels with
# <sensor>
#     <touch name="name_to_display" site="name_of_taxel" />
# </sensor>


def rotate_position(pos, offsets, axis, angle_degrees):
    """
    Rotates a position vector around a given axis by a specified angle.

    Args:
        pos (list or np.array): The position vector to rotate.
        axis (list of str): The axis to sequentially rotate around ('x', 'y', or 'z').
        angle_degrees (list of float): The angle to rotate by in degrees.

    Returns:
        np.array: The rotated position vector.
    """
    if 'x' in axis:
        position = axis.index('x')
        offset = offsets[position]
        angle_radians = np.radians(angle_degrees[position])
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_radians), -np.sin(angle_radians)],
                                    [0, np.sin(angle_radians), np.cos(angle_radians)]])
        pos = np.dot(rotation_matrix, pos) + offset

    if 'y' in axis:
        position = axis.index('y')
        offset = offsets[position]
        angle_radians = np.radians(angle_degrees[position])
        rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
        pos = np.dot(rotation_matrix, pos) + offset

    if 'z' in axis:
        position = axis.index('z')
        offset = offsets[position]
        angle_radians = np.radians(angle_degrees[position])
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0],
                                    [0, 0, 1]])
        pos = np.dot(rotation_matrix, pos) + offset

    if not 'x' in axis and not 'y' in axis and not 'z' in axis:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
    
    return np.round(pos, 12)


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
    all_taxels = []
    for taxels_ini in ini_files_taxels:
        calibration = read_calibration_data(f"{path_to_skin}/{taxels_ini}")
        taxels = validate_taxel_data(calibration)
        all_taxels.append(taxels)

    # now we can open the xml robot config file and start adding all the taxels
    with open(mujoco_model, 'r') as file:
        lines = file.readlines()

    finger_taxels = [
        [12e-3, 3.25e-3, -3.25e-3],
        [12e-3, 6.5e-3, 0.0],
        [12e-3, 3.25e-3, 3.25e-3],
        [13.5e-3, -1.5e-3, 6.5e-3],
        [13.5e-3, -1.5e-3, -6.5e-3],
        [15e-3, 3.25e-3, 3.25e-3],
        [15e-3, 6.5e-3, 0.0],
        [15e-3, 3.25e-3, -3.25e-3],
        [19.625e-3, 1.625e-3, -3.25e-3],
        [21.25e-3, 3.25e-3, 0.0],
        [19.625e-3, 1.625e-3, 3.25e-3],
        [24.5e-3, 0.0, 0.0]
    ]

    # now we iterate over the lines and add the taxels
    keep_processing = True
    add_finger_taxels = False
    line_counter = 0
    parts_to_add = []
    taxel_ids_to_add = []
    while keep_processing:
        print(lines[line_counter])
        if "<body name=" in lines[line_counter]:
            if any(part in lines[line_counter] for part in MOJOCO_SKIN_PARTS):
                # find the part that is being added
                part = lines[line_counter].split('name="')[1].split('"')[0]
                if part == "r_upper_leg":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "right_leg_upper.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "r_lower_leg":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "left_leg_lower.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "l_upper_leg":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "left_leg_upper.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "l_lower_leg":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "left_leg_lower.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "chest":
                    # find the corresponding taxels
                    part = "torso"
                    pos_of_taxels = ini_files_taxels.index("torso.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "r_shoulder_3":
                    # find the corresponding taxels
                    # TODO rotate by 90deg around z axis
                    part = "r_upper_arm"
                    pos_of_taxels = ini_files_taxels.index("right_arm.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "r_forearm":
                    # find the corresponding taxels
                    # TODO rotate by 90deg around z axis and possibly 180deg around x axis?
                    pos_of_taxels = ini_files_taxels.index(
                        "right_forearm_V2.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "r_hand_dh_frame":
                    # find the corresponding taxels
                    # TODO rotate by 90deg around z axis
                    part = "r_palm"
                    pos_of_taxels = ini_files_taxels.index(
                        "right_hand_V2_1.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "r_hand_thumb_3":
                    # find the corresponding taxels
                    part = "r_hand_thumb"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "r_hand_index_3":
                    # find the corresponding taxels
                    part = "r_hand_index"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "r_hand_middle_3":
                    # find the corresponding taxels
                    part = "r_hand_middle"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "r_hand_ring_3":
                    # find the corresponding taxels
                    part = "r_hand_ring"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "r_hand_little_3":
                    # find the corresponding taxels
                    part = "r_hand_little"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "l_shoulder_3":
                    # find the corresponding taxels
                    part = "l_upper_arm"
                    pos_of_taxels = ini_files_taxels.index("left_arm.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "l_forearm":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "left_forearm_V2.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "l_hand_dh_frame":
                    # find the corresponding taxels
                    part = "l_palm"
                    pos_of_taxels = ini_files_taxels.index(
                        "left_hand_V2_1.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "l_hand_thumb_3":
                    # find the corresponding taxels
                    part = "l_hand_thumb"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "l_hand_index_3":
                    # find the corresponding taxels
                    part = "l_hand_index"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "l_hand_middle_3":
                    # find the corresponding taxels
                    part = "l_hand_middle"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "l_hand_ring_3":
                    # find the corresponding taxels
                    part = "l_hand_ring"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "l_hand_little_3":
                    # find the corresponding taxels
                    part = "l_hand_little"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                if add_taxels:
                    add_taxels = False
                    if add_finger_taxels:
                        while not found_spot_to_add:
                            line_counter += 1
                            if "</body" in lines[line_counter]:
                                # read the number of identations of the line above adding the taxels:
                                identation = len(
                                    lines[line_counter].split('<')[0]) + 8
                                found_spot_to_add = True
                                taxel_ids = []
                        idx = 0
                        # line_counter -= 1
                        for taxel in finger_taxels:
                            pos = taxel
                            taxel_ids.append(idx)
                            # add the number of whitespace to the beginning of the line
                            lines.insert(
                                line_counter, f'{" "*identation}<site name="{part}_taxel_{idx}" size="0.005" pos="{pos[0]} {pos[1]} {pos[2]}" rgba="0 1 0 0.0"/>\n')
                            line_counter += 1
                            idx += 1
                        add_finger_taxels = False
                    else:
                        while not found_spot_to_add:
                            line_counter += 1
                            if "<body" in lines[line_counter]:
                                # read the number of identations of the line above adding the taxels:
                                identation = len(
                                    lines[line_counter].split('<')[0]) + 4
                                found_spot_to_add = True
                                taxel_ids = []
                        for taxel in taxels_to_add:
                            # line_counter -= 1  # we want to add the taxels before the next body
                            pos, _, idx = taxel
                            taxel_ids.append(idx)
                            # add the number of whitespace to the beginning of the line
                            if part == "r_upper_arm":
                                pos = rotate_position(pos, [0, 0, 0], ['x', 'y', 'z'], [0, 90, 90])
                            lines.insert(
                                line_counter, f'{" "*identation}<site name="{part}_taxel_{idx}" size="0.005" pos="{pos[0]} {pos[1]} {pos[2]}" rgba="0 1 0 0.0"/>\n')
                            line_counter += 1
                    line_counter -= 1  # go one line back after we added the last taxel
                    pass
                    taxel_ids_to_add.append(taxel_ids)

        elif "<tendon>" in lines[line_counter]:
            keep_processing = False

        line_counter += 1  # we need to keep iterating over the lines

    # now we have to define the sensors as touch sensors
    keep_processing = True
    identation = 4
    while keep_processing:
        # above that we need to add each taxel using:
        # <sensor>
        #     <touch name="name_to_display" site="name_of_taxel" />
        # </sensor>
        if "<contact>" in lines[line_counter]:
            for part_to_add, taxels_to_add in zip(parts_to_add, taxel_ids_to_add):
                lines.insert(line_counter, '\n')
                line_counter += 1
                lines.insert(
                    line_counter, f'{" "*identation}<!--{part_to_add}-->\n')
                line_counter += 1
                for id in taxels_to_add:
                    lines.insert(line_counter, f'{" "*identation}<sensor>\n')
                    line_counter += 1
                    lines.insert(
                        line_counter, f'{" "*(identation+4)}<touch name="{part_to_add}_{id}" site="{part_to_add}_taxel_{id}" />\n')
                    line_counter += 1
                    lines.insert(line_counter, f'{" "*identation}</sensor>\n')
                    line_counter += 1
                pass
            lines.insert(line_counter, '\n')
            keep_processing = False
            pass
        pass
        line_counter += 1
    pass

    # now we can write the new xml file
    with open(mujoco_model_out, 'w') as file:
        file.writelines(lines)
    pass


if __name__ == "__main__":
    include_skin_to_mujoco_model(MUJOCO_MODEL, TAXEL_INI_PATH, skin_parts)
    print("*******************")
    print("DONE")
    pass
