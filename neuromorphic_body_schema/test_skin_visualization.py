import matplotlib.pyplot as plt
import numpy as np
from draw_pads import fingertip3L, fingertip3R, palmL, palmR, triangle_10pad

triangle_files = ['left_arm_V2_7', 'left_forearm_V2', 'left_hand_V2_2', 'left_leg_lower', 'left_leg_upper',
                  'torso', 'right_arm_V2_7', 'right_forearm_V2', 'right_hand_V2_2', 'right_leg_lower', 'right_leg_upper']

TRIANGLE_INI_PATH = "../icub-main/app/skinGui/conf/skinGui"


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
    config_type = []
    start_found = False
    read_header = False
    with open(file_path, 'r') as file:
        for line in file:
            if start_found and not read_header:
                if not line.strip():
                    continue
                entry = line.split()
                conf_type = entry[0]
                config_type.append(conf_type)
                triangle = list(map(float, entry[1:]))
                triangles.append((np.array(triangle[1:]), int(triangle[0])))
            if read_header:
                read_header = False
                continue
            if "[SENSORS]" in line:
                # TODO for the hand files the header comes before the [SENSORS] line, for now I change the ini files locally. Find a better solution later.
                # TODO also in torso.ini removes #leftlower
                start_found = True
                read_header = True
    return config_type, triangles


def visualize_skin(path_to_triangles, triangle_files):
    for triangles_ini in triangle_files:
        print(triangles_ini)
        config_types, triangles = read_triangle_data(
            f"{path_to_triangles}/{triangles_ini}.ini")
        tr_ID = []
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
        for tri, config_type in zip(triangles, config_types):
            # print(config_type)
            cx, cy, th, gain, lr_mirror = tri[0][0], tri[0][1], tri[0][2], tri[0][3], int(tri[0][4])
            tr_ID.append(tri[1])
            if config_type == "triangle_10pad":
                to_draw = triangle_10pad(
                    cx=cx, cy=cy, th=th, lr_mirror=lr_mirror)
            elif config_type == "fingertip3R":
                to_draw = fingertip3R(
                    cx=cx, cy=cy, th=th, lr_mirror=lr_mirror)
            elif config_type == "fingertip3L":
                to_draw = fingertip3L(
                    cx=cx, cy=cy, th=th, lr_mirror=lr_mirror)
            elif config_type == "palmR":
                to_draw = palmR(cx=cx, cy=cy, th=th, lr_mirror=1)
                for i in range(len(to_draw[0])):
                    to_draw[0][i] = to_draw[0][i] + 20.0
            elif config_type == "palmL":
                to_draw = palmL(cx=cx, cy=cy, th=th, lr_mirror=0)
                for i in range(len(to_draw[0])):
                    to_draw[0][i] = to_draw[0][i] - 20.0
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
        # now we can draw the triangles
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(dX)):
            ax.scatter(dX[i], dY[i], marker='o')
            ax.text(np.mean(dX[i]), np.mean(dY[i]), f"{tr_ID[i]}", fontsize=12, horizontalalignment='center', verticalalignment='center')
            if not (config_types[i] == "palmL" or config_types[i] == "palmR"):            
                for j in range(len(dXv[i])):
                    ax.plot([dXv[i][j-1], dXv[i][j]], [dYv[i][j-1], dYv[i][j]], linewidth=0.2, color='black')
                ax.plot([dXv[i][-1], dXv[i][0]], [dYv[i][-1], dYv[i][0]], linewidth=0.2, color='black')  # Close the triangle
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        fig.savefig(f"{triangles_ini}.pdf", bbox_inches='tight')
        plt.close(fig)
        pass
    pass


if __name__ == "__main__":
    # dX, dY, dXv, dYv, dXmin, dYmin, dXmax, dYmax, dXc, dYc, dYv =
    visualize_skin(path_to_triangles=TRIANGLE_INI_PATH,
                   triangle_files=triangle_files)
    print("All skin parts visualized")
