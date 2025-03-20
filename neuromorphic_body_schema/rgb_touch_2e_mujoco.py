import logging
import math
import re
import threading
import time
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from draw_pads import fingertip3L, fingertip3R, palmL, palmR, triangle_10pad
from mujoco import viewer

DEBUG = False  # use to visualize the triangles
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

VISUALIZE_CAMERA_FEED = False
VISUALIZE_ED_CAMERA_FEED = False
VISUALIZE_SKIN = True

# set model path
MODEL_PATH = './neuromorphic_body_schema/models/icub_v2_full_body_contact_sensors.xml'  # full iCub
TRIANGLE_INI_PATH = "../icub-main/app/skinGui/conf/skinGui"
FIG_PATH = "./figures"

# DEBUG
# MODEL_PATH = './models/icub_v2_full_body_contact_sensors.xml'  # full iCub
# TRIANGLE_INI_PATH = "../../icub-main/app/skinGui/conf/skinGui"
# FIG_PATH = "../figures"

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
    def __init__(self, data, grouped_sensors):
        self.data = data
        self.grouped_sensors = grouped_sensors

    def __getitem__(self, key):
        adrs = self.grouped_sensors[key]
        return self.data.sensordata[adrs]
    

class CameraEventSimulator:
    def __init__(self, img, time, Cp=0.5, Cm=0.5, sigma_Cp=0.01, sigma_Cm=0.01, log_eps=1e-6, refractory_period_ns=100, use_log_image=True):
        self.Cp = Cp
        self.Cm = Cm
        self.sigma_Cp = sigma_Cp
        self.sigma_Cm = sigma_Cm
        self.log_eps = log_eps
        self.use_log_image = use_log_image
        self.refractory_period_ns = refractory_period_ns
        logging.info(
            f"Initialized event camera simulator with sensor size: {img.shape}")
        logging.info(
            f"and contrast thresholds: C+ = {self.Cp}, C- = {self.Cm}")

        if self.use_log_image:
            logging.info(
                f"Converting the image to log image with eps = {self.log_eps}.")
            img = cv2.log(self.log_eps + img)
        self.last_img = img.copy()
        self.ref_values = img.copy()
        self.last_event_timestamp = np.zeros(img.shape, dtype=np.ulonglong)
        self.current_time = time
        self.size = img.shape[:2]

    def imageCallback(self, img, time):
        assert time >= 0

        if self.use_log_image:
            img = cv2.log(self.log_eps + img)

        # For each pixel, check if new events need to be generated since the last image sample
        tolerance = 1e-6
        events = []
        delta_t_ns = time - self.current_time

        assert delta_t_ns > 0, "delta_t_ns must be greater than 0."
        # assert img.size() == self.size

        for y in range(self.size[0]):
            for x in range(self.size[1]):
                itdt = img[y, x]
                it = self.last_img[y, x]
                prev_cross = self.ref_values[y, x]

                if abs(it - itdt) > tolerance:
                    pol = +1.0 if itdt >= it else -1.0
                    C = self.Cp if pol > 0 else self.Cm
                    sigma_C = self.sigma_Cp if pol > 0 else self.sigma_Cm

                    if sigma_C > 0:
                        C += np.random.normal(0, sigma_C)
                        minimum_contrast_threshold = 0.01
                        C = max(minimum_contrast_threshold, C)

                    curr_cross = prev_cross
                    all_crossings = False

                    while not all_crossings:
                        curr_cross += pol * C

                        if (pol > 0 and curr_cross > it and curr_cross <= itdt) or \
                                (pol < 0 and curr_cross < it and curr_cross >= itdt):

                            edt = int(abs((curr_cross - it) *
                                      delta_t_ns / (itdt - it)))
                            t = self.current_time + edt

                            # check that pixel (x,y) is not currently in a "refractory" state
                            # i.e. |t-that last_timestamp(x,y)| >= refractory_period
                            last_stamp_at_xy = self.last_event_timestamp[y, x]
                            assert t >= last_stamp_at_xy
                            dt = t - last_stamp_at_xy

                            if self.last_event_timestamp[y, x] == 0 or dt >= self.refractory_period_ns:
                                events.append((x, y, t, pol > 0))
                                self.last_event_timestamp[y, x] = t
                            else:
                                logging.info(
                                    f"Dropping event because time since last event ({dt} ns) < refractory period ({self.refractory_period_ns} ns).")
                            self.ref_values[y, x] = curr_cross
                        else:
                            all_crossings = True
                # end tolerance

        # update simvars for next loop
        self.current_time = time
        self.last_img = img.copy()  # it is now the latest image

        # Sort the events by increasing timestamps, since this is what
        # most event processing algorithms expect

        events = np.array(events)
        if len(events):
            events[np.argsort(events[:, 2])]
        return events


class SkinEventSimulator:
    def __init__(self, data, time, Cp=0.5, Cm=0.5, sigma_Cp=0.01, sigma_Cm=0.01, log_eps=1e-6, refractory_period_ns=100):
        self.Cp = Cp
        self.Cm = Cm
        self.sigma_Cp = sigma_Cp
        self.sigma_Cm = sigma_Cm
        self.log_eps = log_eps
        self.refractory_period_ns = refractory_period_ns
        logging.info(
            f"Initialized event skin simulator with sensor size: {data.shape}")
        logging.info(
            f"and contrast thresholds: C+ = {self.Cp}, C- = {self.Cm}")

        self.last_data = data.copy()
        self.ref_values = data.copy()
        self.last_event_timestamp = np.zeros(data.shape, dtype=np.ulonglong)
        self.current_time = time
        self.size = data.shape[0]

    def skinCallback(self, data, time):
        assert time >= 0

        # For each pixel, check if new events need to be generated since the last image sample
        tolerance = 1e-6
        events = []
        delta_t_ns = time - self.current_time

        assert delta_t_ns > 0
        # assert data.size() == self.size

        for taxel_ID in range(self.size):
            itdt = data[taxel_ID]
            it = self.last_data[taxel_ID]
            prev_cross = self.ref_values[taxel_ID]

            if abs(it - itdt) > tolerance:
                pol = +1.0 if itdt >= it else -1.0
                C = self.Cp if pol > 0 else self.Cm
                sigma_C = self.sigma_Cp if pol > 0 else self.sigma_Cm

                if sigma_C > 0:
                    C += np.random.normal(0, sigma_C)
                    minimum_contrast_threshold = 0.01
                    C = max(minimum_contrast_threshold, C)

                curr_cross = prev_cross
                all_crossings = False

                while not all_crossings:
                    curr_cross += pol * C

                    if (pol > 0 and curr_cross > it and curr_cross <= itdt) or \
                            (pol < 0 and curr_cross < it and curr_cross >= itdt):

                        edt = int(abs((curr_cross - it) *
                                      delta_t_ns / (itdt - it)))
                        t = self.current_time + edt

                        # check that taxel (taxel_ID) is not currently in a "refractory" state
                        # i.e. |t-that last_timestamp(taxel_ID)| >= refractory_period
                        last_stamp_at_xy = self.last_event_timestamp[taxel_ID]
                        assert t >= last_stamp_at_xy
                        dt = t - last_stamp_at_xy

                        if self.last_event_timestamp[taxel_ID] == 0 or dt >= self.refractory_period_ns:
                            events.append((taxel_ID, t, pol > 0))
                            self.last_event_timestamp[taxel_ID] = t
                        else:
                            logging.info(
                                f"Dropping event because time since last event ({dt} ns) < refractory period ({self.refractory_period_ns} ns).")
                        self.ref_values[taxel_ID] = curr_cross
                    else:
                        all_crossings = True
            # end tolerance

        # update simvars for next loop
        self.current_time = time
        self.last_data = data.copy()  # it is now the latest image

        # Sort the events by increasing timestamps, since this is what
        # most event processing algorithms expect

        events = np.array(events)
        if len(events):
            events[np.argsort(events[:, 2])]
        return events


def make_camera_event_frame(events, width=320, height=240):
    img = np.zeros((height, width))
    if len(events):
        coords = events[:, :2]
        img[coords[:, 1], coords[:, 0]] = 255
    return img


def visualize_camera(model, viewer_closed_event, show_raw_feed=True, show_ed_feed=True):
    # TODO find a elegant way to make sure the renderer starts up!
    time.sleep(1.0)
    renderer = mujoco.Renderer(model)
    esim = None

    camera_feed_window_name = 'Camera Feed'
    events_window_name = 'Events'

    while not viewer_closed_event.is_set():
        renderer.update_scene(data, camera=camera_name)
        pixels = renderer.render()
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)  # convert BRG to RGB

        if esim is None:
            esim = CameraEventSimulator(cv2.cvtColor(
                pixels, cv2.COLOR_RGB2GRAY), time.perf_counter_ns())
            if show_ed_feed:
                cv2.namedWindow(events_window_name)

                def on_thresh_slider(val):
                    val /= 100
                    esim.Cm = val
                    esim.Cp = val

                cv2.createTrackbar("Threshold", events_window_name, int(
                    esim.Cm * 100), 100, on_thresh_slider)
                cv2.setTrackbarMin("Threshold", events_window_name, 1)
            continue
        events = esim.imageCallback(cv2.cvtColor(
            pixels, cv2.COLOR_RGB2GRAY), time.perf_counter_ns())

        if show_raw_feed:
            cv2.imshow(camera_feed_window_name, pixels)
        if show_ed_feed:
            cv2.imshow(events_window_name, make_camera_event_frame(events))
        # Exit the loop if the ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # close OpenCV windows
    if show_raw_feed or show_ed_feed:
        cv2.destroyAllWindows()


def visualize_skin_patches(path_to_triangles, triangles_ini, DEBUG=False):
    print(triangles_ini)
    config_types, triangles = read_triangle_data(
        f"{path_to_triangles}/{triangles_ini}.ini")
    patch_ID = []
    dX = []
    dY = []
    dXv = []
    dYv = []
    scale = 3.0
    for tri, config_type in zip(triangles, config_types):
        # print(config_type)
        cx, cy, th, lr_mirror = tri[0][0], tri[0][1], tri[0][2], int(
            tri[0][4])
        patch_ID.append(tri[1])
        if config_type == "triangle_10pad":
            to_draw = triangle_10pad(
                cx=cx, cy=cy, th=th, lr_mirror=lr_mirror)
            # remove the thermal pads
            to_remove = [1, 5]  # always at the same position
            to_draw = list(to_draw)  # Convert tuple to list
            to_draw[0] = np.delete(to_draw[0], to_remove)
            to_draw[1] = np.delete(to_draw[1], to_remove)
            to_draw = tuple(to_draw)  # Convert back to tuple if needed

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
        # rearrange some triangles to make the fig more compact
        if (triangles_ini == "left_forearm_V2" or triangles_ini == "right_forearm_V2") and tri[1] in [16, 17, 19, 22, 24, 25, 28, 29]:
            for i in range(len(to_draw[1])):
                to_draw[1][i] += 40.0
            for i in range(len(to_draw[3])):
                to_draw[3][i] += 40.0
        if triangles_ini == "left_leg_upper" and tri[1] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 44, 45, 53, 54, 56, 57, 58, 64, 65, 66, 76, 78, 79]:
            for i in range(len(to_draw[1])):
                to_draw[1][i] += 83.0
            for i in range(len(to_draw[3])):
                to_draw[3][i] += 83.0
        if triangles_ini == "right_leg_upper" and tri[1] in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 44, 45, 51, 53, 54, 57, 58, 59, 64, 65, 66, 67, 76]:
            for i in range(len(to_draw[1])):
                to_draw[1][i] += 20.0
            for i in range(len(to_draw[3])):
                to_draw[3][i] += 20.0
        dX.extend(to_draw[0])
        dY.extend(to_draw[1])
        dXv.append(to_draw[2])
        dYv.append(to_draw[3])

    # let's set everything with repect to 0,0
    dX_min = np.min(dX)
    dY_min = np.min(dY)

    dX -= dX_min
    dY -= dY_min

    if not "hand" in triangles_ini:
        for i in range(len(dXv)):
            for j in range(len(dXv[i])):
                dXv[i][j] -= dX_min
                dYv[i][j] -= dY_min

    if DEBUG:
        # now we can draw the triangles
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dX, dY, marker='o')
        if not "hand" in triangles_ini:
            # draw the triangles
            for i in range(len(dXv)):
                for j in range(len(dXv[i])):
                    ax.plot([dXv[i][j-1], dXv[i][j]], [dYv[i][j-1],
                            dYv[i][j]], linewidth=0.2, color='black')
                ax.plot([dXv[i][-1], dXv[i][0]], [dYv[i][-1], dYv[i][0]],
                        linewidth=0.2, color='black')  # Close the triangle
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        fig.savefig(f"../figures/{triangles_ini}.pdf", bbox_inches='tight')
        plt.close(fig)

    # scale
    dX = dX*scale
    dY = dY*scale
    if not "hand" in triangles_ini:
        for i in range(len(dXv)):
            for j in range(len(dXv[i])):
                dXv[i][j] = dXv[i][j]*scale
                dYv[i][j] = dYv[i][j]*scale

    # flip axis and shift to positive values
    dY = -dY
    dYv_min = np.abs(np.min(dY))
    dY += dYv_min
    if not "hand" in triangles_ini:
        for i in range(len(dYv)):
            for j in range(len(dYv[i])):
                dYv[i][j] = -dYv[i][j] + dYv_min

    # find the width and height of the image
    width = np.max(dX)
    offset_x = width*0.2
    width += offset_x
    width = int(width+0.5)
    # offset_x = int((offset_x/2))

    height = np.max(dY)
    offset_y = height*0.2
    height += offset_y
    height = int(height+0.5)
    # offset_y = int((offset_y/2))

    # convert to int and apply offset to give somoe extra space at the corners
    dX = [int(x+0.5+offset_x/2) for x in dX]
    dY = [int(y+0.5+offset_y/2) for y in dY]
    if not "hand" in triangles_ini:
        for i in range(len(dYv)):
            for j in range(len(dYv[i])):
                dXv[i][j] = int(dXv[i][j]+0.5+offset_x/2)
                dYv[i][j] = int(dYv[i][j]+0.5+offset_y/2)

    if DEBUG:
        # now we can draw the triangles
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dX, dY, marker='o')
        if not "hand" in triangles_ini:
            # draw the triangles
            for i in range(len(dXv)):
                for j in range(len(dXv[i])):
                    ax.plot([dXv[i][j-1], dXv[i][j]], [dYv[i][j-1], dYv[i][j]], linewidth=0.2, color='black')
                ax.plot([dXv[i][-1], dXv[i][0]], [dYv[i][-1], dYv[i][0]], linewidth=0.2, color='black')  # Close the triangle
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        fig.savefig(
            f"../figures/{triangles_ini}_processed.pdf", bbox_inches='tight')
        plt.close(fig)

    # Create a blank image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw the triangles on the image
    for taxel_counter in range(len(dX)):
        cv2.circle(img, (dX[taxel_counter],  dY[taxel_counter]), 5, (0, 0, 255), 1)  # RGB color
    if not "hand" in triangles_ini:
        for i in range(len(dXv)):
            for j in range(len(dXv[i])):
                pt1 = (dXv[i][j-1], dYv[i][j-1])
                pt2 = (dXv[i][j], dYv[i][j])
                cv2.line(img, pt1, pt2, (255, 255, 255), 1)
            pt1 = (dXv[i][-1], dYv[i][-1])
            pt2 = (dXv[i][0], dYv[i][0])
            cv2.line(img, pt1, pt2, (255, 255, 255), 1)

    return (img, dX, dY)


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
                # for the hand files the header comes before the [SENSORS] line, for now I change the ini files locally. Find a better solution later.
                # also in torso.ini removes #leftlower
                start_found = True
                read_header = True
    return config_type, triangles


def make_skin_event_frame(img, events, locations):
    # events = taxel_ID, t, pol
    if len(events):
        active_taxel, nb_events = np.unique(events[:, 0], return_counts=True)
    for i, loc in enumerate(zip(locations[0], locations[1])):
        if len(events):
            if i in active_taxel:
                # event happened at this location
                if events[np.where(events[:, 0] == active_taxel[active_taxel == i])[0][0]][-1] > 0:
                    # TODO now we can scale the color according to the nb of spikes
                    cv2.circle(img, (int(loc[0]), int(loc[1])), 4, (0, 255, 0), -1)
                    print(f"On event at location {loc} at taxel {active_taxel[active_taxel == i]}")
                else:
                    cv2.circle(img, (int(loc[0]), int(loc[1])), 4, (255, 0, 0), -1)
                    print(f"Off event at location {loc} at taxel {active_taxel[active_taxel == i]}")
        else:
            # no event happened, return to blank
            cv2.circle(img, (int(loc[0]), int(loc[1])), 4, (0, 0, 0), -1)
    return img


def visualize_skin(viewer_closed_event, grouped_sensors, show_skin=True):
    time.sleep(1.0)
    # renderer = mujoco.Renderer(model)
    esim = []
    taxel_locs = {}
    imgs = {}
    initialized = False
    total_len = 0

    while not viewer_closed_event.is_set():
        if not initialized:
            # figure out which data to hand over
            
            for triangle_ini in TRIANGLE_FILES:
                # TODO make sure we hand over the right data here
                if "right_hand" in triangle_ini:
                    # TODO double check the order of the taxels!
                    taxel_data = []
                    for key in KEY_MAPPING["r_hand"]:
                        taxel_data.extend(grouped_sensors[key])
                elif "left_hand" in triangle_ini:
                    taxel_data = []
                    for key in KEY_MAPPING["l_hand"]:
                        taxel_data.extend(grouped_sensors[key])
                else:
                    taxel_data = grouped_sensors[KEY_MAPPING[triangle_ini]]
                esim.append(SkinEventSimulator(
                    np.array(taxel_data), time.perf_counter_ns()))
                if show_skin:
                    img, x, y = visualize_skin_patches(path_to_triangles=TRIANGLE_INI_PATH,
                                                       triangles_ini=triangle_ini, DEBUG=DEBUG)
                    total_len += len(x)
                    taxel_locs[triangle_ini] = [x, y]
                    imgs[triangle_ini] = img
                    cv2.namedWindow(triangle_ini, cv2.WINDOW_NORMAL)
                    cv2.imshow(triangle_ini, img)

                    def on_thresh_slider(val):
                        val /= 100
                        esim.Cm = val
                        esim.Cp = val

                    cv2.createTrackbar("Threshold", triangle_ini, int(
                        esim[-1].Cm * 100), 100, on_thresh_slider)
                    cv2.setTrackbarMin("Threshold", triangle_ini, 1)
            initialized = True
            cv2.waitKey(10)
            continue

        for triangle_ini, esim_single in zip(TRIANGLE_FILES, esim):
            # TODO make sure we hand over the right data here
            if "right_hand" in triangle_ini:
                # TODO double check the order of the taxels!
                taxel_data = []
                for key in KEY_MAPPING["r_hand"]:
                    taxel_data.extend(grouped_sensors[key])
            elif "left_hand" in triangle_ini:
                taxel_data = []
                for key in KEY_MAPPING["l_hand"]:
                    taxel_data.extend(grouped_sensors[key])
            else:
                taxel_data = grouped_sensors[KEY_MAPPING[triangle_ini]]

            events = esim_single.skinCallback(
                taxel_data, time.perf_counter_ns())

            if show_skin:                
                cv2.imshow(triangle_ini, make_skin_event_frame(
                    img = imgs[triangle_ini], events = events, locations = taxel_locs[triangle_ini]))
                # Exit the loop if the ESC key is pressed
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    # close OpenCV windows
    cv2.destroyAllWindows()


def control_loop(model, data, viewer_closed_event):
    time.sleep(1.0)
    # joints = ['r_index_proximal', 'r_index_distal', 'r_middle_proximal', 'r_middle_distal']
    joints = ['r_pinky']

    # Define parameters for the sine wave
    frequencies = [0.15, 0.15]
    start_time = time.time()

    min_max_pos = []
    for joint in joints:
        min_max_pos.append(model.actuator(joint).ctrlrange)

    # NOTE: testing only
    for i in range(len(min_max_pos)):
        min_max_pos[i][0] = min_max_pos[i][1]*0.9

    update_joint_positions(data, {"r_shoulder_roll": 0.565})
    update_joint_positions(data, {"r_hand_finger": 0.0})

    while not viewer_closed_event.is_set():
        elapsed_time = time.time() - start_time
        for (min_max, frequency, joint) in zip(min_max_pos, frequencies, joints):
            joint_position = min_max[0] + (min_max[1] - min_max[0]) * (
                0.5 * (1 + math.sin(2 * math.pi * frequency * elapsed_time)))
            # Update joint positions
            if DEBUG:
                print(joint, joint_position)
            update_joint_positions(data, {joint: joint_position})

        time.sleep(0.01)  # Update at 100 Hz


def update_joint_positions(data, joint_positions):
    """
    Updates the joint positions in the MuJoCo model.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data.
        joint_positions: A dictionary with joint names as keys and target positions as values.
    """
    for joint_name, target_position in joint_positions.items():
        joint = data.actuator(joint_name)
        joint.ctrl[0] = target_position


if __name__ == '__main__':
    viewer_closed_event = threading.Event()

    camera_name = 'head_cam'
    camera_window_name = 'Camera Feed'

    # Load the MuJoCo model and create a simulation
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    print("Model loaded")

    threading.Thread(target=control_loop, args=(
        model, data, viewer_closed_event)).start()

    # threading.Thread(target=visualize_camera, args=(
    #     model, viewer_closed_event, VISUALIZE_CAMERA_FEED, VISUALIZE_ED_CAMERA_FEED)).start()

    # prepare the mapping from skin to body parts
    """
    model.sensor_adr gives sensor ID
    model.name_sensoradr
    model.names gives names of everything included in the model
    """
    # Decode the bytes object to a string and split using the null character
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

    threading.Thread(target=visualize_skin, args=(
        viewer_closed_event, dynamic_grouped_sensors, VISUALIZE_SKIN)).start()

    # print("Threads started")

    viewer.launch(model, data, show_left_ui=False, show_right_ui=True)
    viewer_closed_event.set()
