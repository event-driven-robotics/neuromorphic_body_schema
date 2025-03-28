import logging
import math
import threading
import time

import cv2
import mujoco
import numpy as np
from draw_pads import fingertip3L, fingertip3R, palmL, palmR, triangle_10pad
from mujoco import viewer

from proprioception import generalized_sigmoid, linear, proprioception


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

VISUALIZE_CAMERA_FEED = True
VISUALIZE_ED_CAMERA_FEED = False
VISUALIZE_SKIN = False
VISUALIZE_PROPRIOCEPTION = False

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


def make_camera_event_frame(events, width=320, height=240):
    img = np.zeros((height, width))
    if len(events):
        coords = events[:, :2]
        img[coords[:, 1], coords[:, 0]] = 255
    return img


def visualize_camera(model, viewer_closed_event, show_raw_feed=True, show_ed_feed=True):
    time.sleep(1.0)  # TODO find a elegant way to make sure the renderer starts up!
    renderer = mujoco.Renderer(model)
    esim = None

    while not viewer_closed_event.is_set():
        renderer.update_scene(data, camera=camera_name)
        pixels = renderer.render()
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)  # convert BRG to RGB
        camera_feed_window_name = 'Camera Feed'
        events_window_name = 'Events'

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

##############################################################################################################

# TODO COmpare class with proprioception class

# class ProprioceptionEventSimulator:
#     def __init__(self, img, time, Cp=0.5, Cm=0.5, sigma_Cp=0.01, sigma_Cm=0.01, log_eps=1e-6, refractory_period_ns=100, use_log_image=True):
#         # these are all values useful for the event camera
#         # think about using the same structure for proprioception
#         self.Cp = Cp
#         self.Cm = Cm
#         self.sigma_Cp = sigma_Cp
#         self.sigma_Cm = sigma_Cm
#         self.log_eps = log_eps
#         self.use_log_image = use_log_image
#         self.refractory_period_ns = refractory_period_ns
#         logging.info(
#             f"Initialized event proprioception simulator with sensor size: {img.shape}")
#         logging.info(
#             f"and contrast thresholds: C+ = {self.Cp}, C- = {self.Cm}")

#         if self.use_log_image:
#             logging.info(
#                 f"Converting the image to log image with eps = {self.log_eps}.")
#             img = cv2.log(self.log_eps + img)
#         self.last_img = img.copy()
#         self.ref_values = img.copy()
#         self.last_event_timestamp = np.zeros(img.shape, dtype=np.ulonglong)
#         self.current_time = time
#         self.size = img.shape[:2]

    
    



def make_prop_event_frame(img, events, width=320, height=240):
    # TODO work out make_prop_event_frame
    img = np.zeros((height, width))
    if len(events):
        coords = events[:, :2]
        img[coords[:, 1], coords[:, 0]] = 255
    # take the old image, move content öne time step" to the left
    # add new content at the most right
    # return new img
    cv2.line(img, (x_start, y_start), (x_stop, y_stop), (255, 255, 255), 1)  # last entry is line thickness, x stop == x start (vertical line) 
    return img



def visualize_proprioception(model, viewer_closed_event, show_proprioception=False):
    time.sleep(1.0)
    renderer = mujoco.Renderer(model)
    esim = None
    
    events_window_name = 'Proprioception events'

    # here I define the joint(s) I want to visualize in proprioception
    joint_list = ['neck_yaw', 'neck_pitch', 'neck_roll']
    
    while not viewer_closed_event.is_set():

        # TODO replace with joint states I want to access
        for joint_name in joint_list:
            joint = data.qpos(joint_name)  # example for single joint
            joint_vel = data.qvel(joint_name) 
            joint_load = data.act(joint_name) 

        if esim is None:
            esim = proprioception(n_joints=len(joint_list), position_limits= [ model.actuator(joint_list).ctrlrange[0]   , model.actuator(joint_list).ctrlrange[1] ] , 
                                  velocity_limit=100.0, load_limit= 100.5, position_max_freq=np.array(100.), velocity_max_freq=np.array(100.), 
                                  load_max_freq=np.array(100.), limits_max_freq =np.array(100.))
            
            cv2.namedWindow(events_window_name)

            def on_thresh_slider(val):
            # adds slider to change online parameters of the class
                val /= 100
                esim.position_max_freq = val
               

            cv2.createTrackbar("Threshold", events_window_name, int(
                esim.Cm * 100), 100, on_thresh_slider)
            cv2.setTrackbarMin("Threshold", events_window_name, 1)
            continue
        
        events = esim.update(x = np.asarray(joint), v = joint_vel, load = joint_load , time_stamp = time.perf_counter_ns())   # imageCallback(   , time.perf_counter_ns())

        # how does imshow work in this case? what is the first argument and what is the second?
        # TODO work out make_prop_event_frame
        cv2.imshow(events_window_name, make_prop_event_frame(events))
        
        # Exit the loop if the ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # close OpenCV windows
    cv2.destroyAllWindows()

    
























##############################################################################################################




def control_loop(model, data, viewer_closed_event):
    time.sleep(1.0)

    joints = ['neck_yaw', 'neck_pitch', 'neck_roll']

    # Define parameters for the sine wave
    frequencies = [0.01, 0.02, 0.04]
    amplitudes = [1.0, 0.5, 0.25]
    start_time = time.time()

    min_max_pos = []
    for joint in joints:
        min_max_pos.append(model.actuator(joint).ctrlrange)

    while not viewer_closed_event.is_set():
        elapsed_time = time.time() - start_time
        for (min_max, amplitude, frequency, joint) in zip(min_max_pos, amplitudes, frequencies, joints):
            neck_position = min_max[0] + (min_max[1] - min_max[0]) * (
                0.5 * (1 + math.sin(2 * math.pi * frequency * elapsed_time)))
            # Update joint positions
            update_joint_positions(data, {joint: neck_position})

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
    # set model path
    model_path = '../neuromorphic_body_schema/models/icub_v2_full_body_contact_sensors.xml'  # full iCub
    # model_path = './models/icub_v2_full_body_contact_sensors.xml'  # full iCub
    # model_path = './models/icub_v2_full_body_contact_sensors_automated.xml'  # full iCub

    # model_path = './models/icub_v2_full_body_contact_sensors.xml'  # full iCub
    # model_path = 'neuromorphic_body_schema/models/icub_v2_right_hand_mk2_contact_sensor.xml'  # right hand only
    # DEBUG #
    # model_path = './models/icub_v2_full_body_contact_sensors.xml'  # full iCub
    # model_path = './models/icub_v2_right_hand_mk2_contact_sensor.xml'  # right hand only

    camera_name = 'head_cam'
    camera_window_name = 'Camera Feed'

    # Load the MuJoCo model and create a simulation
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    print("Model loaded")

    
    threading.Thread(target=control_loop, args=(
        model, data, viewer_closed_event)).start()

    #threading.Thread(target=visualize_camera, args=(
    #    model, viewer_closed_event, VISUALIZE_CAMERA_FEED, VISUALIZE_ED_CAMERA_FEED)).start()
    
    # threading.Thread(target=visualize_proprioception, args=(
    #     model, viewer_closed_event, VISUALIZE_PROPRIOCEPTION)).start()



    # threading.Thread(target=visualize_skin, args=(
    #     model, viewer_closed_event, VISUALIZE_SKIN)).start()

    # print("Threads started")

    viewer.launch(model, data, show_left_ui=False, show_right_ui=True)
    
    viewer_closed_event.set()
