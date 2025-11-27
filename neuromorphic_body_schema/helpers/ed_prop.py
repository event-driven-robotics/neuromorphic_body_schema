"""
ed_prop.py

Author: Simon F. Muller-Cleve, Miriam Barborini
Affiliation: Istituto Italiano di Tecnologia (IIT)
Department: Event-Driven Perception for Robotics (EDPR)
Date: 29.04.2025

Description:
This module provides functionality for simulating event-based proprioception in the iCub robot. 
It includes classes and functions for generating spiking events based on joint positions, velocities, 
and loads, as well as visualizing proprioception data.

Classes:
- ProprioceptionEventSimulator: Simulates spiking representations of proprioceptive inputs.
- ICubProprioception: Represents the iCub robot's proprioception system, integrating joint data with event-based spiking representations.

Functions:
- generalized_sigmoid: Computes a generalized sigmoid function for scaling input values.
- linear: Computes a linear function value for the given input.
- make_proprioception_event_frame: Updates a visual representation of proprioception events over time.

"""


import logging

import cv2
import numpy as np
from helpers import HEIGHT, MARGIN, TICK_HEIGHT, TIME_WINDOW, WIDTH


def generalized_sigmoid(x: np.array,  x_min: np.array, x_max: np.array, y_min: np.array, y_max: np.array, B=np.array) -> np.array:
    """
    Computes a generalized sigmoid function value for the given input.

    This function normalizes the input `x` to a range defined by `x_min` and `x_max`, applies a sigmoid 
    transformation, and scales the output to the range defined by `y_min` and `y_max`.

    Args:
        x (np.array): Input value(s) where the sigmoid function is evaluated.
        x_min (np.array): Minimum value(s) for input normalization.
        x_max (np.array): Maximum value(s) for input normalization.
        y_min (np.array): Minimum value(s) for output scaling.
        y_max (np.array): Maximum value(s) for output scaling.
        B (np.array): Steepness parameter(s) of the sigmoid function.

    Returns:
        np.array: The computed generalized sigmoid function value(s), scaled to the range [y_min, y_max].
    """

    x_norm = (x - x_min) / (x_max - x_min)

    # Apply sigmoid function centered at 0.5
    y_norm = 1 / (1 + np.exp(-B * (x_norm - 0.5)))

    y_min_local = 1 / (1 + np.exp(-B * (0 - 0.5)))

    y_max_local = 1 / (1 + np.exp(-B * (1 - 0.5)))

    # Stretch y_norm to ensure it spans from 0 to 1
    y_norm = (y_norm - y_min_local) / (y_max_local - y_min_local)

    # Scale y to [y_min, y_max]
    y_scaled = y_min + (y_max - y_min) * y_norm

    y_scaled = np.where(np.asarray(y_scaled) < 0.0, 0, y_scaled)

    return y_scaled


def linear(x, m=1, q=0):
    """
    Computes a linear function value for the given input.

    Args:
        x (float or np.array): Input value(s) where the linear function is evaluated.
        m (float): Slope of the linear function. Default is 1.
        q (float): Offset (y-intercept) of the linear function. Default is 0.

    Returns:
        float or np.array: The computed linear function value(s).
    """

    return m*x + q


class ProprioceptionEventSimulator():
    """
    Simulates spiking representations of proprioceptive inputs, such as joint positions, velocities, and loads.
    Inputs can come from fictitious data, MuJoCo simulations, or iCub's YARP values.

    Attributes:
        position_limit_min (np.array): Minimum joint position limits.
        position_limit_max (np.array): Maximum joint position limits.
        velocity_limit (float): Maximum joint velocity limit.
        load_limit (float): Maximum joint load limit.
        position_max_freq (np.array): Maximum firing frequency for position neurons.
        velocity_max_freq (np.array): Maximum firing frequency for velocity neurons.
        load_max_freq (np.array): Maximum firing frequency for load neurons.
        limits_max_freq (np.array): Maximum firing frequency for limit neurons.
        time_of_last_spike (np.array): Timestamps of the last spike for each neuron.
        nb_neurons (int): Number of neurons (e.g., for position, velocity, load, and limits).
        DEBUG (bool): Whether to enable debug logging.
    """

    def __init__(self, position_limits, velocity_limit, load_limit,  position_max_freq=np.array(1000.), velocity_max_freq=np.array(1000.), load_max_freq=np.array(1000.), limits_max_freq=np.array(1000.), DEBUG=False):
        """
        Initializes the ProprioceptionEventSimulator with joint limits, firing frequencies, and debug options.

        Args:
            position_limits (list): A list containing minimum and maximum joint position limits as [[min_limits], [max_limits]].
            velocity_limit (float): Maximum joint velocity limit.
            load_limit (float): Maximum joint load limit.
            position_max_freq (np.array): Maximum firing frequency for position neurons. Default is 1000 Hz.
            velocity_max_freq (np.array): Maximum firing frequency for velocity neurons. Default is 1000 Hz.
            load_max_freq (np.array): Maximum firing frequency for load neurons. Default is 1000 Hz.
            limits_max_freq (np.array): Maximum firing frequency for limit neurons. Default is 1000 Hz.
            DEBUG (bool): Whether to enable debug logging. Default is False.
        """

        # self.pos = np.zeros(2)  # position
        # self.v = np.zeros(2)  # velocity
        # self.l = np.zeros(2)  # load
        # self.lim = np.zeros(2)  # closeness to limits
        self.nb_neurons = 8  # 2 for pos, 2 for vel, 2 for load, 2 for limits

        # measuring unity is in Hz. 1000Hz means we expect 1000 events per second (circa 1 per ms).
        self.position_max_freq = position_max_freq
        self.velocity_max_freq = velocity_max_freq
        self.load_max_freq = load_max_freq
        self.limits_max_freq = limits_max_freq

        # parameters of the system
        # limits of the joint positions
        self.position_limit_min = position_limits[0]
        # limits of the joint positions
        self.position_limit_max = position_limits[1]
        self.velocity_limit = velocity_limit
        self.load_limit = load_limit

        self.time_of_last_spike = np.zeros((8))

        self.DEBUG = DEBUG

    def position(self, x: float, B):
        """
        Converts joint position values into spiking frequencies for agonistic-antagonistic neurons.

        Args:
            x (float): Joint position value.
            B (float): Sigmoid steepness parameter.

        Returns:
            tuple: Firing frequencies for agonistic and antagonistic neurons.
        """

        if (x - self.position_limit_min).any() < 0:
            print("WARNING: joint value outside scope")
        if (-x + self.position_limit_max).any() < 0:
            print("WARNING: joint value outside scope")

        y1 = 0.1 * generalized_sigmoid(x=x,   x_min=self.position_limit_min,
                                       x_max=self.position_limit_max, y_min=0.0, y_max=self.position_max_freq, B=B)
        y2 = 0.1*(self.position_max_freq - 10 * y1)

        # if self.DEBUG:
        #     x_debug = np.linspace(self.position_limit_min, self.position_limit_max, 1000)
        #     y_debug = generalized_sigmoid(x=x_debug,   x_min = self.position_limit_min, x_max=self.position_limit_max, y_min= 0. , y_max=self.position_max_freq, B = B )
        #     plt.figure()
        #     plt.plot(x_debug, y_debug)
        #     plt.plot(x_debug, self.position_max_freq - y_debug)
        #     plt.xlabel("Joint position [deg]")
        #     plt.ylabel("Firing rate [Hz]")
        #     plt.title("Position")
        #     plt.show()

        return (max(y1, 0.0), max(y2, 0.0))

    def velocity(self, v):
        """
        Converts joint velocity values into spiking frequencies for agonistic-antagonistic neurons.

        Args:
            v (float): Joint velocity value.

        Returns:
            tuple: Firing frequencies for agonistic and antagonistic neurons.
        """

        if np.min(v) < - self.velocity_limit:
            v = np.where(v < (-self.velocity_limit), - self.velocity_limit, v)
            print("WARNING: joint velocity value reached")
        if np.max(v) > self.velocity_limit:
            v = np.where(v > self.velocity_limit,  self.velocity_limit, v)
            print("WARNING: joint velocity value reached")

        module = linear(np.abs(v), m=self.velocity_max_freq /
                        self.velocity_limit)
        direction_pos = np.where(v > 0, 1.0, 0.0)
        direction_neg = np.where(v < 0, 1.0, 0.0)

        y1 = module * direction_pos  # element-wise multiplication
        y2 = module * direction_neg  # element-wise multiplication

        return (y1, y2)

    def load(self, load):
        """
        Converts joint load values into spiking frequencies for agonistic-antagonistic neurons.

        Args:
            load (float): Joint load value.

        Returns:
            tuple: Firing frequencies for agonistic and antagonistic neurons.
        """

        if np.min(load) < - self.load_limit:
            print("WARNING: joint load value reached")
            load = np.where(load < (-self.load_limit), - self.load_limit, load)
        if np.max(load) > self.load_limit:
            print("WARNING: joint load value reached")
            load = np.where(load > self.load_limit,  self.load_limit, load)

        module = linear(np.abs(load), m=self.load_max_freq / self.load_limit)
        direction_pos = np.where(load > 0, 1.0, 0.0)
        direction_neg = np.where(load < 0, 1.0, 0.0)

        y1 = module * direction_pos  # element-wise multiplication
        y2 = module * direction_neg  # element-wise multiplication

        return (y1, y2)

    def limit(self, limit, B, delta_x):
        """
        Converts joint position proximity to limits into spiking frequencies for neurons.

        Args:
            limit (float): Joint position value.
            B (float): Sigmoid steepness parameter.
            delta_x (float): Range around the joint limits to activate neurons.

        Returns:
            tuple: Firing frequencies for neurons representing proximity to lower and upper limits.
        """

        # TODO we want this neurons only to be active when CLOSE to the limits! Because each joint can have different ranges this should be 10% of max range!
        if (limit - self.position_limit_min).any() < 0:
            print("WARNING: joint value outside scope")
        if (-limit + self.position_limit_max).any() < 0:
            print("WARNING: joint value outside scope")

        # parameters of the limit function

        # below here the two B must be the first positive and the second negative
        y1 = generalized_sigmoid(x=limit,   x_min=self.position_limit_min + 2*delta_x,
                                 x_max=self.position_limit_min, y_min=0.0, y_max=self.limits_max_freq, B=-B)
        y2 = generalized_sigmoid(x=limit,   x_min=self.position_limit_max-2*delta_x,
                                 x_max=self.position_limit_max, y_min=0.0, y_max=self.position_max_freq, B=B)

        if self.DEBUG:
            x_debug = np.linspace(self.position_limit_min,
                                  self.position_limit_max, 1000)
            y_debug1 = generalized_sigmoid(x=x_debug,   x_min=self.position_limit_min + 2*delta_x,
                                           x_max=self.position_limit_min, y_min=0., y_max=self.limits_max_freq, B=-B)
            y_debug2 = generalized_sigmoid(x=x_debug,   x_min=self.position_limit_max-2*delta_x,
                                           x_max=self.position_limit_max, y_min=0., y_max=self.position_max_freq, B=B)

            # plt.figure()
            # plt.plot(x_debug, y_debug1, label="descending")
            # plt.plot(x_debug, y_debug2, label="ascending")
            # plt.xlabel("Joint position [deg]")
            # plt.ylabel("Firing rate [Hz]")
            # plt.title("Limit")
            # plt.legend()
            # plt.show()

        return (y1, y2)

    def proprioceptionCallback(self, x, v, load, time_stamp, B_pos=5.0, B_lim=20.0, delta_x=1.0):
        """
        Processes joint position, velocity, and load values to generate spiking events.

        Args:
            x (float): Joint position value.
            v (float): Joint velocity value.
            load (float): Joint load value.
            time_stamp (int): Current simulation timestamp.
            B_pos (float): Sigmoid steepness parameter for position. Default is 5.0.
            B_lim (float): Sigmoid steepness parameter for limits. Default is 20.0.
            delta_x (float): Range around the joint limits to activate neurons. Default is 1.0.

        Returns:
            list: A list of events, where each event is a tuple (neuron_id, timestamp, polarity).
        """

        # update the joint positions values
        pos = self.position(x, B=B_pos)
        v = self.velocity(v)
        l = self.load(load)
        lim = self.limit(limit=x, B=B_lim, delta_x=delta_x)

        def invert_tuple(frequ):
            # Check if the current element is a tuple
            if isinstance(frequ, tuple):
                # Apply inversion to each element recursively
                return tuple(invert_tuple(x) for x in frequ)
            else:
                # delta_t = np.where(frequ == 0.0, np.inf, 1/frequ)
                if frequ == 0.0:
                    delta_t = np.inf
                else:
                    delta_t = 1/frequ
                return delta_t

        # Develop the events according to the AER protocol

        # estimate of the delta time before we should have another spike
        time_to_spike = np.zeros(self.nb_neurons)
        time_to_spike[0] = invert_tuple(pos[0])
        # TODO when we have a negative joint value this returns negative times and the simulation crashes!
        time_to_spike[1] = invert_tuple(pos[1])
        time_to_spike[2] = invert_tuple(v[0])
        time_to_spike[3] = invert_tuple(v[1])
        time_to_spike[4] = invert_tuple(l[0])
        time_to_spike[5] = invert_tuple(l[1])
        time_to_spike[6] = invert_tuple(lim[0])
        time_to_spike[7] = invert_tuple(lim[1])

        # if not working look for np.hstack or np.vstack
        # time_to_spike = place_holder

        events = []
        # loop over neuron types
        for i, single_time_to_spike in enumerate(time_to_spike):
            while self.time_of_last_spike[i] + single_time_to_spike < time_stamp:
                self.time_of_last_spike[i] = self.time_of_last_spike[
                    i] + single_time_to_spike
                events.append(
                    (i, self.time_of_last_spike[i], 0))

        if len(events) > 1:
            events = np.array(events)
            events = events[np.argsort(events[:, 2])]

        return events


def make_proprioception_event_frame(img, time, events):
    """
    Updates a visual representation of proprioception events over time.

    This function takes an existing image, shifts its content to the left to simulate a time window, 
    and overlays new proprioception events as colored lines based on their neuron index and timestamp.

    Args:
        img (np.array): A 2D or 3D numpy array representing the current event frame.
        time (int): Current simulation timestamp in nanoseconds.
        events (np.array): A numpy array of shape (N, 3), where each row represents an event with:
                           - neuron_id (int): Index of the neuron that generated the event.
                           - timestamp (int): Timestamp of the event in nanoseconds.
                           - polarity (int): Polarity of the event (not used in visualization).

    Returns:
        np.array: An updated numpy array representing the event frame, with new events added.
    """

    # here we take the old image, move it's content to the left and add the new events
    # remove old events
    if len(events) and events[-1, 1] < time - TIME_WINDOW:
        # TODO improve the if case to only be active if needed (last event < time - TIME_WINDOW)
        # and events is just the output of update
        while len(events) and events[0, 1] < time - TIME_WINDOW:
            events = np.delete(events, 0, axis=0)
        # events = list(events)

    # TODO do I need esim?
    # tick_height = 20
    # margin = 5
    # height = 8 * (tick_height + margin)
    # current_img = np.zeros((height, width, 3))

    t_left = time-TIME_WINDOW
    scale = img.shape[1]/TIME_WINDOW
    # TODO find out how many pixel along x is 'one event frame'
    # we override the first 'event column' with empty entries and apply np.roll to move them to the end
    img = np.roll(img, -1, axis=1)
    img[:, -1, :] = np.zeros((img.shape[0], 3))  # , dtype=np.float64
    if len(events):
        for single_event in events:
            y_val = int(single_event[0]*(TICK_HEIGHT + MARGIN/2))
            x_val = int((single_event[1]-t_left)*scale)
            neuron_idx = single_event[0]

            color_index = int((neuron_idx * 255 / 8))
            color = cv2.applyColorMap(
                np.array([[color_index]], dtype=np.uint8), cv2.COLORMAP_HSV)[0][0]
            color = color/255

            cv2.line(img, (x_val, y_val), (x_val, y_val+TICK_HEIGHT), color, 1)

    return img


class ICubProprioception:
    """
    Represents the iCub robot's proprioception system, integrating joint position, velocity, and load data 
    with event-based spiking representations.

    Attributes:
        esim (list): A list of ProprioceptionEventSimulator instances for each joint.
        joint_dict (dict): A dictionary containing joint-specific parameters (e.g., max frequencies).
        imgs (list): A list of images for visualizing proprioception events for each joint.
        show_proprioception (bool): Whether to display proprioception event visualizations.
        DEBUG (bool): Whether to enable debug logging.
    """

    def __init__(self, model, joint_dict, show_proprioception=False, DEBUG=False):
        """
        Initializes the ICubProprioception class with joint-specific simulators and visualization options.

        Args:
            model (mujoco.MjModel): The MuJoCo model of the robot.
            joint_dict (dict): A dictionary containing joint-specific parameters, such as:
                               - position_max_freq (float): Maximum firing frequency for position neurons.
                               - velocity_max_freq (float): Maximum firing frequency for velocity neurons.
                               - load_max_freq (float): Maximum firing frequency for load neurons.
                               - limits_max_freq (float): Maximum firing frequency for limit neurons.
            show_proprioception (bool): Whether to display proprioception event visualizations. Default is False.
            DEBUG (bool): Whether to enable debug logging. Default is False.
        """

        # nb_joints = len(joint_dict.keys())
        # TODO take arguments for the event encoding!
        # max_max_freq = 1e-4
        self.esim = []
        self.joint_dict = joint_dict
        self.imgs = []
        self.show_proprioception = show_proprioception
        self.DEBUG = DEBUG

        # TODO replace with joint states I want to access
        for joint_name in list(joint_dict.keys()):
            position_max_freq = joint_dict[joint_name]['position_max_freq']
            velocity_max_freq = joint_dict[joint_name]['velocity_max_freq']
            load_max_freq = joint_dict[joint_name]['load_max_freq']
            limits_max_freq = joint_dict[joint_name]['limits_max_freq']
            self.imgs.append(np.zeros((HEIGHT, WIDTH, 3)))  # , dtype=np.uint8
            self.esim.append(ProprioceptionEventSimulator(position_limits=model.actuator(joint_name).ctrlrange,
                                                          velocity_limit=20.0, load_limit=10000000.0, position_max_freq=position_max_freq, velocity_max_freq=velocity_max_freq,
                                                          load_max_freq=load_max_freq, limits_max_freq=limits_max_freq, DEBUG=DEBUG))

            # The values used for the next object are absolutely arbitrary, they will be changed in a later stage
            # also based on the platform the code is going to be used on
            # max freq is 0.1 cause time is in ns

            if show_proprioception:

                # events_window_name = events_window_names[i]
                cv2.namedWindow(joint_name)

                # cv2.createTrackbar("Threshold", events_window_name, int(
                #     esim.position_max_freq * 100), 100, on_thresh_slider)
                # cv2.setTrackbarMin("Threshold", events_window_name, 1)

    def update_proprioception(self, time, data):
        """
        Updates the proprioception system by processing joint position, velocity, and load data to generate events.

        Args:
            time (int): Current simulation timestamp in nanoseconds.
            data (mujoco.MjData): The MuJoCo data object containing joint states.

        Returns:
            list: A list of events for all joints, where each event is a tuple (neuron_id, timestamp, polarity).
        """

        all_events = []
        for i, esim_single, joint_name in zip(range(len(self.esim)), self.esim, list(self.joint_dict.keys())):
            joint_pos = data.joint(joint_name).qpos  # example for single joint
            joint_vel = data.joint(joint_name).qvel
            # TODO access load here, not acceleration!
            joint_load = data.joint(joint_name).qacc
            events = esim_single.proprioceptionCallback(
                x=joint_pos, v=joint_vel, load=joint_load, time_stamp=time)
            all_events.append(events)

            if self.DEBUG:
                if len(events):
                    logging.info(
                        f"Generated {len(events)} proprioception events.")

            if self.show_proprioception:
                events_array = np.array(events)
                # TODO check why pos enc is similar for both neurons!
                # TODO some events seem to pop up out of nowhere!
                self.imgs[i] = make_proprioception_event_frame(
                    self.imgs[i], time, events_array)
                cv2.imshow(joint_name, self.imgs[i])
                cv2.waitKey(1)

        return all_events
