"""
ed_cam.py

Author: Simon F. Muller-Cleve, Massimiliano Iocano
Affiliation: Istituto Italiano di Tecnologia (IIT)
Department: Event-Driven Perception for Robotics (EDPR)
Date: 29.04.2025

Description:
This module provides functionality for simulating an event-based camera and integrating it with the iCub robot's 
visual system. It includes classes and functions for generating events based on pixel intensity changes, 
visualizing event data, and managing the camera feed.

Classes:
- CameraEventSimulator: Simulates an event-based camera by generating events based on pixel intensity changes.
- ICubEyes: Represents the iCub robot's eyes, integrating a renderer and an event-based camera simulator.

Functions:
- make_camera_event_frame: Generates a visual representation of event-based camera data as a 2D image.

"""


import logging

import cv2
import mujoco
import numpy as np

# set the color for the events
red = (0, 0, 255)   # positive events
blue = (255, 0, 0)  # negative events


class CameraEventSimulator:
    """
    Simulates an event-based camera by generating events based on changes in pixel intensity.

    Attributes:
        Cp (float): Positive contrast threshold.
        Cm (float): Negative contrast threshold.
        sigma_Cp (float): Standard deviation for noise in the positive contrast threshold.
        sigma_Cm (float): Standard deviation for noise in the negative contrast threshold.
        log_eps (float): Small constant added to avoid log(0) when using logarithmic images.
        refractory_period_ns (int): Minimum time (in nanoseconds) between consecutive events for the same pixel.
        use_log_image (bool): Whether to use logarithmic transformation of the input image.
        last_img (np.array): The last processed image.
        ref_values (np.array): Reference values for contrast threshold crossings.
        last_event_timestamp (np.array): Timestamps of the last event for each pixel.
        current_time (int): Current simulation time.
        size (tuple): Size of the input image (height, width).
    """

    def __init__(self, img, time, Cp=0.5, Cm=0.5, sigma_Cp=0.01, sigma_Cm=0.01, log_eps=1e-6, refractory_period_ns=100, use_log_image=True):
        """
        Initializes the CameraEventSimulator.

        Args:
            img (np.array): Initial image to initialize the simulator.
            time (int): Initial simulation time in nanoseconds.
            Cp (float): Positive contrast threshold. Default is 0.5.
            Cm (float): Negative contrast threshold. Default is 0.5.
            sigma_Cp (float): Noise standard deviation for Cp. Default is 0.01.
            sigma_Cm (float): Noise standard deviation for Cm. Default is 0.01.
            log_eps (float): Small constant for logarithmic transformation. Default is 1e-6.
            refractory_period_ns (int): Refractory period in nanoseconds. Default is 100.
            use_log_image (bool): Whether to use logarithmic transformation. Default is True.
        """

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
        """
        Processes a new image and generates events based on pixel intensity changes.

        Args:
            img (np.array): The new image to process.
            time (int): The current simulation time in nanoseconds.

        Returns:
            np.array: A list of events, where each event is a tuple (x, y, t, polarity).
        """
        assert time >= 0

        if self.use_log_image:
            img = cv2.log(self.log_eps + img)

        tolerance = 1e-6
        delta_t_ns = time - self.current_time
        assert delta_t_ns > 0

        # Compute difference from previous image
        itdt = img
        it = self.last_img
        prev_cross = self.ref_values

        delta_img = itdt - it

        # Avoid division by zero
        valid_mask = np.abs(delta_img) > tolerance

        # Prepare output
        events = []

        # Only process pixels where there is a change
        yx = np.argwhere(valid_mask)
        if yx.size == 0:
            self.current_time = time
            self.last_img = img.copy()
            return np.empty((0, 4), dtype=object)

        y_idx, x_idx = yx[:, 0], yx[:, 1]
        itdt_v = itdt[y_idx, x_idx]
        it_v = it[y_idx, x_idx]
        prev_cross_v = prev_cross[y_idx, x_idx]
        delta_v = itdt_v - it_v

        # Determine polarity for each pixel
        pol_v = np.where(delta_v > 0, 1.0, -1.0)
        C_v = np.where(pol_v > 0, self.Cp, self.Cm)
        sigma_C_v = np.where(pol_v > 0, self.sigma_Cp, self.sigma_Cm)

        # For each pixel, compute all threshold crossings
        # Number of crossings = floor((itdt_v - prev_cross_v) / (pol_v * C_v))
        # But need to handle noise per crossing!
        for idx in range(len(y_idx)):
            y, x = y_idx[idx], x_idx[idx]
            it0 = it[y, x]
            it1 = itdt[y, x]
            ref = prev_cross[y, x]
            pol = 1.0 if it1 >= it0 else -1.0
            C = self.Cp if pol > 0 else self.Cm
            sigma_C = self.sigma_Cp if pol > 0 else self.sigma_Cm

            curr_cross = ref
            all_crossings = False
            while not all_crossings:
                # Add noise to threshold
                C_eff = C + (np.random.normal(0, sigma_C) if sigma_C > 0 else 0)
                C_eff = max(0.01, C_eff)
                curr_cross += pol * C_eff

                # Check if crossing occurred in this interval
                if (pol > 0 and curr_cross > it0 and curr_cross <= it1) or \
                   (pol < 0 and curr_cross < it0 and curr_cross >= it1):

                    # Interpolate event time
                    edt = int(abs((curr_cross - it0) * delta_t_ns / (it1 - it0)))
                    t_evt = self.current_time + edt

                    # Refractory check
                    last_stamp = self.last_event_timestamp[y, x]
                    dt = t_evt - last_stamp
                    if last_stamp == 0 or dt >= self.refractory_period_ns:
                        events.append((x, y, t_evt, pol > 0))
                        self.last_event_timestamp[y, x] = t_evt
                        self.ref_values[y, x] = curr_cross
                    else:
                        # Don't update ref_values if event is dropped
                        pass
                else:
                    all_crossings = True

        # Update state for next call
        self.current_time = time
        self.last_img = img.copy()

        # Sort events by timestamp
        if len(events):
            events = np.array(events)
            events = events[np.argsort(events[:, 2])]
        else:
            events = np.empty((0, 4), dtype=object)
        return events


def make_camera_event_frame(events, width=320, height=240):
    """
    Generates a visual representation of event-based camera data as a 2D image.

    Args:
        events (np.array): A numpy array of shape (N, 4), where each row represents an event with:
                           - x (int): X-coordinate of the event.
                           - y (int): Y-coordinate of the event.
                           - t (int): Timestamp of the event (not used in visualization).
                           - polarity (bool): Polarity of the event (not used in visualization).
        width (int): Width of the output image. Default is 320.
        height (int): Height of the output image. Default is 240.

    Returns:
        np.array: A 2D numpy array of shape (height, width) representing the event frame, 
                  where pixel values are set to 255 for event locations and 0 elsewhere.
    """

    img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(events):
        coords = events[:, :2].astype(int)
        # seperate positive and negative events
        pos_events = events[events[:, 3] == 1]
        neg_events = events[events[:, 3] == 0]
        if len(pos_events):
            coords = pos_events[:, :2].astype(int)
            img[coords[:, 1], coords[:, 0]] = red
        if len(neg_events):
            coords = neg_events[:, :2].astype(int)
            img[coords[:, 1], coords[:, 0]] = blue
    return img


class ICubEyes:
    """
    Represents the iCub robot's eyes, integrating a renderer and an event-based camera simulator.

    Attributes:
        camera_name (str): Name of the camera to use in the MuJoCo model.
        model (mujoco.MjModel): The MuJoCo model of the robot.
        data (mujoco.MjData): The MuJoCo data object for simulation.
        show_raw_feed (bool): Whether to display the raw camera feed.
        show_ed_feed (bool): Whether to display the event-based camera feed.
        DEBUG (bool): Whether to enable debug logging.
        renderer (mujoco.Renderer): Renderer for the MuJoCo simulation.
        camera_feed_window_name (str): Name of the window displaying the raw camera feed.
        events_window_name (str): Name of the window displaying the event-based camera feed.
        esim (CameraEventSimulator): Event-based camera simulator.
    """

    def __init__(self, time, model, data, camera_name, show_raw_feed=True, show_ed_feed=True, DEBUG=False):
        """
        Initializes the ICubEyes class with a renderer and an event-based camera simulator.

        Args:
            time (int): Initial simulation time in nanoseconds.
            model (mujoco.MjModel): The MuJoCo model of the robot.
            data (mujoco.MjData): The MuJoCo data object for simulation.
            camera_name (str): Name of the camera to use in the MuJoCo model.
            show_raw_feed (bool): Whether to display the raw camera feed. Default is True.
            show_ed_feed (bool): Whether to display the event-based camera feed. Default is True.
            DEBUG (bool): Whether to enable debug logging. Default is False.
        """

        self.camera_name = camera_name
        self.model = model
        self.data = data
        self.show_raw_feed = show_raw_feed
        self.show_ed_feed = show_ed_feed
        self.DEBUG = DEBUG

        self.renderer = mujoco.Renderer(model)

        camera_name_split = camera_name.split('_')
        camera_name_spec = f"{camera_name_split[0].capitalize()} {camera_name_split[1].capitalize()}"
        self.camera_feed_window_name = f'{camera_name_spec} Camera Feed'
        self.events_window_name = f'{camera_name_spec} Event Feed'

        self.renderer.update_scene(self.data, camera=self.camera_name)
        pixels = self.renderer.render()
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)  # convert BGR to RGB

        self.esim = CameraEventSimulator(cv2.cvtColor(
            pixels, cv2.COLOR_RGB2GRAY), time)
        if show_ed_feed:
            cv2.namedWindow(self.events_window_name)

            def on_thresh_slider(val):
                val /= 100
                self.esim.Cm = val
                self.esim.Cp = val

            cv2.createTrackbar("Threshold", self.events_window_name, int(
                self.esim.Cm * 100), 100, on_thresh_slider)
            cv2.setTrackbarMin("Threshold", self.events_window_name, 1)

    def update_camera(self, time):
        """
        Updates the camera feed and processes events using the event-based camera simulator.

        Args:
            time (int): Current simulation time in nanoseconds.

        Returns:
            np.array: A list of events generated by the event-based camera simulator.
                      Each event is a tuple (x, y, t, polarity):
                      - x, y: Pixel coordinates.
                      - t: Timestamp of the event.
                      - polarity: True for positive events, False for negative events.
        """

        self.renderer.update_scene(self.data, camera=self.camera_name)
        pixels = self.renderer.render()
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)  # convert BRG to RGB
        events = self.esim.imageCallback(cv2.cvtColor(
            pixels, cv2.COLOR_RGB2GRAY), time)

        if self.DEBUG:
            if len(events):
                logging.info(f"Generated {len(events)} camera events.")

        if self.show_raw_feed:
            cv2.imshow(self.camera_feed_window_name, pixels)
        if self.show_ed_feed:
            cv2.imshow(self.events_window_name,
                       make_camera_event_frame(events))
        if self.show_ed_feed or self.show_raw_feed:
            cv2.waitKey(1)
        return events
