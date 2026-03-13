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
from pathlib import Path

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
        """Create an event-camera simulator from an initial grayscale frame.

        Args:
            img (np.ndarray): Initial grayscale frame (typically uint8) used to
                seed internal references.
            time (int): Initial timestamp in nanoseconds.
            Cp (float): Positive contrast threshold for ON events.
            Cm (float): Negative contrast threshold for OFF events.
            sigma_Cp (float): Std-dev of Gaussian noise added to ``Cp``.
            sigma_Cm (float): Std-dev of Gaussian noise added to ``Cm``.
            log_eps (float): Small value added before log transform to avoid
                log(0) when ``use_log_image`` is enabled.
            refractory_period_ns (int): Minimum interval between two accepted
                events at the same pixel.
            use_log_image (bool): If True, apply logarithmic encoding before
                event generation.
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
        """Process a new frame and emit event tuples for threshold crossings.

        Args:
            img (np.ndarray): New grayscale frame sampled at ``time``.
            time (int): Current timestamp in nanoseconds. Must be strictly
                greater than the previous callback timestamp.

        Returns:
            np.ndarray: Event array of shape ``(N, 4)`` with rows
                ``(x, y, t_ns, polarity)`` where polarity is ``True`` for ON
                events and ``False`` for OFF events. Returns an empty object
                array when no events are generated.

        Notes:
            - Events are sorted by timestamp before returning.
            - Refractory filtering is applied per pixel.
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
                C_eff = C + (np.random.normal(0, sigma_C)
                             if sigma_C > 0 else 0)
                C_eff = max(0.01, C_eff)
                curr_cross += pol * C_eff

                # Check if crossing occurred in this interval
                if (pol > 0 and curr_cross > it0 and curr_cross <= it1) or \
                   (pol < 0 and curr_cross < it0 and curr_cross >= it1):

                    # Interpolate event time
                    edt = int(abs((curr_cross - it0) *
                              delta_t_ns / (it1 - it0)))
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
    """Render a color event image from an event array.

    Args:
        events (np.ndarray): Array of shape ``(N, 4)`` with rows
            ``(x, y, t_ns, polarity)``.
        width (int): Output image width in pixels.
        height (int): Output image height in pixels.

    Returns:
        np.ndarray: BGR image of shape ``(height, width, 3)``. ON events are
            drawn in red and OFF events in blue.
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

    Supports one or both eyes via the ``eye`` parameter.

    Attributes:
        model (mujoco.MjModel): The MuJoCo model of the robot.
        data (mujoco.MjData): The MuJoCo data object for simulation.
        show_raw_feed (bool): Whether to display the raw camera feed.
        show_ed_feed (bool): Whether to display the event-based camera feed.
        DEBUG (bool): Whether to enable debug logging.
    """

    _EYE_CAMERA_NAMES = {
        "left": "l_eye_camera",
        "right": "r_eye_camera",
    }

    def __init__(self, time, model, data, eye="all", camera_mode="frame_based", show_raw_feed=True, show_ed_feed=False, DEBUG=False):
        """Initialize camera renderers and optional event simulators per eye.

        Args:
            time (int): Initial simulation timestamp in nanoseconds.
            model (mujoco.MjModel): MuJoCo model used by the renderer.
            data (mujoco.MjData): MuJoCo runtime data used by the renderer.
            eye (str): Active eye selection: ``"left"``, ``"right"``, or
                ``"all"``.
            camera_mode (str): ``"frame_based"`` for RGB only or
                ``"event_driven"`` to run event simulation.
            show_raw_feed (bool): If True, display raw eye images in OpenCV
                windows.
            show_ed_feed (bool): If True, display event images in OpenCV
                windows.
            DEBUG (bool): If True, emit additional logging.

        Raises:
            ValueError: If ``eye`` is not one of ``"left"``, ``"right"``, or
                ``"all"``.

        Notes:
            - If ``show_ed_feed`` is requested while ``camera_mode`` is
              ``"frame_based"``, event mode is enabled automatically.
        """

        if eye == "all":
            eye_keys = ["left", "right"]
        elif eye in ("left", "right"):
            eye_keys = [eye]
        else:
            raise ValueError(f"eye must be 'all', 'left', or 'right', got '{eye}'")

        self.model = model
        self.data = data
        self.show_raw_feed = show_raw_feed
        self.show_ed_feed = show_ed_feed
        self.DEBUG = DEBUG
        self.camera_mode = camera_mode
        self.latest_raw_frames = {}
        self.latest_ed_frames = {}

        if show_ed_feed and self.camera_mode == "frame_based":
            self.camera_mode = "event_driven"
            logging.warning(
                "Event camera disabled. Enabling it to show event feed.")

        self._eyes = []
        for eye_key in eye_keys:
            camera_name = self._EYE_CAMERA_NAMES[eye_key]
            renderer = mujoco.Renderer(model)
            renderer.update_scene(self.data, camera=camera_name)
            pixels = renderer.render()
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

            label = eye_key.capitalize()
            camera_feed_window_name = f'{label} Eye Camera Feed'
            events_window_name = f'{label} Eye Event Feed'

            if self.camera_mode == "event_driven":
                esim = CameraEventSimulator(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY), time)
                if show_ed_feed:
                    cv2.namedWindow(events_window_name)

                    def on_thresh_slider(val, _esim=esim):
                        val /= 100
                        _esim.Cm = val
                        _esim.Cp = val

                    cv2.createTrackbar("Threshold", events_window_name, int(
                        esim.Cm * 100), 100, on_thresh_slider)
                    cv2.setTrackbarMin("Threshold", events_window_name, 1)
            else:
                esim = None

            self._eyes.append({
                "key": eye_key,
                "camera_name": camera_name,
                "renderer": renderer,
                "esim": esim,
                "camera_feed_window_name": camera_feed_window_name,
                "events_window_name": events_window_name,
                "events": None,
            })

    def update_camera(self, time):
        """Render current eye frames, update event streams, and refresh windows.

        Args:
            time (int): Current simulation timestamp in nanoseconds.

        Returns:
            dict[str, np.ndarray | None]: Mapping from eye key (``"left"`` and/or
            ``"right"``) to its event array. When running in frame-based mode,
            values are ``None`` for each eye.

        Side Effects:
            - Updates ``latest_raw_frames`` and ``latest_ed_frames`` caches.
            - Pushes frames to OpenCV windows when the corresponding
              ``show_*_feed`` flags are enabled.
        """

        results = {}
        for eye in self._eyes:
            eye["renderer"].update_scene(self.data, camera=eye["camera_name"])
            pixels = eye["renderer"].render()
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            self.latest_raw_frames[eye["key"]] = pixels.copy()
            if self.show_raw_feed:
                cv2.imshow(eye["camera_feed_window_name"], pixels)
            if eye["esim"] is not None:
                eye["events"] = eye["esim"].imageCallback(
                    cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY), time)
                event_frame = make_camera_event_frame(eye["events"])
                self.latest_ed_frames[eye["key"]] = event_frame
                if self.show_ed_feed:
                    cv2.imshow(eye["events_window_name"], event_frame)
                if self.DEBUG and len(eye["events"]):
                    logging.info(
                        f"Generated {len(eye['events'])} camera events for {eye['key']} eye.")
            results[eye["key"]] = eye["events"]

        if self.show_ed_feed or self.show_raw_feed:
            cv2.waitKey(1)

        return results

    def save_feed_images(
        self,
        output_dir: str,
        prefix: str = "",
        time_ns: int | None = None,
        include_raw: bool = True,
        include_ed: bool = True,
    ) -> list[str]:
        """Save one image per eye for raw and/or event-driven camera feeds.

        This method writes the latest rendered frames stored by ``update_camera``
        to disk. It is intended to be called after each simulation step where
        ``update_camera`` ran, or at a lower save cadence (for example every
        N-th step) to reduce disk I/O.

        Args:
            output_dir: Directory where image files are written. It is created
                automatically if it does not exist.
            prefix: Optional filename prefix (for example ``"frame_"``).
            time_ns: Optional timestamp (typically simulation time in ns) added
                to each filename as ``_<time_ns>``.
            include_raw: If True, save RGB camera frames captured from each eye.
            include_ed: If True, save event-rendered frames for each eye.

        Returns:
            list[str]: Absolute/relative paths of all files written in this call.

        Notes:
            - Raw frames are internally stored as RGB and converted to BGR before
              ``cv2.imwrite`` so saved colors look correct.
            - If an eye/feed has no available frame yet (for example event feed
              before the first ``update_camera``), that file is skipped.

        Example:
            eye_camera_object.update_camera(int(data.time * 1e9))
            eye_camera_object.save_feed_images(
                output_dir="neuromorphic_body_schema/camera_frames",
                prefix="frame_",
                time_ns=int(data.time * 1e9),
                include_raw=True,
                include_ed=True,
            )
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        time_label = f"_{int(time_ns)}" if time_ns is not None else ""
        saved_files = []

        if include_raw:
            for eye_key, frame in self.latest_raw_frames.items():
                filename = f"{prefix}{eye_key}_eye_raw{time_label}.png"
                file_path = out_path / filename
                # Convert RGB back to BGR for OpenCV file writing.
                cv2.imwrite(str(file_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                saved_files.append(str(file_path))

        if include_ed:
            for eye_key, frame in self.latest_ed_frames.items():
                filename = f"{prefix}{eye_key}_eye_ed{time_label}.png"
                file_path = out_path / filename
                cv2.imwrite(str(file_path), frame)
                saved_files.append(str(file_path))

        return saved_files
