"""
sample_vision.py

Description:
Reads the cameras in sensor_config.CAMERAS from a simulation that is already
running. Start main.py first, then run this script. For each camera it renders
the RGB frame from a local copy of the simulation and converts the change
between frames into events with the event-camera simulator
(CameraEventSimulator). It shows the RGB frame and the events (ON red, OFF blue)
in a window per camera and prints the number of events.

Usage (from the repository root):
    python src/sensor/sample_vision.py
    python src/sensor/sample_vision.py --headless    # print only, no windows

"""

# Add the project root to the path for direct script execution
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import argparse
import time

import cv2
import mujoco
import numpy as np
from neuromorphic_body_schema.helpers.ed_cam import (CameraEventSimulator,
                                                     make_camera_event_frame)
from src.sim_link import SimClient, SimMirror
from src.sensor.sensor_config import CAMERAS

WIDTH, HEIGHT = 320, 240  # pixels
FRAME_PERIOD = 0.03  # sec, time between frames
PRINT_PERIOD = 0.5  # sec, time between printed lines


def render(renderer, data, camera: str) -> np.ndarray:
    """
    Renders one camera.

    Args:
        renderer (mujoco.Renderer): Renderer for the model.
        data (mujoco.MjData): MuJoCo data object.
        camera (str): Name of the camera.

    Returns:
        np.ndarray: BGR image of shape (HEIGHT, WIDTH, 3), ready for OpenCV.
    """
    renderer.update_scene(data, camera=camera)
    return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read the cameras of a running simulation.")
    parser.add_argument("--headless", action="store_true",
                        help="print only, no windows")
    args = parser.parse_args()

    try:
        client = SimClient()
    except ConnectionRefusedError:
        sys.exit("No simulation found. Start main.py first.")

    with client:
        mirror = SimMirror(client)
        renderer = mujoco.Renderer(mirror.model, HEIGHT, WIDTH)
        last_time_ns = int(mirror.data.time * 1e9)
        event_cameras = {
            name: CameraEventSimulator(
                cv2.cvtColor(render(renderer, mirror.data, name), cv2.COLOR_BGR2GRAY),
                last_time_ns)
            for name in CAMERAS.values()
        }

        event_counts = {name: 0 for name in CAMERAS.values()}
        last_print = time.perf_counter()
        try:
            while True:
                time_ns = int(mirror.update() * 1e9)
                # the event simulator needs the time to move forward between frames
                if time_ns > last_time_ns:
                    for number, name in CAMERAS.items():
                        frame = render(renderer, mirror.data, name)
                        # array of (x, y, timestamp in ns, polarity) events
                        events = event_cameras[name].imageCallback(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), time_ns)
                        event_counts[name] += len(events)
                        if not args.headless:
                            cv2.imshow(f"{number}: {name}", frame)
                            cv2.imshow(f"{number}: {name} events",
                                       make_camera_event_frame(events, WIDTH, HEIGHT))
                    last_time_ns = time_ns
                if not args.headless:
                    cv2.waitKey(1)

                if time.perf_counter() - last_print >= PRINT_PERIOD:
                    counts = "  ".join(
                        f"{number}: {name} {event_counts[name]:>6} events"
                        for number, name in CAMERAS.items())
                    print(f"t = {time_ns / 1e9:8.3f} s  {counts}")
                    event_counts = {name: 0 for name in CAMERAS.values()}
                    last_print = time.perf_counter()
                time.sleep(FRAME_PERIOD)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
