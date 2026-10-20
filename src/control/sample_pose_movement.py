"""
sample_pose_movement.py

Description:
Moves the iCub in a simulation that is already running, by position control:
each move goes to target angles (a pose). Start main.py first, then run this
script to play the poses in joint_config.SEQUENCE. Stopping the script leaves
the simulation running.

For velocity control, see sample_velocity_movement.py.

Usage (from the repository root):
    python src/control/sample_pose_movement.py                    # sequence once
    python src/control/sample_pose_movement.py --loop             # repeat until Ctrl+C
    python src/control/sample_pose_movement.py --pose right_arm_up

"""

# Add the project root to the path for direct script execution
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import argparse
import math
import time

from src.control.joint_config import JOINTS, POSES, SEQUENCE
from src.sim_link import SimClient

HOLD_TIME = 0.5  # sec, pause after each move
SEND_PERIOD = 0.01  # sec, time between target updates


def move_to_pose(client: SimClient, pose: dict, duration: float) -> None:
    """
    Moves smoothly from the current targets to `pose` over `duration` seconds of
    simulation time, then holds the pose for HOLD_TIME.

    Args:
        client (SimClient): Connection to the running simulation.
        pose (dict): Dictionary mapping joint numbers (int) to targets in radians (float).
        duration (float): Simulation time in seconds to reach the pose.
    """
    targets = {JOINTS[index]: value for index, value in pose.items()}
    state = client.get_state()
    start = {name: state["ctrl"][name] for name in targets}
    start_time = state["time"]

    elapsed = 0.0
    while elapsed < duration + HOLD_TIME:
        elapsed = state["time"] - start_time
        # cosine ease-in/ease-out: 0 at the start, 1 at the end of the move
        blend = 0.5 - 0.5 * math.cos(math.pi * min(1.0, elapsed / duration))
        state = client.set_targets({
            name: start[name] + blend * (target - start[name])
            for name, target in targets.items()
        })
        time.sleep(SEND_PERIOD)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move the iCub through poses in a running simulation.")
    parser.add_argument("--loop", action="store_true", help="repeat until Ctrl+C")
    parser.add_argument("--pose", choices=sorted(POSES), help="move to one pose only")
    args = parser.parse_args()

    sequence = [(args.pose, 2.0)] if args.pose else SEQUENCE

    try:
        client = SimClient()
    except ConnectionRefusedError:
        sys.exit("No simulation found. Start main.py first.")

    with client:
        try:
            while True:
                for pose_name, duration in sequence:
                    print(f"Moving to {pose_name}")
                    move_to_pose(client, POSES[pose_name], duration)
                if not args.loop:
                    break
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
