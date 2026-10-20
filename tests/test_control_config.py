"""
Test the joint configuration, sample movement and simulation link in src/control.
"""

import mujoco
import pytest

from src.control.joint_config import JOINTS, POSES, SEQUENCE, VELOCITY_SEQUENCE
from src.control.sample_velocity_movement import move_to_vel
from src.sim_link import CommandServer, SimClient

MjData = getattr(mujoco, "MjData")


def test_joint_numbers_are_contiguous():
    """Test that joint numbers run from 0 without gaps."""
    assert sorted(JOINTS) == list(range(len(JOINTS)))


def test_joint_names_are_unique():
    """Test that no actuator is listed twice."""
    assert len(set(JOINTS.values())) == len(JOINTS)


def test_poses_use_known_joints():
    """Test that every pose only uses joint numbers from JOINTS."""
    for pose_name, pose in POSES.items():
        unknown = set(pose) - set(JOINTS)
        assert not unknown, f"Pose '{pose_name}' uses unknown joints {unknown}"


def test_sequence_uses_known_poses():
    """Test that every sequence entry names a pose and has a positive duration."""
    for pose_name, duration in SEQUENCE:
        assert pose_name in POSES
        assert duration > 0


def test_joints_match_model_actuators(mujoco_model):
    """Test that JOINTS[i] is the actuator behind data.ctrl[i]."""
    assert len(JOINTS) == mujoco_model.nu
    for index, name in JOINTS.items():
        assert mujoco_model.actuator(name).id == index


def test_pose_targets_within_ctrlrange(mujoco_model):
    """Test that every pose target is inside its actuator's ctrlrange."""
    for pose_name, pose in POSES.items():
        for index, target in pose.items():
            low, high = mujoco_model.actuator(JOINTS[index]).ctrlrange
            assert low <= target <= high, (
                f"Pose '{pose_name}': {JOINTS[index]} target {target} "
                f"outside [{low}, {high}]"
            )


def test_client_targets_reach_simulation(mujoco_model):
    """Test that targets sent by a client are applied by the server, without stepping."""
    data = MjData(mujoco_model)
    server = CommandServer(
        mujoco_model, data, fixed_base=False, address=("localhost", 0)
    )
    with SimClient(server.address) as client:
        ranges = client.get_ranges()
        assert ranges["r_elbow"] == pytest.approx(
            tuple(mujoco_model.actuator("r_elbow").ctrlrange)
        )

        client.set_targets({"r_elbow": 1.2})
        server.update()  # main.py calls this once per step
        assert data.actuator("r_elbow").ctrl[0] == pytest.approx(1.2)
        assert client.get_state()["ctrl"]["r_elbow"] == pytest.approx(1.2)

        with pytest.raises(KeyError):
            client.set_targets({"not_an_actuator": 0.0})


def test_velocity_sequence_stays_in_range(mujoco_model):
    """Test that the velocity sequence, starting from home, keeps every target in range."""
    positions = {index: target for index, target in POSES["home"].items()}
    for velocities, duration in VELOCITY_SEQUENCE:
        assert duration > 0
        for index, velocity in velocities.items():
            assert index in JOINTS
            low, high = mujoco_model.actuator(JOINTS[index]).ctrlrange
            positions[index] = positions.get(index, 0.0) + velocity * duration
            assert low <= positions[index] <= high, (
                f"{JOINTS[index]} would reach {positions[index]:.3f}, "
                f"outside [{low}, {high}]"
            )


class FakeClient:
    """Stands in for SimClient; the simulation time advances 10 ms per request."""

    def __init__(self):
        self.time = 0.0
        self.ctrl = {"r_elbow": 1.0}
        self.sent = []

    def get_ranges(self):
        return {"r_elbow": (0.1, 1.8)}

    def get_state(self):
        return {"time": self.time, "ctrl": dict(self.ctrl)}

    def set_targets(self, targets):
        self.sent.append(targets["r_elbow"])
        self.ctrl.update(targets)
        self.time += 0.01
        return self.get_state()


def test_move_to_vel_moves_target_at_velocity():
    """Test that the target moves at the requested velocity for the duration."""
    client = FakeClient()
    move_to_vel(client, {16: 0.5}, duration=0.2)  # 16 = r_elbow
    assert client.sent[-1] == pytest.approx(1.0 + 0.5 * 0.2)
    steps = [b - a for a, b in zip(client.sent, client.sent[1:])]
    assert all(step == pytest.approx(0.5 * 0.01) for step in steps[:-1])


def test_move_to_vel_stops_at_limit():
    """Test that the target stops at the joint limit instead of running past it."""
    client = FakeClient()
    move_to_vel(client, {16: 5.0}, duration=0.5)
    assert max(client.sent) == pytest.approx(1.8)
