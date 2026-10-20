"""
Test the sensor configuration and the simulation mirror used by src/sensor.
"""

import mujoco
import pytest

from src.control.joint_config import JOINTS
from src.sim_link import CommandServer, SimClient, SimMirror
from src.sensor.sensor_config import CAMERAS, PROPRIOCEPTION_JOINTS

MjData = getattr(mujoco, "MjData")


def test_proprioception_numbers_match_control():
    """Test that a joint has the same number for moving and for reading it."""
    for number, name in PROPRIOCEPTION_JOINTS.items():
        assert JOINTS[number] == name


def test_proprioception_joints_exist(mujoco_model):
    """Test that every joint to read is both a joint and an actuator in the model."""
    for name in PROPRIOCEPTION_JOINTS.values():
        mujoco_model.joint(name)
        mujoco_model.actuator(name)


def test_cameras_exist(mujoco_model):
    """Test that every camera is in the model."""
    for name in CAMERAS.values():
        mujoco_model.camera(name)


def test_mirror_copies_state(mujoco_model):
    """Test that SimMirror receives the state of the server's data, without stepping."""
    data = MjData(mujoco_model)
    data.joint("r_elbow").qpos[0] = 1.0
    data.joint("r_elbow").qvel[0] = 0.5
    data.actuator("r_elbow").force[0] = 2.0
    server = CommandServer(
        mujoco_model, data, fixed_base=False, address=("localhost", 0)
    )
    server.update()  # main.py calls this once per step

    with SimClient(server.address) as client:
        mirror = SimMirror(client)
        assert mirror.data.joint("r_elbow").qpos[0] == pytest.approx(1.0)
        assert mirror.data.joint("r_elbow").qvel[0] == pytest.approx(0.5)
        assert mirror.data.actuator("r_elbow").force[0] == pytest.approx(2.0)


def test_server_accepts_several_clients(mujoco_model):
    """Test that a movement script and readouts can be connected at the same time."""
    data = MjData(mujoco_model)
    server = CommandServer(
        mujoco_model, data, fixed_base=False, address=("localhost", 0)
    )
    with SimClient(server.address) as first, SimClient(server.address) as second:
        first.set_targets({"r_elbow": 1.2})
        server.update()
        assert second.get_state()["ctrl"]["r_elbow"] == pytest.approx(1.2)


def test_read_joint_states(mujoco_model):
    """Test that the joint state is read per joint number, without stepping."""
    from src.sensor.sample_proprioception import read_joint_states

    data = MjData(mujoco_model)
    data.joint("r_elbow").qpos[0] = 1.0
    data.joint("r_elbow").qvel[0] = 0.5
    data.joint("r_elbow").qacc[0] = 0.25
    data.actuator("r_elbow").force[0] = 2.0

    states = read_joint_states(data)
    assert set(states) == set(PROPRIOCEPTION_JOINTS)
    assert states[16] == {
        "name": "r_elbow",
        "position": 1.0,
        "velocity": 0.5,
        "acceleration": 0.25,
        "torque": 2.0,
    }
