"""
Shared fixtures for tests.
"""

import mujoco
import pytest

from neuromorphic_body_schema.helpers.helpers import MODEL_PATH


@pytest.fixture(scope="module")
def mujoco_model():
    """Load MuJoCo model once per test module."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def mujoco_data(mujoco_model):
    """Create MuJoCo data from model."""
    data = mujoco.MjData(mujoco_model)
    return data
