"""
Shared fixtures for tests.
"""

import mujoco
import pytest

from neuromorphic_body_schema.helpers.helpers import MODEL_PATH

MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")


@pytest.fixture(scope="module")
def mujoco_model():
    """Load MuJoCo model once per test module."""
    try:
        model = MjModel.from_xml_path(MODEL_PATH)
    except ValueError as exc:
        # CI may not have STL mesh assets because they are not committed to git.
        # Skip model-dependent tests instead of failing the full suite.
        pytest.skip(f"MuJoCo model assets unavailable: {exc}")
    return model


@pytest.fixture(scope="module")
def mujoco_data(mujoco_model):
    """Create MuJoCo data from model."""
    data = MjData(mujoco_model)
    return data
