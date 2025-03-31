
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
