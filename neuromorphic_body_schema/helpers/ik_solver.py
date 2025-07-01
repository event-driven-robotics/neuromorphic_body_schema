"""
ik_solver.py

Author: Ruidong Ma
Affiliation: Sheffield Hallam University
Date: 01.05.2025

Description: 
This script contains inverse kinematic solver for iCub's arm's 6D manipualtion. Given the target end-effector's position 
and quaternion, the Levenberg-Marquardt solver is used to iteratively caculate the joint configuration to reach such goal.
Please note that, current ik solver is only for a single arm while a valid kinematic chain should be predefined. 

NOTE: https://github.com/google-deepmind/mujoco/blob/main/test/engine/testdata/actuation/refsite.xml 
an example of how to use mujoco's inverse kinematics solver which is not the one presented here.
"""


import logging
from typing import List

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_relative_transform(model, data, body_name1, body_name2):
    """
    Compute the relative transformation (rotation and translation) from body_name2 to body_name1.

    Args:
        model: MuJoCo model object.
        data: MuJoCo data object.
        body_name1 (str): Target frame.
        body_name2 (str): Current frame.

    Returns:
        R_rel (np.ndarray): Relative rotation matrix (3x3).
        p_rel (np.ndarray): Relative translation vector (3,).
    """
    try:
        id1 = model.body(body_name1).id
        id2 = model.body(body_name2).id
    except AttributeError:
        raise ValueError(
            f"One of the body names '{body_name1}' or '{body_name2}' does not exist in the model.")

    p1 = data.xpos[id1]           # shape (3,)
    p2 = data.xpos[id2]           # shape (3,)
    R1 = data.xmat[id1].reshape(3, 3)
    R2 = data.xmat[id2].reshape(3, 3)

    # Relative rotation: R_rel = R1^T * R2
    R_rel = R1.T @ R2

    # Relative translation: p_rel = R1^T * (p2 - p1)
    p_rel = R1.T @ (p2 - p1)

    return R_rel, p_rel


class Ik_solver:
    def __init__(self, model, data, joint_name: List[str], end_effector_name: str, option: str, damp=0.15):
        """
        Levenberg-Marquardt inverse kinematics solver for 6D manipulation

        Args:
        model: mujoco model attribute
        data: mujoco data attribute
        joint_name: the list of joints being controled
        end_effector_name: end_effector body's name 
        option: to use euler angles or quaternion 
        damp: damping ratio for LM
        alpha: step size for LM

        """

        self.model = model
        self.data = data

        try:
            self.joint_ids = [self.model.joint(name).id for name in joint_name]
        except:
            raise ValueError("Error: Joint name does not exist")

        try:
            self.end_effector_id = model.body(end_effector_name).id
        except:
            raise ValueError(
                f"Error: end effector {end_effector_name} does not exist ")

        nv = model.nv

        self.jacp = np.zeros((3, nv), dtype=np.float64)
        self.jacr = np.zeros((3, nv), dtype=np.float64)

        self.damp = damp
        # self.alpha = alpha

        assert option in ["euler", "quat"], print(
            "should chooese between euler or quat")
        self.option = option

        n = len(self.joint_ids)
        self.I = np.eye(n)

    def check_limites(self, joint):
        """
        Clips the given joint values to their respective limits as defined in the model.

        Args:
            joint (array-like): Joint values to be checked and clipped.

        Returns:
            np.ndarray: Joint values clipped to their valid range.
        """

        joint = np.asarray(joint)
        lower = self.model.jnt_range[self.joint_ids, 0]
        upper = self.model.jnt_range[self.joint_ids, 1]
        # Vectorized clipping
        return np.clip(joint, lower, upper)

    def compute_quat(self, R_mat):
        """
        Converts a 3x3 rotation matrix to a quaternion in [w, x, y, z] format.

        Args:
            R_mat (np.ndarray): A 3x3 rotation matrix.

        Returns:
            np.ndarray: Quaternion as [w, x, y, z].
        """
        quat = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]
        # Reorder to [w, x, y, z] if needed
        return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)

    def compute_quat_error(self, q_cur, q_tgt):
        """
        Calculates the quaternion orientation error: e = 2 * vec(q_err),
        where q_err = q_tgt * q_cur^{-1}.

        Args:
            q_cur (np.ndarray): Current orientation quaternion [w, x, y, z].
            q_tgt (np.ndarray): Target orientation quaternion [w, x, y, z].

        Returns:
            np.ndarray: Orientation error vector (3,).
        """

        # Convert to [x, y, z, w] for scipy
        q_cur_xyzw = np.array([q_cur[1], q_cur[2], q_cur[3], q_cur[0]])
        q_tgt_xyzw = np.array([q_tgt[1], q_tgt[2], q_tgt[3], q_tgt[0]])

        # Compute relative quaternion: q_err = q_tgt * q_cur^{-1}
        r_cur = R.from_quat(q_cur_xyzw)
        r_tgt = R.from_quat(q_tgt_xyzw)
        r_err = r_tgt * r_cur.inv()
        q_err = r_err.as_quat()  # [x, y, z, w]

        # Orientation error: 2 * vector part of q_err
        e = 2.0 * q_err[:3]
        return e

    def compute_euler(self, R_mat):
        """
        Converts a 3x3 rotation matrix to Euler angles [roll, pitch, yaw] (in radians).

        Args:
            R_mat (np.ndarray): A 3x3 rotation matrix.

        Returns:
            np.ndarray: Euler angles [roll, pitch, yaw] in radians.
        """

        # 'xyz' gives roll, pitch, yaw (intrinsic rotations)
        return R.from_matrix(R_mat).as_euler('xyz', degrees=False)

    def angle_diff(self, o_tgt, o_cur):
        """
        Computes the minimal difference between angles, normalized to [-pi, pi].

        Args:
            o_tgt (np.ndarray): Target angles.
            o_cur (np.ndarray): Current angles.

        Returns:
            np.ndarray: Angle differences in [-pi, pi].
        """

        d = o_tgt - o_cur
        return (d + np.pi) % (2 * np.pi) - np.pi

    def compute_dynamic_alpha(self, err_norm, alpha_min=0.001, alpha_max=1.0, decay=1000.0):
        """
        Computes a dynamic step size alpha for the IK update based on the current error norm.

        Args:
            err_norm (float): The norm of the error vector (position + orientation).
            alpha_min (float): Minimum step size.
            alpha_max (float): Maximum step size.
            decay (float): Controls how quickly alpha decreases as error shrinks.

        Returns:
            float: The dynamically computed step size alpha.
        """

        # Exponential decay: alpha = alpha_min - (alpha_max - alpha_min) * exp(-decay * err_norm)
        alpha = alpha_max - (alpha_max - alpha_min) * np.exp(-decay * err_norm)
        return alpha

    def ik_step(self, target_pos, target_ori, max_iter=50000, pos_thres=0.001, ori_thres=0.001):
        """
        Performs inverse kinematics using the Levenberg-Marquardt method to find joint angles
        that achieve the desired end-effector position and orientation.

        Args:
            target_pos (array-like): Target 3D position of the end-effector (shape: [3,]).
            target_ori (array-like): Target orientation (Euler angles [3,] or quaternion [4,], depending on self.option).
            max_iter (int, optional): Maximum number of iterations. Default is 100000.
            pos_thres (float, optional): Position error threshold for convergence. Default is 0.001.
            ori_thres (float, optional): Orientation error threshold for convergence. Default is 0.001.

        Returns:
            np.ndarray: Joint configuration that achieves the target pose within the specified thresholds.

        Raises:
            ValueError: If a solution is not found within the maximum number of iterations or if the Jacobian is ill-conditioned.
        """

        q_arm = self.data.qpos[self.joint_ids].copy()

        if self.option == "euler":
            assert len(target_ori) == 3, print(
                "target euler angle shoule eqaul to 3")
        if self.option == "quat":
            assert len(target_ori) == 4, print(
                "target quaternion shoulde eqaul to 4")

        for i in range(max_iter):
            self.data.qpos[self.joint_ids] = q_arm
            mujoco.mj_forward(self.model, self.data)

            ''' 
            NOTE: Seems the jacobian matrix function mujoco.mj_jac() from mujoco 
            already did the coordinate transformation between frames so here we use 
            the rotation and transformation according to world frame
            '''

            current_pos = self.data.xpos[self.end_effector_id]
            current_rotation = self.data.xmat[self.end_effector_id].reshape(
                3, 3)

            if self.option == "euler":
                current_ori = self.compute_euler(current_rotation)
                delta_p = target_pos-current_pos
                delta_o = self.angle_diff(target_ori, current_ori)

            if self.option == "quat":
                current_ori = self.compute_quat(current_rotation)
                delta_p = target_pos-current_pos
                delta_o = self.compute_quat_error(current_ori, target_ori)

            if np.linalg.norm(delta_p) <= pos_thres and np.linalg.norm(delta_o) <= ori_thres:
                logging.info(
                    f"Converged after {i+1} iterations with position error {np.linalg.norm(delta_p)} and orientation error {np.linalg.norm(delta_o)}")
                return q_arm

            mujoco.mj_jac(self.model, self.data, self.jacp,
                          self.jacr, current_pos, self.end_effector_id)

            jacp_ = self.jacp[:, self.joint_ids]

            jacr_ = self.jacr[:, self.joint_ids]

            # Jacobian matrix : [6,joints] and error matrix :[pos,orientation]
            J = np.vstack((jacp_, jacr_))       # (6, n)
            cond0 = np.linalg.cond(J)
            if cond0 > 1e8:
                # TODO We might want to try regularization or log a warning and continue, depending on our application.
                raise RuntimeError(
                    f"Jacobian is too ill-conditioned: cond={cond0:.1e}")
            e = np.concatenate((delta_p, delta_o))  # (6,)

            # LM Matrix
            H = J.T @ J + self.damp * self.I  # (n, n)
            g = J.T @ e  # (n,)

            # cacualte joints increment
            try:
                delta_q = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta_q = np.linalg.pinv(H) @ g

            err_norm = np.linalg.norm(np.concatenate((delta_p, delta_o)))
            alpha = self.compute_dynamic_alpha(err_norm)
            q_arm = q_arm + alpha * delta_q
            q_arm = self.check_limites(q_arm)
            # logging.info(
            #     f"err_norm: {err_norm}, alpha: {alpha}, target:{target_pos}, current:{current_pos}, delta_p:{delta_p}, target_ori:{target_ori}, current_ori:{current_ori}, delta_o:{delta_o}, iteration:{i+1}")
            logging.info(
                f"err_norm: {err_norm}, alpha: {alpha}, iteration:{i+1}")
        # TODO You raise a ValueError if the solution is not found. Consider returning the best found q_arm and a flag, or logging the final error for easier debugging.
        raise ValueError(f"Solution not found after {i+1} iterations.")
