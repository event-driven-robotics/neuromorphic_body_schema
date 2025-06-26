"""
ik_solver.py

Author: Ruidong Ma
Affiliation: Sheffield Hallam University
Date: 01.05.2025

Description: 
This script contains inverse kinematic solver for iCub's arm's 6D manipualtion. Given the target end-effector's position 
and quaternion, the Levenberg-Marquardt solver is used to iteratively caculate the joint configuration to reach such goal.
Please note that, current ik solver is only for a single arm while a valid kinematic chain should be predefined. 

"""


import copy
import logging
from math import pi
from typing import List

import mujoco
import numpy as np


def compute_relative_transform(model, data, body_name1, body_name2):
    """
    Compute the transformation matrix T=[Rotation,Translation]

    Args:
    model: mujoco model attribute
    data: mujoco data attribute
    boday_name1: target frame
    body_name2: current frame

    """
    id1 = model.body(body_name1).id
    id2 = model.body(body_name2).id

    p1 = data.xpos[id1]           # shape (3,)
    p2 = data.xpos[id2]           # shape (3,)
    R1 = data.xmat[id1].reshape(3, 3)
    R2 = data.xmat[id2].reshape(3, 3)

    # 相对旋转：R_rel = R1^T * R2
    R_rel = R1.T @ R2

    # 相对平移：p_rel = R1^T * (p2 - p1)
    p_rel = R1.T @ (p2 - p1)

    return R_rel, p_rel


class Ik_solver:
    def __init__(self, model, data, joint_name: List[str], end_name: str, option: str, damp=0.15):
        """
        Levenberg-Marquardt inverse kinematics solver for 6D manipulation

        Args:
        model: mujoco model attribute
        data: mujoco data attribute
        joint_name: the list of joints being controled
        end_name: end_effector body's name 
        option: to use euler angles or quaternion 
        damp: damping ratio for LM
        alpha: step size for LM

        """

        self.model = model
        self.data = data

        try:
            self.joint_id = [self.model.joint(name).id for name in joint_name]
        except:
            raise ValueError("Error: Joint name does not exist")

        try:
            self.ee_id = model.body(end_name).id
        except:
            raise ValueError(f"Error: end effector {end_name} does not exist ")

        nv = model.nv

        self.jacp = np.zeros((3, nv), dtype=np.float64)
        self.jacr = np.zeros((3, nv), dtype=np.float64)

        self.damp = damp
        # self.alpha = alpha

        assert option in ["euler", "quat"], print(
            "should chooese between euler or quat")
        self.option = option

        n = len(self.joint_id)
        self.I = np.eye(n)

    def check_limites(self, joint):
        """
        To ensure the joints are within the limits
        """
        joints = []
        for j, index in zip(joint, self.joint_id):
            j = max(self.model.jnt_range[index][0], min(
                j, self.model.jnt_range[index][1]))
            joints.append(j)
        return np.array(joints)

    # Quaternion Caculation
    def compute_quat(self, R):
        """
        R (3x3) -> [w,x,y,z]
        Reference: SHoemake  method
        """
        R = np.array(R, dtype=np.float64)
        t = np.trace(R)

        if t > 0:
            S = np.sqrt(t + 1.0) * 2  # S = 4*w
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:

            diag = np.array([R[0, 0], R[1, 1], R[2, 2]])
            i = np.argmax(diag)
            if i == 0:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4*x
                w = (R[2, 1] - R[1, 2]) / S
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S
                z = (R[0, 2] + R[2, 0]) / S
            elif i == 1:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4*y
                w = (R[0, 2] - R[2, 0]) / S
                x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4*z
                w = (R[1, 0] - R[0, 1]) / S
                x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S
                z = 0.25 * S

        return np.array([w, x, y, z], dtype=np.float64)

    def compute_quat_error(self, q_cur, q_tgt):
        """
        To caculate the quaternion error: e = 2 * vec(q_err), where q_err = q_tgt * q_cur^{-1}
        """
        #  q_err = q_tgt * inv(q_cur)
        w0, x0, y0, z0 = q_tgt
        w1, x1, y1, z1 = q_cur
        # inv(q_cur) = [w1, -x1, -y1, -z1]
        qe_w = w0*w1 - x0*(-x1) - y0*(-y1) - z0*(-z1)
        qe_x = w0*(-x1) + x0*w1 + y0*(-z1) - z0*(-y1)
        qe_y = w0*(-y1) - x0*(-z1) + y0*w1 + z0*(-x1)
        qe_z = w0*(-z1) + x0*(-y1) - y0*(-x1) + z0*w1

        #  e = 2 * vec(q_err)
        e = 2.0 * np.array([qe_x, qe_y, qe_z])
        return e

    # Euler angle computation

    def compute_euler(self, R):
        """
        R (3x3) ->[roll pitch yaw]
        """
        sy = np.sqrt(R[0][0]**2+R[1][0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2][1], R[2][2])
            pitch = np.arctan2(-R[2][0], sy)
            yaw = np.arctan2(R[1][0], R[0][0])
        else:
            # Gimbal lock
            roll = np.arctan2(-R[1][2], R[1][1])
            pitch = np.arctan2(-R[2][0], sy)
            yaw = 0
        return np.array([roll, pitch, yaw])

    def angle_diff(self, o_tgt, o_cur):
        """
        Error: o_tgt - o_cur and normalize to [-2pi , 2pi]
        """
        d = o_tgt - o_cur
        return (d + 2*np.pi) % (4*np.pi) - 2*np.pi

    # Levenberg-Marquardt for inverse kinematics

    def compute_dynamic_alpha(self, err_norm, alpha_min=0.001, alpha_max=1.0, decay=1000.0):
        """
        Compute a dynamic alpha based on error norm.
        - err_norm: the norm of the error vector (position + orientation)
        - alpha_min: minimum step size
        - alpha_max: maximum step size
        - decay: controls how quickly alpha decreases as error shrinks
        """
        # Exponential decay: alpha = alpha_min - (alpha_max - alpha_min) * exp(-decay * err_norm)
        alpha = alpha_max - (alpha_max - alpha_min) * np.exp(-decay * err_norm)
        return alpha

    def ik_step(self, target_pos, target_ori, max_iter=100000, pos_thres=0.001, ori_thres=0.001):
        """
        LM for inverse kinematics

        Args:
        target_pos: target 3D pose
        target_ori: target angles
        max_iter: maximum step size
        pos_thres: pose error threshol
        ori_thres: angle error threshold
        """

        q_arm = self.data.qpos[self.joint_id].copy()

        if self.option == "euler":
            assert len(target_ori) == 3, print(
                "target euler angle shoule eqaul to 3")
        if self.option == "quat":
            assert len(target_ori) == 4, print(
                "target quaternion shoulde eqaul to 4")

        for i in range(max_iter):

            mujoco.mj_forward(self.model, self.data)

            # Seems the jacobian matrix function mujoco.mj_jac() from mujoco already did the coordinate transformation between frames
            # so here we use the raotation and transforamtion according to world frame

            current_pos = self.data.xpos[self.ee_id]
            current_rotation = self.data.xmat[self.ee_id].reshape(3, 3)

            if self.option == "euler":
                current_ori = self.compute_euler(current_rotation)
                delta_p = target_pos-current_pos
                delta_o = self.angle_diff(target_ori, current_ori)

            if self.option == "quat":
                current_ori = self.compute_quat(current_rotation)
                delta_p = target_pos-current_pos
                delta_o = self.compute_quat_error(current_ori, target_ori)

            if np.linalg.norm(delta_p) <= pos_thres and np.linalg.norm(delta_o) <= ori_thres:
                return q_arm

            mujoco.mj_jac(self.model, self.data, self.jacp,
                          self.jacr, target_pos, self.ee_id)

            jacp_ = self.jacp[:, self.joint_id]

            jacr_ = self.jacr[:, self.joint_id]

            # Jacobian matrix : [6,joints] and error matrix :[pos,orientation]
            J = np.vstack((jacp_, jacr_))       # (6, n)
            cond0 = np.linalg.cond(J)
            if cond0 > 1e8:
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

            err_norm = np.linalg.norm(delta_p) + np.linalg.norm(delta_o)
            alpha = self.compute_dynamic_alpha(err_norm)
            q_arm = q_arm + alpha * delta_q
            q_arm = self.check_limites(q_arm)
            self.data.qpos[self.joint_id] = q_arm
            # logging.info(
            #     f"err_norm: {err_norm}, alpha: {alpha}, target:{target_pos}, current:{current_pos}, delta_p:{delta_p}, target_ori:{target_ori}, current_ori:{current_ori}, delta_o:{delta_o}, iteration:{i+1}")
            logging.info(
                f"err_norm: {err_norm}, alpha: {alpha}, iteration:{i+1}")
            pass
        raise ValueError(f"Solution not found after {i+1} iterations.")


# if __name__ == "__main__":
#     # Example usage--icub

#     # Load the MuJoCo model and create a simulation
#     model_path = "./neuromorphic_body_schema/models/icub_v2_full_body.xml"
#     model = mujoco.MjModel.from_xml_path(model_path)
#     data = mujoco.MjData(model)
#     mujoco.mj_step(model, data)
#     data_copy = copy.deepcopy(data)

#     # Valid kinematic links shoulde be predefined
#     joint_name = ["l_shoulder_pitch", "l_shoulder_roll",
#                   "l_shoulder_yaw", "l_elbow", "l_wrist_prosup"]
#     end_name = "l_forearm"

#     # IK with Quaternion seems more robust for Icub
#     ik_solver = Ik_solver(model, data_copy, joint_name, end_name, "quat")

#     # Ground truth joints [-0.587,1.13,0.43,1,0]
#     target_pos = [-0.13067764, -0.25348467,  1.12211061]
#     target_ori = [-0.18286161, -0.0885009,   0.46619002, 0.8610436]
#     q_arm = ik_solver.ik_step(target_pos, target_ori)

#     joint_pose = {joint_name: pose for joint_name,
#                   pose in zip(joint_name, q_arm)}

#     print(joint_pose)

#     # Ground truth joints [-1.69,0.895,-0.0787,0.553,0.3]
#     target_pos = [-0.16273532, -0.23288355,  1.20810485]
#     target_ori = [-0.38835096,  0.17977812, -0.02433204,  0.90347734]
#     q_arm = ik_solver.ik_step(target_pos, target_ori)

#     # ' Example useage -- UR5e

#     # model_path="/docker-ros/local_ws/mujoco_menagerie/universal_robots_ur5e/scene.xml"
#     # model = mujoco.MjModel.from_xml_path(model_path)
#     # data = mujoco.MjData(model)
#     # print("Model loaded")
#     # mujoco.mj_step(model, data)

#     # target_pos=[-0.034,  0.492,  0.888]
#     # target_ori=[ 7.07106781e-01, -7.07106781e-01, -5.55111512e-17, -5.55111512e-17]
#     # joint_name=["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]
#     # end_name=model.body("wrist_3_link").id
#     # ik_solver=Ik_solver(model,data,joint_name,end_name,"quat")
#     # ik_solver.ik_step(target_pos,target_ori)
