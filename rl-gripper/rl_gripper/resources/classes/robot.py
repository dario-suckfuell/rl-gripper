import pybullet as p
import rl_gripper.resources.functions.jointFunctions as jointFunctions
import numpy as np
import math
import random
import os
import ntpath

### GRIPPER SETTINGS ###
gripperIndices = [8, 9, 10, 11, 12, 13]
maxJointVel = 8
endEffectorIdx = 15 #TCP

### CAMERA SETTINGS ###
width, height = 64, 64
aspect = width / height
near, far = 0.0003, 0.3 #0.02, 2.0
fov = 120


class Robot:

    def __init__(self, client, gripper_start_pos):
        robot_start_pos = [0, 0, 0.01]
        robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        f_path = "rl_gripper/resources/models/robot.urdf"
        self.ll_joints = np.array([-6.283, -2.059, -3.927, -6.283, -1.692, -6.283])
        self.ul_joints = np.array([6.283, 2.094, 0.191, 6.283, 3.141, 6.283])

        self.client = client
        self.id = p.loadURDF(f_path, robot_start_pos, robot_start_orn, flags=p.URDF_MAINTAIN_LINK_ORDER)

        self.gripper_start_pos = gripper_start_pos
        self.gripper_start_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

        joint_angles = p.calculateInverseKinematics(self.id, endEffectorIdx, self.gripper_start_pos, self.gripper_start_orn,
                                                    lowerLimits=self.ll_joints, upperLimits=self.ul_joints)
        p.resetJointStatesMultiDof(self.id, [1, 2, 3, 4, 5, 6],
                                   [[joint_angles[0]], [joint_angles[1]], [joint_angles[2]], [joint_angles[3]],
                                    [joint_angles[4]], [joint_angles[5]]])

        self.state = np.array([*p.getLinkState(self.id, 14)[0], 0], dtype=np.float32)  # Start Position [X, Y, Z, Gw]
        self.timer = 0

    def get_ids(self):
        return self.client, self.id

    def apply_action_xyz(self, action, dist_to_goal):
        ### APPLY POSITIONAL ACTION TO ACTUATORS - PIXEL TO RELATIVE POSITION ###

        # gripperWidth = np.clip(self.state[3] + 0.1 * action[3], 0.0, 0.85)
        # self.state[3] = gripperWidth

        if dist_to_goal < 0.04:
            self.close_gripper()

        ### POSITION ###
        rot_matrix_endEff_to_world = np.array(p.getMatrixFromQuaternion(p.getLinkState(self.id, endEffectorIdx)[1])).reshape(3, 3)
        curr_endEff_pos_world = p.getLinkState(self.id, endEffectorIdx)[4]  # 3x1
        action_pos_cam = np.array([action[0], action[1], action[2]]) * 0.1  # 3x1
        action_pos_world = rot_matrix_endEff_to_world @ action_pos_cam  # 3x1
        next_endEff_pos_world = curr_endEff_pos_world + action_pos_world  # 3x1

        ### ORIENTATION  EULER ###
        # rot_matrix_cam = np.array(p.getMatrixFromQuaternion(p.getLinkState(self.id, 14)[1])).reshape(3, 3)
        # curr_cam_orn_world = p.getEulerFromQuaternion(p.getLinkState(self.id, 14)[1])
        # action_orn_cam = np.array([0, 0, action[3]])
        # action_orn_world = rot_matrix_cam @ action_orn_cam
        # next_cam_orn_world = p.getQuaternionFromEuler(curr_cam_orn_world + action_orn_world)

        ### ORIENTATION QUATS ###
        # curr_cam_orn_world = p.getLinkState(self.id, 14)[1]
        # rot_winkel = action[3]
        # rot_quat = np.array([0, 0, -math.sin(rot_winkel/2), math.cos(rot_winkel/2)])
        # next_cam_orn_world = self.multiply_quaternions(curr_cam_orn_world, rot_quat)

        joint_angles = p.calculateInverseKinematics(self.id, endEffectorIdx, next_endEff_pos_world, self.gripper_start_orn,
                                                    lowerLimits=self.ll_joints, upperLimits=self.ul_joints)

        for i in range(6):
            p.setJointMotorControl2(self.id, i + 1,
                                    controlMode=p.POSITION_CONTROL,
                                    maxVelocity=maxJointVel,
                                    targetPosition=joint_angles[i])

        # for joint_index in gripperIndices:
        #     p.setJointMotorControl2(self.id, joint_index,
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPosition=gripperWidth,
        #                             maxVelocity=15,
        #                             force=100)

    def get_camera_data(self):
        # Get camera output
        camera_pos = p.getLinkState(self.id, 14)[0]
        rot_matrix = np.array(p.getMatrixFromQuaternion(p.getLinkState(self.id, 14)[1])).reshape(3, 3)

        lookat_pos = rot_matrix @ np.array([0, 0, 1]) + camera_pos
        upvec = rot_matrix @ np.array([1, 0, 0])
        rightvec = rot_matrix @ np.array([0, 1, 0])

        p.addUserDebugLine(camera_pos, camera_pos + rightvec, [0, 1, 0], 5, 0.1)
        p.addUserDebugLine(camera_pos, lookat_pos, [0, 0, 1], 5, 0.1)
        p.addUserDebugLine(camera_pos, camera_pos + upvec, [1, 0, 0], 5, 0.1)

        view_matrix = p.computeViewMatrix(camera_pos, lookat_pos, upvec)
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far) #Abspeichern! ZEIT CHECKEN

        _, _, rgb_flat, depth, segmentation = p.getCameraImage(width, height, view_matrix, projection_matrix,
                                                               shadow=False,
                                                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # rgb_flat = np.array(rgb_flat).reshape(16384, 1).ravel()
        # rgb_flat = np.array(rgb_flat.reshape(16384, 1)).ravel()
        depth = (np.array(depth) * 255).reshape(width, height, 1).astype(np.uint8)

        return depth, rgb_flat

    def get_tcp_world(self):
        tcp = p.getLinkState(self.id, 15)[0]
        p.addUserDebugLine(tcp, [tcp[0], tcp[1], tcp[2] - 0.005], [1, 0, 0], 10, 0.1)
        return tcp

    def print_joint_info(self):
        joint_info = []
        for i in range(p.getNumJoints(self.id)):
            id, name, type = jointFunctions.get_joint_info(self.id, i)
            pos, vel, torq = jointFunctions.get_joint_kinematics(self.id, i)
            ll, up, jr = jointFunctions.get_joint_ranges(self.id, i)
            joint_info.append([id, name, type, pos, vel, ll, up, jr])
            print('ID: {}\nName: {}\nType: {}\nLower Limit: {}\nUpper Limit: {}\n'.format(id, name, type, ll, up))

    def multiply_quaternions(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return (x, y, z, w)

    def close_gripper(self):

        gripperWidth = np.clip(self.state[3] + 0.1 * 1.6, 0.0, 0.7)
        self.state[3] = gripperWidth

        for joint_index in gripperIndices:
            p.setJointMotorControl2(self.id, joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=gripperWidth,
                                    maxVelocity=20,
                                    force=200)


'''
XX (0, b'world_joint', 4)
(1, b'joint1', 0)
(2, b'joint2', 0)
(3, b'joint3', 0)
(4, b'joint4', 0)
(5, b'joint5', 0)
(6, b'joint6', 0)
XX (7, b'gripper_fix', 4)
(8, b'drive_joint', 0)
(9, b'left_finger_joint', 0)
(10, b'left_inner_knuckle_joint', 0)
(11, b'right_outer_knuckle_joint', 0)
(12, b'right_finger_joint', 0)
(13, b'right_inner_knuckle_joint', 0)
XX (14, b'camera_joint', 4)
XX (15, b'camera_target_joint', 4)
'''
