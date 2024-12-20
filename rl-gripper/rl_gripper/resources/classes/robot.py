import pybullet as p
import rl_gripper.resources.functions.jointFunctions as jointFunctions
from rl_gripper.resources.functions.helper import load_config

import numpy as np
import math
import cv2
import random
import os
import ntpath
import time
import torch

config = load_config()

### GRIPPER SETTINGS ###
gripperIndices = [8, 9, 10, 11, 12, 13]
maxJointVel = 3.14
endEffectorIdx = 15  #TCP

LOWER_JOINT_LIMITS = np.array([-6.283, -2.059, -3.927, -6.283, -1.692, -6.283])
UPPER_JOINT_LIMITS = np.array([6.283, 2.094, 0.191, 6.283, 3.141, 6.283])

### CAMERA SETTINGS ###
height = config['camera']['height']
width = config['camera']['width']
aspect = width / height
near = config['camera']['near']
far = config['camera']['far']


fov = config['camera']['fov']



class Robot:

    def __init__(self, client, gripper_start_pos):
        self.client = client

        robot_start_pos = [0, 0, 0.01]
        robot_start_orn = p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.client)
        f_path = "rl_gripper/resources/models/robot.urdf"

        self.id = p.loadURDF(f_path, robot_start_pos, robot_start_orn, flags=p.URDF_MAINTAIN_LINK_ORDER, physicsClientId=self.client)

        #actual_gripper_start_x = np.random.normal(gripper_start_pos[0], 0.03)
        #actual_gripper_start_y = np.random.normal(gripper_start_pos[1], 0.03)
        actual_gripper_start_z = np.random.normal(gripper_start_pos[2], gripper_start_pos[2]/4)

        self.gripper_start_pos = [gripper_start_pos[0], gripper_start_pos[1], actual_gripper_start_z]
        self.gripper_start_orn = p.getQuaternionFromEuler([math.pi, 0, 0], physicsClientId=self.client)

        self.start_joint_angles = p.calculateInverseKinematics(self.id, endEffectorIdx, self.gripper_start_pos,
                                                               self.gripper_start_orn,
                                                               lowerLimits=LOWER_JOINT_LIMITS, upperLimits=UPPER_JOINT_LIMITS,
                                                               physicsClientId=self.client)
        p.resetJointStatesMultiDof(self.id, [1, 2, 3, 4, 5, 6],
                                   [[self.start_joint_angles[0]], [self.start_joint_angles[1]],
                                    [self.start_joint_angles[2]], [self.start_joint_angles[3]],
                                    [self.start_joint_angles[4]], [self.start_joint_angles[5]]],
                                   physicsClientId=self.client)

        self.curr_joint_angles = self.start_joint_angles

        # Get the link state (assuming this returns a list or array-like object)
        link_state = p.getLinkState(self.id, endEffectorIdx, physicsClientId=self.client)
        position = link_state[4]  # Extract the relevant part
        self.state = torch.tensor(list(position) + [0, 0], dtype=torch.float32)

        self.timer = 0
        self.counter = 0

    def get_ids(self):
        return self.client, self.id

    def apply_action(self, action, dist_to_goal):
        ### APPLY POSITIONAL ACTION TO ACTUATORS - PIXEL TO RELATIVE POSITION ###

        self.timer += 1

        action[3] = 0.5

        if self.timer > 40:
            action[3] = -0.5
            action[0] = 0.2

        if self.timer > 80:
            action[3] = 0
            action[0] = 0
            action[2] = -0.3

        # ### POSITION ###
        rot_matrix_tcp_to_world = np.array(
            p.getMatrixFromQuaternion(p.getLinkState(self.id, endEffectorIdx, physicsClientId=self.client)[5])).reshape(3, 3)
        curr_tcp_pos_world = p.getLinkState(self.id, endEffectorIdx, physicsClientId=self.client)[4]  # 3x1
        action_pos_tcp = np.array([action[0], action[1], action[2]]) * 0.025  # 3x1
        action_pos_world = rot_matrix_tcp_to_world @ action_pos_tcp  # 3x1
        next_tcp_pos_world = curr_tcp_pos_world + action_pos_world  # 3x1

        ### POSITION YAW TEST ###
        # rot_matrix_tcp_to_world = np.array(
        #     p.getMatrixFromQuaternion(p.getLinkState(self.id, endEffectorIdx)[5])).reshape(3, 3)
        # curr_tcp_pos_world = self.gripper_start_pos
        # action_pos_tcp = np.array([action[0], action[1], action[2]]) * 0.025  # 3x1
        # action_pos_world = rot_matrix_tcp_to_world @ action_pos_tcp  # 3x1
        # next_tcp_pos_world = curr_tcp_pos_world + action_pos_world  # 3x1

        ### ORIENTATION  EULER ###
        # rot_matrix_cam = np.array(p.getMatrixFromQuaternion(p.getLinkState(self.id, endEffectorIdx)[1])).reshape(3, 3)
        # curr_cam_orn_world = p.getEulerFromQuaternion(p.getLinkState(self.id, endEffectorIdx)[1])
        # action_orn_cam = np.array([0, 0, action[3]])
        # action_orn_world = rot_matrix_cam @ action_orn_cam
        # next_endEff_orn_world = p.getQuaternionFromEuler(curr_cam_orn_world + action_orn_world)

        ### ORIENTATION QUATS ###
        curr_tcp_orn_world = p.getLinkState(self.id, endEffectorIdx, physicsClientId=self.client)[5]
        # rot_angle = action[3]*1.5
        # rot_quat = np.array([0, 0, math.sin(rot_angle / 2), math.cos(rot_angle / 2)])
        # next_endEff_orn_world = self.multiply_quaternions(rot_quat, curr_tcp_orn_world)
        # print(next_endEff_orn_world)

        new_joint_angles = p.calculateInverseKinematics(self.id, endEffectorIdx, next_tcp_pos_world, curr_tcp_orn_world,
                                                        lowerLimits=LOWER_JOINT_LIMITS,
                                                        upperLimits=UPPER_JOINT_LIMITS,
                                                        maxNumIterations=50,
                                                        physicsClientId=self.client)
        #print(new_joint_angles[0:6])

        self.set_yaw(action[3])
        self.set_gripper(action[4])

        for i in range(5):
            p.setJointMotorControl2(self.id, i + 1,
                                    controlMode=p.POSITION_CONTROL,
                                    maxVelocity=maxJointVel,
                                    targetPosition=new_joint_angles[i],
                                    physicsClientId=self.client)


    def apply_action_jacobian(self, action):

        trans_action = np.array(action[:3]) * 0.6
        rot_action = np.array([0, 0, action[3]]) * 5

        # Desired change in TCP position (dx, dy, dz) and orientation (dRoll, dPitch, dYaw)
        action_pose = np.hstack((trans_action, rot_action))

        joint_positions, _, _ = self.get_joint_data()
        # print(joint_positions[0:6])

        result = p.getLinkState(self.id, endEffectorIdx, computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId=self.client)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result

        zero_vec = [0.0] * len(joint_positions)
        jac_t, jac_r = p.calculateJacobian(self.id, endEffectorIdx, com_trn, joint_positions, zero_vec, zero_vec, physicsClientId=self.client)
        jacobian = np.vstack((np.array(jac_t), np.array(jac_r)))

        # Calculate the change in joint angles
        jacobian_pseudo_inv = np.linalg.pinv(jacobian)  # Moore-Penrose pseudoinverse of the Jacobian
        delta_theta = jacobian_pseudo_inv @ action_pose # Geschwidigkeiten rad/s


        #Enfore Joint Limits
        targetVelocities = np.zeros(6)
        for i in range(6):
            targetVelocities[i] = self.enforce_joint_limits(i+1, delta_theta[i])

        p.setJointMotorControlArray(self.id, [1, 2, 3, 4, 5, 6],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=targetVelocities,
                                    physicsClientId=self.client)

        self.set_gripper(action[4])



    def get_camera_data(self):

        #print("in get_camera_data")
        # Get camera output
        camera_pos = p.getLinkState(self.id, 14, physicsClientId=self.client)[0]
        rot_matrix = np.array(p.getMatrixFromQuaternion(p.getLinkState(self.id, 14, physicsClientId=self.client)[1])).reshape(3, 3)

        lookat_pos = rot_matrix @ np.array([0, 0, 1]) + camera_pos
        upvec = rot_matrix @ np.array([1, 0, 0])
        rightvec = rot_matrix @ np.array([0, 1, 0])

        # p.addUserDebugLine(camera_pos, camera_pos + rightvec, [0, 1, 0], 5, 0.1)
        # p.addUserDebugLine(camera_pos, lookat_pos, [0, 0, 1], 5, 0.1)
        # p.addUserDebugLine(camera_pos, camera_pos + upvec, [1, 0, 0], 5, 0.1)

        view_matrix = p.computeViewMatrix(camera_pos, lookat_pos, upvec, physicsClientId=self.client)
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far, physicsClientId=self.client)  #Abspeichern! ZEIT CHECKEN
        results = p.getCameraImage(width, height, view_matrix, projection_matrix,
                                   shadow=False,
                                   lightColor=[1, 1, 1],
                                   renderer=p.ER_TINY_RENDERER,
                                   physicsClientId=self.client)

        # Extract rgb
        rgb = np.asarray(results[2], dtype=np.uint8)
        rgb = np.reshape(rgb, (width, height, 4))[:, :, :3]

        # # Image for Debugging
        # cv2.imwrite('img/direct' + str(self.counter) + '.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # self.counter = self.counter + 1


        ##### Normal POV
        # normal_view_matrix = p.computeViewMatrixFromYawPitchRoll(
        #     cameraTargetPosition=[0, 0, 0],
        #     distance=1.5,
        #     yaw=70,
        #     pitch=-45,
        #     roll=0,
        #     upAxisIndex=2,
        #     physicsClientId=self.client
        # )
        # normal_proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0,
        #                                                   physicsClientId=self.client)
        # normal_results = p.getCameraImage(640, 480, normal_view_matrix, normal_proj_matrix,
        #                                   shadow=False,
        #                                   lightColor=[1, 1, 1],
        #                                   renderer=p.ER_TINY_RENDERER,
        #                                   physicsClientId=self.client)
        # normal_rgb = np.asarray(normal_results[2], dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
        # cv2.imwrite('img/normal' + str(self.counter) + '.png', cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR))
        # self.counter = self.counter + 1

        #######

        # Extract depth image
        depth_buffer = np.asarray(results[3], np.float32).reshape(height, width)
        depth = 1.0 * far * near / (far - (far - near) * depth_buffer)  #Linearisierung

        depth = (np.array(depth) * 255).reshape(width, height, 1).astype(np.uint8)

        # Extract segmentation mask
        mask = results[4]

        return rgb, depth, mask

    def get_proprioceptive_data(self):
        # Return all 6 Joint Angles
        joint_positions, _, _ = self.get_joint_data()

        return np.array(joint_positions)

    def get_state(self):
        # Return X, Y, Z, Yaw, Gw
        tcp_pos = self.get_tcp_world()
        yaw = self.get_joint_data()[0][5]
        gripper_width = self.get_joint_data()[0][8]
        return np.array([tcp_pos[0], tcp_pos[1], tcp_pos[2], yaw, gripper_width])

    def get_tcp_world(self):
        tcp = p.getLinkState(self.id, 15, physicsClientId=self.client)[0]
        #p.addUserDebugLine(tcp, [tcp[0], tcp[1], tcp[2] - 0.005], [1, 0, 0], 10, 0.1)
        return tcp

    def get_joint_data(self):
        joint_states = p.getJointStates(self.id, range(p.getNumJoints(self.id, physicsClientId=self.client)), physicsClientId=self.client)
        joint_infos = [p.getJointInfo(self.id, i, physicsClientId=self.client) for i in range(p.getNumJoints(self.id, physicsClientId=self.client))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]

        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]

        return joint_positions, joint_velocities, joint_torques

    def print_joint_info(self):
        joint_info = []
        for i in range(p.getNumJoints(self.id, physicsClientId=self.client)):
            id, name, type = jointFunctions.get_joint_info(self.id, i)
            pos, vel, torq = jointFunctions.get_joint_kinematics(self.id, i)
            ll, up, jr = jointFunctions.get_joint_ranges(self.id, i)
            joint_info.append([id, name, type, pos, vel, ll, up, jr])
            print('ID: {}\nName: {}\nType: {}\nLower Limit: {}\nUpper Limit: {}\n'.format(id, name, type, ll, up))

    @staticmethod
    def multiply_quaternions(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return x, y, z, w

    def close_gripper(self):

        gripperWidth = np.clip(self.state[4] + 0.15 * 1, 0.0, 0.8)
        self.state[4] = gripperWidth

        for joint_index in gripperIndices:
            p.setJointMotorControl2(self.id, joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=gripperWidth,
                                    maxVelocity=2,
                                    force=100,
                                    physicsClientId=self.client)

    def set_gripper(self, action):
        mapped_action = np.interp(action, [-1, 1], [0, 0.85])
        # targetVelocity = action * 20
        # targetVelocityEnforced = self.enforce_joint_limits(10, targetVelocity)
        # targetVelocities = np.ones(len(gripperIndices)) * targetVelocityEnforced

        p.setJointMotorControlArray(self.id, gripperIndices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=np.ones(len(gripperIndices)) * mapped_action,
                                    forces=np.ones(len(gripperIndices)) * 200,
                                    physicsClientId=self.client)


    def set_yaw(self, action):
        target_position = np.clip(self.state[3] + 0.15 * action, -math.pi/2, math.pi/2)
        self.state[3] = target_position
        p.setJointMotorControl2(self.id, 6,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position,
                                force=100,
                                physicsClientId=self.client)

    def enforce_joint_limits(self, joint_index, target_velocity, safety_margin=0.1):
        joint_state = p.getJointState(self.id, joint_index, physicsClientId=self.client)
        joint_position = joint_state[0]

        joint_info = p.getJointInfo(self.id, joint_index, physicsClientId=self.client)
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]

        # Define a margin to start reducing velocity
        margin = safety_margin

        if joint_position <= (lower_limit + margin) and target_velocity < 0:
            # Reduce velocity as it approaches the lower limit
            target_velocity *= (joint_position - lower_limit) / margin
        elif joint_position >= (upper_limit - margin) and target_velocity > 0:
            # Reduce velocity as it approaches the upper limit
            target_velocity *= (upper_limit - joint_position) / margin

        return target_velocity

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
