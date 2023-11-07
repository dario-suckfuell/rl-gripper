import pybullet as p
import rl_gripper.resources.functions.jointFunctions as jointFunctions
import numpy as np
import random
import os

### GRIPPER SETTINGS ###
gripperIndices = [8, 9, 10, 11, 12, 13]
maxJointVel = .5

### CAMERA SETTINGS ###
width, height = 64, 64
aspect = width / height
near, far = 0.05, 0.6
fov = 140


class Robot:

    def __init__(self, client):
        gripperStartPos = [0, 0, 0.06]
        gripperStartOri = p.getQuaternionFromEuler([0, 0, 0])
        f_path = "rl_gripper/resources/models/xarm6_with_gripper_with_camera_effort.urdf"

        self.client = client
        self.state = np.array([0, 0.3, -1.3, 0.2], dtype=np.float32)
        # self.state = np.array([1, 0, 0, 0], dtype=np.float32)
        self.id = p.loadURDF(f_path, gripperStartPos, gripperStartOri, flags=p.URDF_MAINTAIN_LINK_ORDER)
        p.resetJointStatesMultiDof(self.id, [1, 2, 3], [[self.state[0]], [self.state[1]], [self.state[2]]])
        p.resetJointStatesMultiDof(self.id, gripperIndices, [[self.state[3]] for i in range(1, 7)])
        p.resetJointStatesMultiDof(self.id, [0, 4, 5, 6, 7], [[0], [0], [0], [0], [0]])
        # print(p.getJointState(self.id, 1))

    def get_ids(self):
        return self.client, self.id

    def apply_action(self, action):
        # Calculating relative action
        ll = np.array([-6.283, -2.059, -3.927, 0.0000])
        up = np.array([6.283, 2.094, 0.191, 0.850])
        step_update = np.array([12.566 / 1000, 4.153 / 500, 4.118 / 500, 0.85 / 50]) * action
        self.state += step_update
        self.state = np.clip(self.state, ll, up)

        # p.setJointMotorControl2(self.id, 0, controlMode=p.POSITION_CONTROL, targetPosition=0)
        p.setJointMotorControl2(self.id, 1, controlMode=p.POSITION_CONTROL, maxVelocity=maxJointVel, targetPosition=self.state[0])
        p.setJointMotorControl2(self.id, 2, controlMode=p.POSITION_CONTROL, maxVelocity=maxJointVel, targetPosition=self.state[1])
        p.setJointMotorControl2(self.id, 3, controlMode=p.POSITION_CONTROL, maxVelocity=maxJointVel, targetPosition=self.state[2])
        p.setJointMotorControl2(self.id, 4, controlMode=p.POSITION_CONTROL, maxVelocity=maxJointVel, targetPosition=0)
        p.setJointMotorControl2(self.id, 5, controlMode=p.POSITION_CONTROL, maxVelocity=maxJointVel, targetPosition=0)
        p.setJointMotorControl2(self.id, 6, controlMode=p.POSITION_CONTROL, maxVelocity=maxJointVel, targetPosition=0)
        # p.setJointMotorControl2(self.id, 7, controlMode=p.POSITION_CONTROL, targetPosition=0)

        for joint_index in gripperIndices:
            p.setJointMotorControl2(self.id, joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.state[3],
                                    maxVelocity=15,
                                    force=100)

    def get_observation(self):
        # Get camera output
        camera_pos = p.getLinkState(self.id, 14)[0]
        camera_target = p.getLinkState(self.id, 15)[0]
        # camera_orn = p.getEulerFromQuaternion(p.getLinkState(self.id, 14)[1])

        p.addUserDebugLine(camera_pos, camera_target, [1, 0, 0], 5, 0.1)

        view_matrix = p.computeViewMatrix(camera_pos, camera_target, [0, 0, 1])
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        _, _, rgb_flat, depth, segmentation = p.getCameraImage(width, height, view_matrix, projection_matrix,
                                                               shadow=True,
                                                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        depth = (np.array(depth)*255).reshape(width, height, 1).astype(np.uint8)

        # Get observation of Tool Center Point
        coords_l = p.getLinkState(self.id, 9)[0]
        coords_r = p.getLinkState(self.id, 12)[0]
        tcp = ((coords_l[0] + coords_r[0]) / 2, (coords_l[1] + coords_r[1]) / 2, (coords_l[2] + coords_r[2]) / 2)
        # print(tcp)
        p.addUserDebugLine(tcp, [tcp[0], tcp[1], tcp[2] - 0.01], [1, 0, 0], 10, 0.1)

        # Maybe get observation of Gripper Width

        return depth, tcp, rgb_flat

    def print_joint_info(self):
        joint_info = []
        for i in range(p.getNumJoints(self.id)):
            id, name, type = jointFunctions.get_joint_info(self.id, i)
            pos, vel, torq = jointFunctions.get_joint_kinematics(self.id, i)
            ll, up, jr = jointFunctions.get_joint_ranges(self.id, i)
            joint_info.append([id, name, type, pos, vel, ll, up, jr])
            print('ID: {}\nName: {}\nType: {}\nLower Limit: {}\nUpper Limit: {}\n'.format(id, name, type, ll, up))
