import pybullet as p
import pybullet_data
import time

import resources.functions.jointFunctions as jointFunctions
import resources.functions.helper as helper
from resources.classes import cube, robot, plane


### PHYSICS CLIENT ###yes
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setRealTimeSimulation(0)


### LOAD MODEL ###
#gripperStartPos = [0, 0, 0.06]
#gripperStartOri = p.getQuaternionFromEuler([0, 0, 0])
#planeID = p.loadURDF("plane.urdf")
#gripperID = p.loadURDF("resources/models/xarm6_with_gripper_with_camera_effort.urdf", gripperStartPos, gripperStartOri)
#debugCubeID = p.loadURDF("resources/models/cube.urdf", [0.21, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
#helper.spawn_random_cubes(40)

for id in [9, 12]:
    print(id)
    print("Friction Before:", p.getDynamicsInfo(gripperID, linkIndex=id)[1])
    p.changeDynamics(gripperID, linkIndex=id, lateralFriction=1, rollingFriction=0.01)
    print("Friction After:", p.getDynamicsInfo(gripperID, linkIndex=id)[1])

### PRINT JOINT INFO ###
joint_info = []
for i in range(p.getNumJoints(gripperID)):
    id, name, type = jointFunctions.get_joint_info(gripperID, i)
    pos, vel, torq = jointFunctions.get_joint_kinematics(gripperID, i)
    ll, up, jr = jointFunctions.get_joint_ranges(gripperID, i)
    joint_info.append([id, name, type, pos, vel, ll, up, jr])
    print('ID: {}\nName: {}\nType: {}\nLower Limit: {}\nUpper Limit: {}\n'.format(id, name, type, ll, up))




while True:
    ### CAMERA ###
    camera_pos = p.getLinkState(gripperID, 14)[0]
    camera_orn = p.getEulerFromQuaternion(p.getLinkState(gripperID, 14)[1])
    camera_target = p.getLinkState(gripperID, 15)[0]

    p.addUserDebugLine(camera_pos, camera_target, [1, 0, 0], 5, 0.5)

    view_matrix = p.computeViewMatrix(camera_pos, camera_target, [0, 0, 1])
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    _, _, rgb_flat, depth, segmentation = p.getCameraImage(width, height, view_matrix, projection_matrix, shadow=True,
                                                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
    #helper.render_rgba_flat(width, height, rgb_flat)

    #userParam0 = p.readUserDebugParameter(param0)
    user_yaw = p.readUserDebugParameter(yaw)
    userParam2 = p.readUserDebugParameter(param2)
    userParam3 = p.readUserDebugParameter(param3)
    #userParam4 = p.readUserDebugParameter(param4)
    #userParam5 = p.readUserDebugParameter(param5)
    #userParam6 = p.readUserDebugParameter(param6)
    #userParam7 = p.readUserDebugParameter(param7)

    userParamGripper = p.readUserDebugParameter(paramGripper)

    ### SET MOTOR CONTROL FROM FADER INPUTS ###
    p.setJointMotorControl2(gripperID, 0, controlMode=p.POSITION_CONTROL, targetPosition=0)
    p.setJointMotorControl2(gripperID, 1, controlMode=p.POSITION_CONTROL, targetPosition=user_yaw)
    p.setJointMotorControl2(gripperID, 2, controlMode=p.POSITION_CONTROL, targetPosition=userParam2)
    p.setJointMotorControl2(gripperID, 3, controlMode=p.POSITION_CONTROL, targetPosition=userParam3)
    p.setJointMotorControl2(gripperID, 4, controlMode=p.POSITION_CONTROL, targetPosition=0)
    p.setJointMotorControl2(gripperID, 5, controlMode=p.POSITION_CONTROL, targetPosition=0)
    p.setJointMotorControl2(gripperID, 6, controlMode=p.POSITION_CONTROL, targetPosition=0)
    p.setJointMotorControl2(gripperID, 7, controlMode=p.POSITION_CONTROL, targetPosition=0)

    for joint_index in gripperIndices:
        p.setJointMotorControl2(gripperID, joint_index,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=userParamGripper,
                                force=100)


    p.stepSimulation()
    time.sleep(1. / 240.)

p.disconnect()
