import pybullet as p
import pybullet_data

### GET JOINT INFO ###
def get_joint_info(gripperID, jointID):
    jointInfo = p.getJointInfo(gripperID, jointID)
    id = jointInfo[0]
    name = jointInfo[1]
    type = jointInfo[2]
    return id, name, type

### GET JOINT KINEMATICS ###
def get_joint_kinematics(gripperID, jointID):
    pos = p.getJointState(gripperID, jointID)[0]
    vel = p.getJointState(gripperID, jointID)[1]
    torq = p.getJointState(gripperID, jointID)[3]
    return pos, vel, torq

### GET JOINT RANGES ###
def get_joint_ranges(gripperID, jointID):
    jointInfo = p.getJointInfo(gripperID, jointID)

    ll, ul = jointInfo[8:10]
    jr = ul - ll
    # joint_dumping.append(0.1 if _joint_name_to_ids[joint_name] in self._joints_to_control else 100.)
    return ll, ul, jr

