import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R


def getJointStates(robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques


def getMotorJointStates(robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
    joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques


def setJointPosition(robot, position, kp=1.0, kv=0.3):
    num_joints = p.getNumJoints(robot)
    zero_vec = [0.0] * num_joints
    if len(position) == num_joints:
        p.setJointMotorControlArray(robot,
                                range(num_joints),
                                p.POSITION_CONTROL,
                                targetPositions=position,
                                targetVelocities=zero_vec,
                                positionGains=[kp] * num_joints,
                                velocityGains=[kv] * num_joints)
    else:
        print("Not setting torque. "
              "Expected torque vector of "
              "length {}, got {}".format(num_joints, len(position)))

def getLinkPose(body_id, link_idx, return_vel=False, quat=False, ext=True):
    result = p.getLinkState(body_id,
                            link_idx,
                            computeLinkVelocity=int(return_vel),
                            computeForwardKinematics=0)

    if return_vel:
        link_trn, link_rot, _, _, _, _, lin_vel, ang_vel = result
        if not quat:
            try:
                r = R.from_quat(link_rot)
                if ext:
                    angles = list(r.as_euler('xyz'))
                else:
                    angles = list(r.as_euler('XYZ'))
            except ValueError as e:
                print(e)
                angles = [0.0]*3

            return list(link_trn) + angles, list(lin_vel) + list(ang_vel)
        else:
            return list(link_trn) + list(link_rot), list(lin_vel) + list(ang_vel)
    else:
        link_trn, link_rot, _, _, _, _ = result
        if not quat:
            try:
                r = R.from_quat(link_rot)
                if ext:
                    angles = list(r.as_euler('xyz'))
                else:
                    angles = list(r.as_euler('XYZ'))
            except ValueError as e:
                print(e)
                angles = [0.0]*3
            return list(link_trn) + angles
        else:
            return list(link_trn) + list(link_rot)

def getObjPose(obj_id):
    link_trn, link_rot = p.getBasePositionAndOrientation(obj_id)
    try:
        r = R.from_quat(link_rot)
        angles = list(r.as_euler('xyz'))
    except np.linalg.LinAlgError as e:
        print(e)
        angles = [0.0]*3

    return list(link_trn) + angles

def getObjVel(obj_id):
    link_lin_vel, link_ang_vel = p.getBaseVelocity(obj_id)
    return list(link_lin_vel) + list(link_ang_vel)

def euler_to_quat(rot_euler):
    r = R.from_euler('xyz', rot_euler)
    return list(r.as_quat())
def quat_to_euler(quat):
    r = R.from_quat(quat)
    return list(r.as_euler('xyz'))

def ik_end_effector(robot_id, link_id, pos, orient, euler=True):
    if euler:
        orn = p.getQuaternionFromEuler(orient)
    else:
        orn = orient
    jd = [0.005] * p.getNumJoints(robot_id)
    joint_pos = p.calculateInverseKinematics(robot_id, link_id, pos, orn,
                                             jointDamping=jd,
                                             maxNumIterations=100,
                                             residualThreshold=0.01)
    return list(joint_pos)

def get_current_camera_placement():
    _, _, _, _, _, _, _, _, yaw, pitch, dist, target = p.getDebugVisualizerCamera()
    return yaw, pitch, dist, target

def set_current_camera_placement(yaw, pitch, dist, target):
    p.resetDebugVisualizerCamera(dist, yaw, pitch, target)