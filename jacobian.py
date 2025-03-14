import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data


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
          "length {}, got {}".format(num_joints, len(torque)))


def multiplyJacobian(robot, jacobian, vector):
  result = [0.0, 0.0, 0.0]
  i = 0
  for c in range(len(vector)):
    if p.getJointInfo(robot, c)[3] > -1:
      for r in range(3):
        result[r] += jacobian[r][i] * vector[c]
      i += 1
  return result


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())

time_step = 0.01
gravity_constant = -9.81
p.resetSimulation()
p.setTimeStep(time_step)
p.setGravity(0.0, 0.0, gravity_constant)

p.loadURDF("plane.urdf", [0, 0, 0])


robot_id = p.loadURDF("../yumi/yumi_description/urdf/yumi.urdf",[0,0,0])
p.resetBasePositionAndOrientation(robot_id, [0, 0, 0], [0, 0, 0, 1])
numJoints = p.getNumJoints(robot_id)

robotEndEffectorIndex = 21 # LEFT ARM
# robotEndEffectorIndex = 10 # RIGHT ARM

# Set a joint target for the position control and step the sim.

setJointPosition(robot_id, [0.0] * numJoints)
p.stepSimulation()


ee_vel_ref = [0.0, -0.01, 0.0, 0.0, 0, 0.0]
max_force = 100

positions = []
ref_velocities = []

for i in range(20000):
    # Get the joint and link state directly from Bullet.
    pos, vel, torq = getJointStates(robot_id)
    positions.append(pos)

    mpos, mvel, mtorq = getMotorJointStates(robot_id)

    result = p.getLinkState(robot_id,
                            robotEndEffectorIndex,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)
    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    # Get the Jacobians for the CoM of the end-effector link.
    # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
    # The localPosition is always defined in terms of the link frame coordinates.

    zero_vec = [0.0] * len(mpos)
    jac_t, jac_r = p.calculateJacobian(robot_id, robotEndEffectorIndex, com_trn, mpos, zero_vec, zero_vec)

    # print("Link linear velocity of CoM from getLinkState:")
    # print(link_vt)
    # print("Link linear velocity of CoM from linearJacobian * q_dot:")
    # print(multiplyJacobian(robot_id, jac_t, vel))
    # print("Link angular velocity of CoM from getLinkState:")
    # print(link_vr)
    # print("Link angular velocity of CoM from angularJacobian * q_dot:")
    # print(multiplyJacobian(robot_id, jac_r, vel))

    J_t = np.asarray(jac_t)
    J_r = np.asarray(jac_r)
    J = np.concatenate((J_t, J_r), axis=0)
    # print('Jacobian:', J)
    # print('Jacobian:', J.shape)
    # print(len(vel))

    joints_vel = list(np.dot(np.linalg.pinv(J), np.array(ee_vel_ref)))

    ref_velocities.append(joints_vel)

    # robot_joints = [0.0, 0.0] + list(joints_vel[:9]) + [0.0, 0.0] + list(joints_vel[9:18])
    robot_joints = [0.0, 0.0] + list(joints_vel[:7]) + [0.0] + list([joints_vel[8]]) + [0.0] + list(joints_vel[9:16]) + [0.0, 0.0] + list([joints_vel[17]])

    # print(len(robot_joints))

    p.setJointMotorControlArray(robot_id,
                                list(range(numJoints)),
                                p.VELOCITY_CONTROL,
                                targetVelocities=robot_joints,
                                forces=[max_force] * (numJoints))
    p.stepSimulation()


positions = np.array(positions)

plt.figure()
plt.boxplot(positions, labels=list(range(positions.shape[1])))


ref_velocities = np.array(ref_velocities)
plt.figure()
plt.boxplot(ref_velocities, labels=list(range(ref_velocities.shape[1])))

plt.show()