from code_as_policy import *
import numpy as np




import numpy as np

class CubePick_0(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_initial_position = np.array([0.5519394500648788, 0.05824922025801705, 0.09446659634606903])
        self.red_cube_initial_position = np.array([0.58833315, 0.06057621, 0.09636923])
        self.task_completed = False

    def distance(self, pos1, pos2):
        return np.linalg.norm(pos1-pos2)

    def select_action(self, state):
        blue_cube_position = state[14:17]
        red_cube_position = state[17:20]
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]

        blue_cube_distance = self.distance(self.blue_cube_initial_position, blue_cube_position)
        red_cube_distance = self.distance(self.red_cube_initial_position, red_cube_position)

        # Scenario of failing the task
        if blue_cube_distance > 0.1 or red_cube_distance > 0.005:
            self.task_completed = False
            # Stop any movement
            action = [0.0]*7

        # Scenario of a completed task
        elif blue_cube_position[2] - self.blue_cube_initial_position[2] >= 0.05:
            # Stop any movement
            action = [0.0]*7
            self.task_completed = True

        # Scenario of grasping and lifting the blue cube
        else:
            action = left_end_effector_vel

            # Close the gripper to grasp the blue cube if hasn't grasped yet
            if left_gripper > 0.0:
                action.append(-1.0) # Closing the gripper

            # After gripping, move up by incrementing the z-velocity
            else:
                action[5] = 0.01 # Moving upwards

        return np.array(action)

    def is_task_completed(self):
        return self.task_completed



class CodePolicyRight:
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        right_end_effector_pose = state[:6]
        right_end_effector_vel = state[6:12]
        right_gripper = state[12]
        right_gripper_vel = state[13] # <0 is for closing, > 0 for opening, must be in [-1, 1]
        objects_state = state[14:] # If empty then there are no objects in sim

        # In this way the velocities are kept constant
        action = right_end_effector_vel + right_gripper_vel

        # Instead, this keeps the robot still
        action = [0.0] * 7

        return action

class CubePick_1(CodePolicyLeft, CodePolicyRight):
    def __init__(self):
        super().__init__()
        self.blue_cube_start_position = [0.5519394500648788, 0.05824922025801705, 0.09446659634606903]
        self.red_cube_start_position = [0.58833315, 0.06057621, 0.09636923]
        self.task_height_threshold = 0.05
        self.task_distance_threshold = 0.1
        self.red_cube_moved_distance = 0.005

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        red_cube_position = state[14:17]
        blue_cube_position = state[23:26]

        action = [0.0] * 7

        # Check if the red cube moved
        red_cube_diff = [i - j for i, j in zip(red_cube_position, self.red_cube_start_position)]
        red_cube_moved = any([x > self.red_cube_moved_distance for x in red_cube_diff])

        # Check if the end effector is too far from the blue cube
        dist_to_blue_cube = sum([(i - j) ** 2 for i, j in zip(blue_cube_position, left_end_effector_pose)]) ** 0.5
        too_far_from_blue_cube = dist_to_blue_cube > self.task_distance_threshold

        # If the above conditions are not met, approach the blue cube and grasp it
        if not red_cube_moved and not too_far_from_blue_cube:
            action[:3] = [i - j for i, j in zip(blue_cube_position, left_end_effector_pose)]
            if dist_to_blue_cube < 0.02:  # If close enough, start closing the gripper
                action[6] = -1.0  # Close the gripper forcefully

        # If the gripper is closed around the blue cube, lift it
        if left_end_effector_pose[2] - self.blue_cube_start_position[2] < self.task_height_threshold:
            action[2] = 1.0  # Move upwards

        # Right effector and gripper should remain still.
        action[3:6] = [0.0] * 3
        action[7] = 0.0

        return action



class CubePick_2(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_initial_height = 0.09446659634606903
        self.blue_cube_current_height = self.blue_cube_initial_height
        self.red_cube_initial_position = [0.58833315, 0.06057621, 0.09636923]
        self.red_cube_current_position = self.red_cube_initial_position
        self.blue_cube_grasped = False
        self.task_complete = False
        self.task_failed = False

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        blue_cube_state = state[14:20] # Assuming the blue cube state is at index 14
        red_cube_state = state[20:26] # Assuming the red cube state is at index 20
        self.blue_cube_current_height = blue_cube_state[2]
        self.red_cube_current_position = red_cube_state[:3]

        if not self.blue_cube_grasped:
            if abs(left_end_effector_pose[2] - self.blue_cube_current_height) < 0.1:  # close enough to grasp
                self.blue_cube_grasped = True
                action = [0.0] * 6
                action.append(-1.0)  # close gripper
            else:  # move closer to the cube
                action = [blue_cube_state[0] - left_end_effector_pose[0],
                          blue_cube_state[1] - left_end_effector_pose[1],
                          blue_cube_state[2] - left_end_effector_pose[2]]
                action.extend([0.0] * 4)

        if self.blue_cube_grasped and not self.task_complete:  # lift the cube
            if self.blue_cube_current_height - self.blue_cube_initial_height < 0.05:
                # move upwards with the same horizontal position
                action = [0.0, 0.0, 0.05]
                action.extend([0.0] * 4)
            else:
                self.task_complete = True
                action = [0.0] * 7  # do nothing action

        # check task failed condition
        if (abs(self.red_cube_current_position[0] - self.red_cube_initial_position[0]) > 0.005 or
            abs(self.red_cube_current_position[1] - self.red_cube_initial_position[1]) > 0.005 or
            abs(self.red_cube_current_position[2] - self.red_cube_initial_position[2]) > 0.005 or
            self.blue_cube_current_height - self.blue_cube_initial_height < -0.005):
            self.task_failed = True
            action = [0.0] * 7  # do nothing action

        return action, self.task_complete, self.task_failed



import numpy as np

class CubePick_3(CodePolicyLeft):
    """Manages the action selection for a bimanual robot"""
    
    # blue cube id, must be replaced with the real id
    BLUE_CUBE_ID = 1
    # red cube id, must be replaced with the real id
    RED_CUBE_ID = 2
        
    def __init__(self):
        super().__init__()
        self.blue_cube_initial_position = np.array([0.5519394500648788, 0.05824922025801705, 0.09446659634606903])
        self.red_cube_initial_position = np.array([0.58833315, 0.06057621, 0.09636923])
        self.max_move_distance = 0.1
        self.blue_cube_grasped = False
    
    def select_action(self, state):
        # Get cubes' positions
        blue_cube_position = self.get_object_position(self.BLUE_CUBE_ID, state)
        red_cube_position = self.get_object_position(self.RED_CUBE_ID, state)
        
        # Check distances
        blue_cube_distance = np.linalg.norm(self.blue_cube_initial_position - blue_cube_position)
        red_cube_distance = np.linalg.norm(self.red_cube_initial_position - red_cube_position)
        
        if red_cube_distance > 0.005:
            raise ValueError("Task failed: Red cube has been moved.")
        
        # If blue cube has not been grasped yet
        if not self.blue_cube_grasped:
            # If we are close to the cube, grasp it
            if blue_cube_distance <= self.max_move_distance:
                self.blue_cube_grasped = True
                left_end_effector_vel = np.array([0, 0, -1]) # Movement upwards
                left_gripper_vel = -1.0 # Close gripper
            # If we are not in proximity of the cube, move towards it
            else:
                left_end_effector_vel = (self.blue_cube_initial_position - state[:6]) / blue_cube_distance
                left_gripper_vel = 0.0 # Open gripper
        else:
            # Blue cube has been grasped, now lift it up
            if blue_cube_position[2] >= self.blue_cube_initial_position[2] + 0.05:
                left_end_effector_vel = np.array([0, 0, 0]) # Stop any additional movement
                left_gripper_vel = 0.0 # Keep gripper closed
            else:
                left_end_effector_vel = np.array([0, 0, 1]) # Continue upwards movement
                left_gripper_vel = 0.0 # Keep gripper closed
                  
        return np.concatenate([left_end_effector_vel, [left_gripper_vel]])

    def get_object_position(self, object_id: int, state):
        """Assuming each object's state is [position, orientation]"""
        return np.array(state[14 + 7*(object_id):14 + 7*(object_id) + 3])



import numpy as np

# Define the class
class CubePick_4(CodePolicyLeft):
  def __init__(self, blue_cube_initial_position, red_cube_initial_position, threshold=0.005):
    super().__init__()
    self.blue_cube_initial_position = np.array(blue_cube_initial_position)
    self.red_cube_initial_position = np.array(red_cube_initial_position)
    self.threshold = threshold
    self.pick_height = blue_cube_initial_position[2] + 0.05 # loading height + 0.05m

  def select_action(self, state):
    blue_cube_position = state['blue_cube'][:3]
    red_cube_position = state['red_cube'][:3]

    # Check the condition if the task is failed
    if np.linalg.norm(self.blue_cube_initial_position - blue_cube_position) > 0.1 or \
       np.linalg.norm(self.red_cube_initial_position - red_cube_position) > 0.005:
      raise Exception("The task failed!")

    if state['blue_cube'][:3][2] < self.pick_height:
      # If height is not reached yet, keep moving up
      action = [0.0, 0.0, 1.0, 0.0] # Third value is velocity in vertical direction
    else:
      # Else, task is completed
      action = [0.0] * 4

    return action



import numpy as np

class CubePick_5(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.initial_height = None
        self.target_height = None
        self.blue_cube_position_index = None
        
    def select_action(self, state):
        if self.initial_height is None:
            # Extract blue cube position
            self.blue_cube_position_index = list(state).index(0.5519394500648788)
            blue_cube_pose = state[self.blue_cube_position_index:self.blue_cube_position_index+3]
            self.initial_height = blue_cube_pose[2]
            self.target_height = self.initial_height + 0.05
            
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]

        # Grasping the blue cube
        if left_end_effector_pose[2] <= self.initial_height+0.005 and left_end_effector_pose[2] >= self.initial_height-0.005:
            action = np.concatenate((np.zeros(6), [-1.0]), axis=0)
        # Lifting the blue cube
        elif left_end_effector_pose[2] < self.target_height:
            action = np.concatenate(([0,0,1], np.zeros(3), [0]), axis=0)
        # Holding lifted blue cube
        else:
            action = np.zeros(7)

        # In case of failure conditions
        red_cube_position_index = list(state).index(0.58833315)
        red_cube_pose = state[red_cube_position_index:red_cube_position_index+3]
        
        if np.linalg.norm(np.array(left_end_effector_pose[:3]) - np.array(state[self.blue_cube_position_index:self.blue_cube_position_index+3])) >= 0.1:
            print("Distance from blue cube is more than 0.1m. Task Failed")
            action = np.zeros(7)
        elif np.linalg.norm(np.array(red_cube_pose[:2]) - np.array([0.58833315, 0.06057621])) >= 0.005 or red_cube_pose[2] != 0.09636923:
            print("Red cube has moved from its loading position. Task Failed")
            action = np.zeros(7)

        return action



class CubePick_6(CodePolicyLeft):
    blue_cube_position = [0.5519394500648788, 0.05824922025801705, 0.09446659634606903]
    red_cube_position = [0.58833315, 0.06057621, 0.09636923]

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        blue_cube_state = state[14:20]
        red_cube_state = state[20:26]

        # Set default action to be stationary
        action = [0.0] * 7

        # Compute the distance from the end effector to the blue cube
        dist_to_blue_cube = np.linalg.norm(np.array(left_end_effector_pose[:3]) - np.array(self.blue_cube_position))

        # Check if the end effector is in the gripping range of the blue cube
        if dist_to_blue_cube <= 0.1:
            # Close the gripper to grip the blue cube
            action[6] = -1.0 # close gripper

            # If blue cube height is less than the target height, lift it up
            if self.blue_cube_position[2] < (0.09446659634606903 + 0.05):
                action[2] = 1.0 # lift the gripper
        else:
            # Move the end effector towards the blue cube
            direction_to_blue_cube = np.array(self.blue_cube_position) - np.array(left_end_effector_pose[:3])
            action[:3] = direction_to_blue_cube / np.linalg.norm(direction_to_blue_cube)

        # Compute the distance moved by the red cube
        red_cube_dist_moved = np.linalg.norm(np.array(self.red_cube_position) - np.array(red_cube_state[:3]))

        # Check if the red cube has moved from its original position
        if red_cube_dist_moved > 0.005:
            print("Task Failed: The red cube has moved from its loading position")
            action = [0.0] * 7 # halt

        return action



# Initiating Class
class CubePick_7(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_initial_pos = [0.5519394500648788, 0.05824922025801705, 0.09446659634606903]
        self.red_cube_initial_pos = [0.58833315, 0.06057621, 0.09636923]
        self.target_blue_cube_pos = [pos for pos in self.blue_cube_initial_pos]
        self.target_blue_cube_pos[2] += 0.05  # The target z position is 0.05m higher
        self.tolerance = 0.005  # Tolerance for object positioning

    def select_action(self, state):
        blue_cube_pos = state[:3]
        red_cube_pos = state[3:6]
        left_end_effector_pose = state[6:12]
        left_end_effector_vel = state[12:18]
        left_gripper = state[18]
        left_gripper_vel = state[19] # <0 is for closing, > 0 for opening, must be in [-1, 1]

        if abs(left_end_effector_pose[0] - self.blue_cube_initial_pos[0]) > 0.1 or \
            abs(left_end_effector_pose[1] - self.blue_cube_initial_pos[1]) > 0.1:
                return [0] * 7  # if distance from blue cube is too large, do nothing

        if abs(red_cube_pos[0] - self.red_cube_initial_pos[0]) > self.tolerance or \
            abs(red_cube_pos[1] - self.red_cube_initial_pos[1]) > self.tolerance or \
            abs(red_cube_pos[2] - self.red_cube_initial_pos[2]) > self.tolerance:
                return [0] * 7  # if red cube has moved significantly, do nothing

        if abs(blue_cube_pos[2] - self.target_blue_cube_pos[2]) < self.tolerance:  
            return [0] * 7 # If the blue cube reached the target position and stays within tolerance stop

        # Perform action to move to the cube
        if self._checkProximity(blue_cube_pos, left_end_effector_pose, 0.1):
            # Action if the arm is close to the blue cube
            if left_gripper < 1.0: # If gripper is not fully closed then close it
                return [0.] * 6 + [-1.]
            
            rel_pos = [self.target_blue_cube_pos[i] - left_end_effector_pose[i] for i in range(3)]
            # Normalized to get velocity, assuming maximum speed is 1.0
            rel_vel = [vel / abs(pos + 1e-10) for pos, vel in zip(rel_pos, left_end_effector_vel)]
            return rel_vel + [0.]  # move in the direction of the blue cube
        
        else:
            # Action if the arm is far from the blue cube
            rel_pos = [self.blue_cube_initial_pos[i] - left_end_effector_pose[i] for i in range(3)]
            # Normalized to get velocity, assuming maximum speed is 1.0
            rel_vel = [vel / abs(pos + 1e-10) for pos, vel in zip(rel_pos, left_end_effector_vel)]
            return rel_vel + [1.]  # move in the direction of the blue cube and open gripper
        
    def _checkProximity(self, pos1, pos2, threshold):
        return abs(pos1[0] - pos2[0]) < threshold and abs(pos1[1] - pos2[1]) < threshold and abs(pos1[2] - pos2[2]) < threshold





# Extend CodePolicyLeft class
class CubePick_8(CodePolicyLeft):
    def __init__(self, blue_cube_pos=[0.5519394500648788, 0.05824922025801705, 0.09446659634606903], 
                 red_cube_pos=[0.58833315, 0.06057621, 0.09636923]):
        super().__init__()    
        self.blue_cube_pos = blue_cube_pos
        self.load_height = blue_cube_pos[2]
        self.red_cube_pos = red_cube_pos

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        objects_state = state[14:] # If empty then there are no objects in sim

        # Calculate euclidean distance to the blue cube
        distance_to_blue_cube = ((left_end_effector_pose[0]-self.blue_cube_pos[0])**2 
                                + (left_end_effector_pose[1]-self.blue_cube_pos[1])**2 
                                + (left_end_effector_pose[2]-self.blue_cube_pos[2])**2)**0.5

        # Situation where task fails
        if (distance_to_blue_cube > 0.1) or (abs(objects_state[7:10]-self.red_cube_pos)>0.005).any(): 
            return [0.0] * 7

        # Grasp the blue cube and move it upwards
        elif (left_end_effector_pose[2] < self.blue_cube_pos[2] + 0.05):
            action = list(left_end_effector_vel) + [1] #1 stands for close grip
            action[2] = 0.05 #0.05m/s to move upward

        # Cube lifted, hold position and keep grip
        else:
            action = [0.0]*6 + [0] #no translation or rotation and keep grip closed

        return action



class CubePick_9(CodePolicyLeft):
    def __init__(self):
       super(CubePick_9,self).__init__()
       self.initial_blue_cube_position = None
       self.initial_red_cube_position = None
       
    def select_action(self, state):
        # Objects state contains positions, orientations, linear velocities and angular velocities of all objects
        # Assuming first object is the blue cube and second object is the red cube
        blue_cube_state = objects_state[:14]
        red_cube_state = objects_state[14:]
        
        blue_cube_position = blue_cube_state[:3]
        blue_cube_orientation = blue_cube_state[3:7]

        if self.initial_blue_cube_position is None:
            self.initial_blue_cube_position = blue_cube_position

        if self.initial_red_cube_position is None:
            self.initial_red_cube_position = red_cube_position

        distance_effector_to_blue_cube = sum([(a-b)**2 for a,b in zip(left_end_effector_pose, blue_cube_position)])**(0.5)
        distance_red_cube_moved = sum([(a-b)**2 for a,b in zip(self.initial_red_cube_position, red_cube_position)])**(0.5)
        
        if distance_effector_to_blue_cube>0.1 or distance_red_cube_moved>0.005:
            print("Task failed.")
            return [0.0] * 7
        
        elif blue_cube_position[2] - self.initial_blue_cube_position[2] >= 0.05:
            print("Task completed.")
            return [0.0] * 7
        
        elif left_gripper<1:
            action = [0.0]*5 + [-1, 0.2]
        
        else:
            action = [0.0]*4 + [0.05, -1, 0.0]
        
        return action



import numpy as np

class VialGrasp_0(CodePolicyLeft):
    def __init__(self):
        super(VialGrasp_0, self).__init__()

        # Vial initial position
        self.vial_initial_position = np.array([0.48573319129014236, -0.02303118607992256, 0.05907549086899239])
        # Task completion height
        self.completion_height = self.vial_initial_position[2] + 0.1

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]

        # Check if the task is failed or completed
        if np.linalg.norm(left_end_effector_pose[:2] - self.vial_initial_position[:2]) > 0.025 or \
                left_gripper < 0.2:
            raise Exception('Task failed')
        if left_end_effector_pose[2] >= self.completion_height:
            print('Task completed')
            return np.zeros(7)
        
        # Linearly increase the z velocity to lift the vial
        action = np.copy(state[6:12])
        action[2] += (self.completion_height - left_end_effector_pose[2]) * 0.1

        # Cap the z velocity to 0.1m/s
        action[2] = min(action[2], 0.1)

        # Grasp the vial. Only close the gripper, do not open
        action = np.append(action, min(left_gripper_vel, 0))

        return action



import numpy as np

# implement the vial grasp policy
class VialGrasp_1(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.grasp_threshold = 0.025  # Distance to start the grasp
        self.lift_height = 0.1  # 0.1 meters above the loading height as specified

    def select_action(self, state):
        left_end_effector_pose = np.array(state[:6])
        left_end_effector_vel = np.array(state[6:12])
        left_gripper = state[12]
        left_gripper_vel = state[13] 
        vial_position = np.array([0.48573319129014236, -0.02303118607992256, 0.05907549086899239])  # given in problem

        # Compute the distance in the x-y plane from current end effector position to the vial
        displacement_xy = np.linalg.norm(left_end_effector_pose[:2] - vial_position[:2])
        
        # First, it will approach the vial's z location:
        if np.abs(left_end_effector_pose[2] - vial_position[2]) > self.grasp_threshold:
            action = np.array([0., 0., -0.1,  # Adjust position controls to get close
                               0., 0., 0.,  # Keep orientation the same
                               0])  # Don't close the gripper yet
        # Once close enough, it will perform the grasp:
        elif displacement_xy <= self.grasp_threshold and left_end_effector_pose[2] - vial_position[2] <= self.lift_height:
            action = np.array([0., 0., 1.,  # Lift up
                               0., 0., 0.,  # Keep orientation the same
                               -1.])  # Close the gripper
        # If not within the threshold, it will stop moving:
        else:
            action = np.array([0.0] * 7)  # Stay static

        return action



import numpy as np

class VialGrasp_2(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.vial_in_hand = False
        self.loading_height = 0.05907549086899239
        self.initial_vial_position = np.array([0.48573319129014236, -0.02303118607992256])
        self.lifting_height = self.loading_height + 0.1

    def select_action(self, state):
        # Extracting state info
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] 
        objects_state = state[14:] 

        # Starting pose for grasp
        starting_pose = np.concatenate([self.initial_vial_position, [self.loading_height], np.array([0.0099971, -0.0399884, 0.0064981])], axis=0)

        # Grasping vial
        if not self.vial_in_hand:
            # Approach vial
            if np.linalg.norm(left_end_effector_pose - starting_pose) > 0.01:
                return starting_pose - left_end_effector_pose
            # Close gripper to grasp vial
            else:
                self.vial_in_hand = True
                return np.concatenate([np.zeros(6), [-1.]]) # Close the gripper with max speed

        # Lifting vial
        else:
            lifting_pose = starting_pose.copy()
            lifting_pose[2] += 0.1  # Increase z position to lift vial.

            # If distance in x,y > 0.025m or gripper has opened (failed the task)
            if np.linalg.norm(left_end_effector_pose[:2] - self.initial_vial_position) > 0.025 or left_gripper < 0:
                return np.concatenate([np.zeros(6), [0.]])  # Keep still, task failed.

            # If distance in z < 0.1m (task not completed)
            elif left_end_effector_pose[2] < self.lifting_height:
                return np.concatenate([np.zeros(5), [1.], [0.]])  # Lift up the vial with max speed. 
            
            # Task completed
            else:
                return np.concatenate([np.zeros(6), [0.]])  # Keep still, task completed.



import numpy as np
import math

class VialGrasp_3(CodePolicyLeft):
  
    def __init__(self):
        super().__init__()
        self.vial_position = np.array([0.48573319129014236, -0.02303118607992256, 0.05907549086899239])
        self.loading_height = 0.1
        self.tolerance = 0.025

    def select_action(self, state):

        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]

        objects_state = state[14:] 
        vial_state = objects_state[:7]

        vial_pos = np.array(vial_state[:3])
        vial_orient = vial_state[3:]


        # Compute the euclidean distance between the end effector and the vial in the x-y plane
        distance = np.linalg.norm(vial_pos[:2] - left_end_effector_pose[:2])

        # Compute the required z velocity to reach the target height
        z_vel = self.loading_height - vial_pos[2]

        # If the euclidean distance is within the allowed tolerance and the vial is still on the loading area,
        # let's grasp the vial and lift it by setting the z velocity
        if distance < self.tolerance and vial_pos[2] <= self.loading_height:
            action = np.array([0, 0, z_vel, 0, 0, 0, -1])  # The last -1 is to close the gripper
        else:
            # If the conditions are not met, the task has failed.
            # Let's set the action to keep the robot still and open the gripper
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1])

        return action



import numpy as np

# Define the VialGrasp_4 policy class
class VialGrasp_4(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        vial_pos = np.array([0.48573319129014236, -0.02303118607992256, 0.05907549086899239])
        left_end_effector_pose = np.array(state[:6])
        left_end_effector_vel = np.array(state[6:12])
        left_gripper = state[12]
        left_gripper_vel = state[13]

        distance = np.sqrt(np.sum(np.square(vial_pos[:2] - left_end_effector_pose[:2])))

        # Grasp the vial if it's close
        if np.all(np.isclose(left_end_effector_pose[:3], vial_pos, atol=0.025)) and left_gripper > -0.5:
            left_gripper_vel = -1.0

        # When the vial is grasped, lift it out of the vial carrier
        elif left_gripper < -0.9:
            left_end_effector_vel[2] = np.clip(vial_pos[2] + 0.1 - left_end_effector_pose[2], -1.0, 1.0)
            left_gripper_vel = 0.0

        # If vial is too far in the x-y plane or the fingers loose contact with the vial, the task fails
        elif distance > 0.025 or left_gripper > -0.9:
            left_gripper_vel = 1.0  # Open the gripper
            left_end_effector_vel[:2] = 0.0  # Stop moving

        action = np.concatenate((left_end_effector_vel, [left_gripper_vel, 0.0, 0.0, 0.0, 0.0]))

        return action.tolist()



import numpy as np

class VialGrasp_5(CodePolicyLeft):
    def select_action(self, state):
        super(VialGrasp_5, self).select_action(state)
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        objects_state = state[14:]

        vial_position = np.array([0.48573319129014236, -0.02303118607992256, 0.05907549086899239])
        vial_carrier_position = np.array([0.55, 0, 0.0025])

        # compute the distance in the x-y plane from the vial's loading position
        distance = np.linalg.norm(vial_position[:2] - left_end_effector_pose[:2])

        task_completed = left_end_effector_pose[2] >= (vial_carrier_position[2] + 0.1)
        task_failed = distance > 0.025 or left_gripper < 0.6  # fingers lose contact if gripper < 0.6

        if task_completed:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # keeps the robot still
        elif task_failed:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # close the gripper
        else:
            action = left_end_effector_vel.tolist()
            action.append(0.0)  # keep the gripper still for now
            return action



import numpy as np

class VialGrasp_6(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        # get position and orientation of the vial and vial carrier
        vial_pos = [0.48573319129014236, -0.02303118607992256, 0.05907549086899239]
        vial_carrier_pos = [0.55, 0, 0.0025]
        
        # Compute the desired pose (grasp the vial and lift of 0.1m above loading height)
        target_pos = np.array(vial_pos) 
        target_pos[2] += 0.1

        # Check if the gripper is already at the target position and orientation
        if np.linalg.norm(target_pos - state[:3]) <= 0.025 and state[12] > 0:
            # If yes, keep the gripper position and close the gripper
            action = np.array([0.0] * 7)
            action[-1] = -1
        else:
            # If not, move towards the target position and orientation and open the gripper
            action = np.hstack(((target_pos - state[:3]), [0., 0., 0., 1]))

        return action



import numpy as np
from scipy.spatial.transform import Rotation as R


class VialGrasp_7(CodePolicyLeft):
    rotation_threshold = np.deg2rad(30)  # exceeds this, task failed.
    def __init__(self):
        super().__init__()
        self.vial_initial_position = np.array([0.48573319129014236, -0.02303118607992256, 0.05907549086899239])
        self.vial_initial_orientation = R.from_quat([0.0099971, -0.0399884, 0.0064981, 0.999129]).as_euler('xyz')

    def select_action(self, state):
        left_end_effector_pose = np.array(state[:6]) 
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        vial_state = state[14:20]
        grabbed = False

        action = [0.0] * 14  # Including right end effector and gripper vel
        vial_position, vial_orientation, vial_vertical_angle = vial_state[0:3], R.from_quat(vial_state[3:7]).as_euler('xyz'), R.from_quat(vial_state[3:7]).as_euler('yxz')[1]
        vial_xy_distance = np.linalg.norm(vial_position[0:2] - self.vial_initial_position[0:2])
        
        if vial_xy_distance <= 0.025 and vial_position[2] < self.vial_initial_position[2] + 0.1:
            if vial_vertical_angle > self.rotation_threshold or np.linalg.norm(vial_orientation - np.roll(self.vial_initial_orientation,-1)) > self.rotation_threshold:
                # If the xy_distance is within threshold, check if the vial orientation is within allowable range
                return action
            else:
                grabbed = True
                action[6:9] = [0, 0, 0.01]  # Steadily lift the vial straight up
                action[9] = -1 if left_gripper < 0.8 else 0  # Try to grasp the vial if it isn't already completely grasped; else, do nothing
        return action




import numpy as np

class VialGrasp_8(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.vial_pose = np.array([0.48573319129014236, -0.02303118607992256, 0.05907549086899239])
        self.loading_height = self.vial_pose[2] + 0.1 # Adding the 0.1m above loading height
        self.grip_threshold = 0.025

    def select_action(self, state):
        
        left_end_effector_pose = np.array(state[:6])
        left_end_effector_vel = np.array(state[6:12])

        # Gripper is set to close
        left_gripper_vel = -1

        # Temp action keeps the velocities constant
        temp_action = np.concatenate((left_end_effector_vel, [left_gripper_vel]))

        # If gripper still holds the vial and hasn't strayed too far away
        if (abs(left_end_effector_pose[0:2] - self.vial_pose[0:2]) < self.grip_threshold).all():
            
            # Lifting the vial only in Z direction without affecting other positional states.
            temp_action[2] = self.loading_height - left_end_effector_pose[2]
        
        # If the vial has been lifted above required height
        if left_end_effector_pose[2] >= self.loading_height:
            action = np.zeros(7) # Stop further actions assuming that task is completed
        else:
            action = temp_action
        
        # Padding zeros for the right_end_effector_vel and right_gripper_vel
        action = np.concatenate((action, np.zeros(8)))
        
        return action



import numpy as np

class VialGrasp_9(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] # <0 is for closing, > 0 for opening, must be in [-1, 1]
        right_end_effector_vel = [0] * 6
        right_gripper_vel = 0.0

        # Positions for the vial and the vial_carrier
        vial_pos = [0.48573319129014236, -0.02303118607992256, 0.05907549086899239]
        vial_carrier_pos = [0.55, 0, 0.0025]
       
        #calculating distance in the x-y plane between the robot and the vial
        distance = np.sqrt((vial_pos[0]-left_end_effector_pose[0])**2 + (vial_pos[1]-left_end_effector_pose[1])**2)
        
        if distance > 0.025:
            # Task failed
            return 'Task failed. Vial is out of reach.'
        else:
            # Task successful up until now
            lift = vial_pos[2] + 0.1  # Lift vial 0.1m above loading height
            if left_end_effector_pose[2] < lift:
                # Move robot arm up
                left_end_effector_vel[2] = 0.01
            else:
                # Stop movement once the vial has been lifted to the correct height
                left_end_effector_vel = [0] * 6
                left_gripper_vel = 0.0

        action = list(np.concatenate([left_end_effector_vel, [left_gripper_vel], right_end_effector_vel, [right_gripper_vel]]))

        return action



import numpy as np

class CubeSlide_0(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_position = np.array([0.5331053387889144, -0.04764524062526641, 0.09397886814957689])
        self.red_cube_position = np.array([0.6082525320598264, 0.06880255016518819, 0.09776131752240294])
        self.task_solved = False
        self.task_failed = False

    def select_action(self, state):
        left_end_effector_pose = np.array(state[:6])
        left_end_effector_vel = np.array(state[6:12])
        left_gripper = np.array(state[12])
        left_gripper_vel = np.array(state[13]) # <0 is for closing, > 0 for opening, must be in [-1, 1]
        objects_state = state[14:] # If empty then there are no objects in sim

        gripper_cube_distance = np.linalg.norm(self.blue_cube_position - left_end_effector_pose[:3])
        cubes_distance = np.linalg.norm(self.red_cube_position - self.blue_cube_position)

        # if the end effector is more than 0.1m away from the blue cube, the task failed
        if gripper_cube_distance > 0.1:
            self.task_failed = True
            return [0.0] * 7

        # if the blue cube is less than 0.04m away from the red cube, the task is solved
        if cubes_distance < 0.04:
            self.task_solved = True
            return [0.0] * 7

        # push the blue cube towards the red one
        direction_vector = self.red_cube_position - self.blue_cube_position
        action = np.concatenate([direction_vector / np.linalg.norm(direction_vector), [0, 0, 0]])

        return action.tolist() + [0]



import numpy as np

class CubeSlide_1(CodePolicyLeft):
    
    def __init__(self):
        super().__init__()
        
        # Setting the goal
        self.cube_goal = np.array([0.6082525320598264, 0.06880255016518819, 0.09776131752240294]) #red cube position
        self.cube_start = np.array([0.5331053387889144, -0.04764524062526641, 0.09397886814957689]) #blue cube position

    def select_action(self, state):
        
        # Define robot's state
        robot_pose = np.array(state[:6]) #x, y, z of robot
        robot_vel = np.array(state[6:12])
        robot_gripper = state[12]
        robot_gripper_vel = state[13] 
        objects_state = state[14:] 
        
        # Compute the distance between red_cube and blue_cube
        distance_cubes = np.linalg.norm(self.cube_goal[:2] - self.cube_start[:2])
        
        # Check if task is done
        if distance_cubes < 0.04:
            action = np.zeros(7) # Returns no movement
            return action.tolist()
        
        # If blue_cube distance from the robot more than 0.1m, fail the task
        distance2cube = np.linalg.norm(robot_pose[:2] - self.cube_start[:2])
        if distance2cube > 0.1:
            print("Failed to accomplish task.")
            action = np.zeros(7)
            return action.tolist()
        
        # Action plan to push the blue_cube towards red_cube
        diff = self.cube_goal[:2] - robot_pose[:2]
        action = np.concatenate([-diff, [0, 0, 0, 0, 0]], axis=0) # Moving towards the destination on x, y axes.

        return action.tolist()



import numpy as np

class CubeSlide_2(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def calc_distance(self, pos1, pos2):
        # Calculate Euclidean distance between two points
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        objects_state = state[14:]

        # Define blue and red cubes positions
        blue_cube_pos = np.array([0.5331053387889144, -0.04764524062526641, 0.09397886814957689])
        red_cube_pos = np.array([0.6082525320598264, 0.06880255016518819, 0.09776131752240294])

        # Calculate distances to the cubes
        dist_blue = self.calc_distance(left_end_effector_pose[:3], blue_cube_pos)
        dist_red = self.calc_distance(blue_cube_pos, red_cube_pos)

        # Define action to stand still
        action = [0.0] * 7

        # If task is solved (cubes are close enough), action is to stand still
        if dist_red < 0.04:
            return action

        # If the blue cube is in reach, close the gripper and move towards red cube
        elif dist_blue <= 0.1:
            action[:6] = (red_cube_pos - left_end_effector_pose[:3])
            action[6] = -1.0 # Close the gripper
            return action

        # If the blue cube is out of reach, move towards it
        else:
            action[:6] = (blue_cube_pos - left_end_effector_pose[:3])
            action[6] = 0.0 # Open the gripper
            return action



import numpy as np

BLUE_CUBE_IDX = 14
RED_CUBE_IDX = 17
END_EFFECTOR_IDX = 0

class CubeSlide_3(CodePolicyLeft):
    
    def __init__(self):
        super().__init__()

    def compute_distance(self, state, idx1, idx2): 
        obj1 = state[idx1: idx1+3]
        obj2 = state[idx2: idx2+3]
        return np.linalg.norm(np.array(obj1) - np.array(obj2))
        
    def select_action(self, state):
        blue_cube_distance = self.compute_distance(state, BLUE_CUBE_IDX, END_EFFECTOR_IDX)
        blue_red_distance = self.compute_distance(state, BLUE_CUBE_IDX, RED_CUBE_IDX)
        
        action = [0.0] * 7  #initialize action to static 

        #checking if task is solved
        if blue_red_distance < 0.04:
            print("Task Solved: Blue cube is close enough to red cube.")
            return action
            
        elif blue_cube_distance > 0.1:   #checking if task is failed
            print("Task Failed: Distance of the end effector from the blue cube is more than 0.1m.")
            return action
            
        else:
            blue_cube_pos = state[BLUE_CUBE_IDX:BLUE_CUBE_IDX+3]
            red_cube_pos = state[RED_CUBE_IDX:RED_CUBE_IDX+3]

            # Calculate direction vector: red_cube_pos - blue_cube_pos, normalized
            direction = np.array(red_cube_pos) - np.array(blue_cube_pos)
            direction /= np.linalg.norm(direction)

            # Speed up closing the gripper and move towards direction of the red cube
            action[:3] = direction.tolist()
            action[-1] = -1.0

        return action



import numpy as np

class CubeSlide_4(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def compute_distance(self, pos1, pos2):
        return np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))

    def select_action(self, state):
        blue_cube_position = state[:3]
        red_cube_position = state[3:6]
        left_end_effector_pose = state[6:13]
        left_end_effector_vel = state[13:19]
        left_gripper = state[19]
        left_gripper_vel = state[20]
        objects_state = state[21:]

        # Compute the distance between the end effector and the blue cube
        dist_effector_blue = self.compute_distance(blue_cube_position, left_end_effector_pose[:3])

        # compute the distance between the two cubes
        dist_blue_red = self.compute_distance(blue_cube_position, red_cube_position)

        # If the end effector is too far away from the blue cube, the task is failed
        if dist_effector_blue > 0.1:
            print("Task failed: The end effector is too far away from the blue cube.")
            return None

        # If the blue cube is close enough to the red cube, the task is solved
        if dist_blue_red < 0.04:
            print("Task solved: The blue cube is close to the red cube.")
            return None

        # Else, move the gripper towards the blue cube and push it to the red cube
        direction = np.array(red_cube_position) - np.array(blue_cube_position)
        direction /= np.linalg.norm(direction) # normalize

        # update end effector velocity to move towards the blue cube
        action_end_effector_vel = left_end_effector_vel + direction.tolist()

        # close the gripper
        action_gripper_vel = -1  # close

        action = np.concatenate([action_end_effector_vel, [action_gripper_vel]])

        return action.tolist()



import numpy as np

class CubeSlide_5(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_position = [0.5331053387889144, -0.04764524062526641, 0.09397886814957689]
        self.red_cube_position = [0.6082525320598264, 0.06880255016518819, 0.09776131752240294]
        self.target_distance = 0.04
        self.close_gripper_value = -1
        self.open_gripper_value = 1
        self.end_effector_limit = 0.1

    def get_distance(self, position1, position2):
        return np.sqrt(np.sum((np.array(position1) - np.array(position2))**2))

    def get_direction(self, position1, position2):
        return (np.array(position2) - np.array(position1)) / self.get_distance(position1, position2)
    
    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        objects_state = state[14:]
        current_position = left_end_effector_pose[:3]
        distance = self.get_distance(current_position, self.blue_cube_position)
        
        if distance > self.end_effector_limit:
            # If distance from blue cube is too high return fail status
            print('Failed. Distance from blue cube Exceeds limit')
            return [0]*7
        
        cube_distance = self.get_distance(self.blue_cube_position, self.red_cube_position)
        if cube_distance <= self.target_distance:
            # If task is complete stop moving.
            return [0]*7
        else:
            # Move towards the blue cube and push it
            move_direction = self.get_direction(current_position, self.blue_cube_position)
            action = np.concatenate((move_direction, [0]*3, [self.close_gripper_value]))
            return action.tolist()



import numpy as np

class CubeSlide_6(CodePolicyLeft): 
    def __init__(self):
        super().__init__()
        self.blue_cube_pos = np.array([0.5331053387889144, -0.04764524062526641, 0.09397886814957689])
        self.red_cube_pos = np.array([0.6082525320598264, 0.06880255016518819, 0.09776131752240294])
        
    def select_action(self, state):
        left_end_effector_pose = np.array(state[:6])
        left_end_effector_vel = np.array(state[6:12])
        left_gripper = state[12]
        left_gripper_vel = state[13]
        objects_state = state[14:]
        
        effector_to_blue_dist = np.linalg.norm(self.blue_cube_pos - left_end_effector_pose[:3])
        blue_to_red_dist = np.linalg.norm(self.red_cube_pos - self.blue_cube_pos)
        
        if blue_to_red_dist < 0.04:
            return [0.0] * 7  # Task solved
        elif effector_to_blue_dist > 0.1:
            return [-1.0] * 7  # Task failed
        else:
            direction = (self.blue_cube_pos - left_end_effector_pose[:3]) / effector_to_blue_dist
            action = np.concatenate([direction, [0, 0, 0], [0]])  # Move towards the blue cube
            return action.tolist()



import numpy as np

class CubeSlide_7(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_pos = np.array([0.5331053387889144, -0.04764524062526641, 0.09397886814957689])
        self.red_cube_pos = np.array([0.6082525320598264, 0.06880255016518819, 0.09776131752240294])
        self.epsilon_distance = 0.04

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        objects_state = state[14:]

        end_effector_cube_distance = np.linalg.norm(self.blue_cube_pos - left_end_effector_pose[:3]) 

        if end_effector_cube_distance > 0.1:
            print("Task failed - Too far from the blue cube")
            action = [0.0] * 7
            return action

        cube_distance = np.linalg.norm(self.blue_cube_pos - self.red_cube_pos)

        if cube_distance < self.epsilon_distance:
            print("Task solved - Cubes are close enough")
            action = [0.0] * 7
            return action

        vel_direction_towards_blue = (self.blue_cube_pos - left_end_effector_pose[:3]) / end_effector_cube_distance
        vel_pointing_to_blue = 0.1 * vel_direction_towards_blue 

        gripper_closing_action = -1 if left_gripper > 0.5 else 0.1

        action = list(vel_pointing_to_blue) + [0, 0] + [gripper_closing_action] 
        return action



import numpy as np

class CubeSlide_8(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_pos = np.array([0.5331053387889144, -0.04764524062526641, 0.09397886814957689])
        self.red_cube_pos = np.array([0.6082525320598264, 0.06880255016518819, 0.09776131752240294])
        
    def select_action(self, state):
        left_end_effector_pose = np.array(state[:6])
        left_end_effector_vel = np.array(state[6:12])
        left_gripper = state[12]
        left_gripper_vel = state[13]
        objects_state = state[14:]
        
        # Calculate distance among all objects
        distance_blue_cube = np.linalg.norm(self.blue_cube_pos - left_end_effector_pose[:3])
        distance_red_cube = np.linalg.norm(self.red_cube_pos - left_end_effector_pose[:3])
        distance_cubes = np.linalg.norm(self.red_cube_pos - self.blue_cube_pos)
        
        # If the distance of the end effector from the blue cube is more than 0.1m the task failed
        if distance_blue_cube > 0.1:
            print("Task failed")
            return [0.0] * 7
        
        # If the distance between the cubes is less than 0.04 the task has been solved
        if distance_cubes < 0.04:
            print("Task solved")
            return [0.0] * 7
        
        if self.blue_cube_pos[0] < self.red_cube_pos[0]:
            action = [0.0]*6 + [1.0] # pushing the blue cube to the red cube
        else:
            action = [0.0]*6 + [-1.0] # pulling the blue cube to red cube

        return action



import numpy as np

def distance_3d(self, point1, point2):
    """Compute the Euclidean distance between two points in 3D space."""
    return np.sqrt(
        (point2[0] - point1[0]) ** 2
        + (point2[1] - point1[1]) ** 2
        + (point2[2] - point1[2]) ** 2
    )



import numpy as np

class VialInsertion_0(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        #input (state vector includes): 
        #left_end_effector_pose, left_end_effector_vel, left_gripper, left_gripper_vel, objects_state

        #desired pose for end effector to reach the carrier
        desired_pose = np.array([0.55, 0, 0.0625, 0, 0, 0])

        #initialization
        action = [0.0]*14

        # calculate error in position
        err_pose = desired_pose - state[:6]

        # simple proportional control to minimize the pose error
        action[:6] = 0.5 * err_pose

        # check if vial is at the correct height or below, set the gripper to hold it, if not, adjust it.
        if state[2]<=0.06 and np.linalg.norm(state[:2] - np.array([0.485, -0.025]))<=0.025:
            action[6:8] = [0, 0]  # no velocity change
            action[8] = 0  # keep gripper closed
        else:
            action[6:8] = [0, -0.2] # descend in -ve z direction.
            action[8] = -1  # keep adjusting gripper to get a firm grip

        # No movement for right end effector and right gripper
        action[9:14] = [0] * 5

        # Makes sure gripper does not exceed velocity limits
        action[7] = np.clip(action[7], -1, 1)  

        return action



import numpy as np
from scipy.spatial import distance

class VialInsertion_1(CodePolicyLeft):

    def __init__(self):
        super().__init__()

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] # <0 is for closing, > 0 for opening, must be in [-1, 1]
        objects_state = state[14:] # If empty then there are no objects in sim

        vial_position = np.array([0.485, -0.025, 0.125])
        vial_carrier_position = np.array([0.55, 0, 0.0025])

        # vial insertion policy implementation
        if left_end_effector_pose[2] > 0.06:
            # continue to hold the vial and lower it
            action = [0,0,-0.1,0,0,0, 0]
        elif distance.euclidean(left_end_effector_pose[:2], vial_position[:2]) > 0.025:
            # if the vial's distance in the x-y plane from its loading position is more than 0.025m, task failed
            action = [0.0] * 7
        elif left_gripper <= 0:
            # if the fingers lose contact with the vial, task failed
            action = [0.0] * 7
        else:
            # task solved
            action = [0.0] * 7

        return action



class VialInsertion_2(CodePolicyLeft):
    def __init__(self, target_height=0.06, target_gripper_pose=[0.485, -0.025, 0.125], max_xy_plane_distance=0.025):
        super().__init__()
        self.target_height = target_height
        self.target_gripper_pose = target_gripper_pose
        self.max_xy_plane_distance = max_xy_plane_distance

    def calculate_distance(self, x, y):
        return ((x[0] - y[0])**2 + (x[1] - y[1])**2)**0.5

    def select_action(self, state):
        left_end_effector_pose = state[:3]
        left_end_effector_vel = state[3:6]
        left_gripper = state[6]
        left_gripper_vel = state[7]

        # Initialize the action to keep the robot still
        action = [0.0]*7 

        # If the end effector's height is more than target height, move down
        if left_end_effector_pose[2] > self.target_height:
            action[2] = -1.0

        # If the end effector's position strays too far in the xy plane, realign it
        xy_distance = self.calculate_distance(left_end_effector_pose[:2], self.target_gripper_pose[:2])
        if xy_distance > self.max_xy_plane_distance:
            # Action for moving the end effector in x and y planes to realign with target
            action[0] = -1 if left_end_effector_pose[0] > self.target_gripper_pose[0] else 1
            action[1] = -1 if left_end_effector_pose[1] > self.target_gripper_pose[1] else 1

        # Keep the gripper closed
        if left_gripper_vel > 0:
            action[4] = -1

        return action



class VialInsertion_3(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        # If the action involves only left arm, set right_end_effector_vel & right_gripper_vel to 0
        right_end_effector_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        right_gripper_vel = 0.0

        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        objects_state = state[14:]

        vial_pos = objects_state[0:3]
        vial_carrier_pos = objects_state[7:10]

        # Desired position is the center of the vial carrier
        desired_position = [vial_carrier_pos[0], vial_carrier_pos[1], 0.06]

        # Calculate the distance between vial and vial carrier
        distance = [a - b for a, b in zip(desired_position, vial_pos)]

        # Calculate the velocity for left_end_effector_pose in the x, y, z directions
        left_end_effector_vel = [0.1*dx for dx in distance]

        # If the vial's height is below 0.06m but above 0.025m, start closing the gripper
        if 0.025 < desired_position[2] - vial_pos[2] < 0.06:
            left_gripper_vel = -0.1

        # If the vial does not meet the required height and x-y plane distance the task failed
        # we can keep the robot still, or reset its action
        if abs(desired_position[2] - vial_pos[2]) > 0.025:
            left_end_effector_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            left_gripper_vel = 0.0

        action = list(left_end_effector_vel) + [left_gripper_vel] + list(right_end_effector_vel) + [right_gripper_vel]

        return action



import numpy as np

class VialInsertion_4(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.vial_position = np.array([0.485, -0.025, 0.125])
        self.carrier_position = np.array([0.55, 0, 0.0025])
        self.success_height = 0.06
        self.failure_distance = 0.025

    def compute_euclidean_distance(self, pos1, pos2):
        return np.sqrt(np.sum((pos1-pos2)**2))

    def get_vial_state(self, state):
        return state[:3], state[3:7] # assuming the vial's position and orientation are the first 7 elements of object state

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        objects_state = state[14:]

        vial_position, vial_orientation = self.get_vial_state(objects_state)

        if self.compute_euclidean_distance(vial_position, self.vial_position) > self.failure_distance:
            # task failed
            return [0]*14 # Fail-safe, return zeros to stop any further action 

        elif vial_position[2] <= self.success_height and left_gripper > 0:
            # task succeeded, hold the vial
            return np.concatenate((left_end_effector_vel, [0], [0], [left_gripper_vel], [0]*7))

        else:
            # continue lowering the vial
            target_pos = self.carrier_position
            action = np.concatenate((target_pos - left_end_effector_pose[:3], [0]*4, [left_gripper_vel], [0]*7))
            return action




class VialInsertion_5(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        vial_position = [0.485, -0.025, 0.125]
        vial_carrier_position = [0.55, 0, 0.0025]
        vial_carrier_offset = 0.06 
        dist_xy = 0.025

        left_end_effector_pose = state[:6] 
        left_end_effector_vel = state[6:12] 
        left_gripper = state[12] 
        left_gripper_vel = state[13]
        right_end_effector_vel = [0, 0, 0, 0, 0, 0] 
        right_gripper_vel = 0 

        action = left_end_effector_vel + left_gripper_vel + right_end_effector_vel + [right_gripper_vel]

        vial_dist = ((vial_position[0] - left_end_effector_pose[0])**2 + (vial_position[1] - left_end_effector_pose[1])**2)**0.5
        if vial_dist > dist_xy:
            print("Task failed due to vial's distance from its loading position being more than 0.025m.")
            return action

        if left_end_effector_pose[2] <= vial_carrier_position[2] + vial_carrier_offset and left_gripper > 0 :
            print("Task completed with the vial still being touched by both gripper fingers and at or below 0.06m height")
            return action

        elif left_gripper <= 0:
            print("Task failed due to the fingers losing contact with the vial.")
            return action

        else:
            print("Task in progress.")
            action = [0, 0, -1, 0, 0, 0, 1] + right_end_effector_vel + [right_gripper_vel]
            return action



import numpy as np

class VialInsertion_6(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] # <0 is for closing, > 0 for opening, must be in [-1, 1]
        objects_state = state[14:] # If empty then there are no objects in sim

        vial_pos = np.array([0.485, -0.025, 0.125]) # vial initial position
        vial_carrier_pos = np.array([0.55, 0, 0.0025]) # vial carrier position
        current_pos = left_end_effector_pose[:3]

        # Compute the remaining distance in the x-y plane and in the z direction
        xy_distance = np.linalg.norm(current_pos[:2] - vial_pos[:2])
        z_distance = current_pos[2] - vial_pos[2]

        # If the vial is not inside the vial carrier and the gripper is holding the vial, move down
        if z_distance > 0.06 and left_gripper == 1 and xy_distance < 0.025:
            action = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0]) # Move down

        # If the vial is at 0.06m height or below, consider the task solved
        elif z_distance <= 0.06 and left_gripper == 1 and xy_distance < 0.025:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Don't move

        # If the fingers lost contact with the vial, or the vial's distance in the x-y plane from its loading position
        # is more than 0.025m, consider the task failed
        elif left_gripper == 0 or xy_distance > 0.025:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Don't move
            print("Task failed")

        return action



class VialInsertion_7(CodePolicyLeft):
    VIAL_POS = [0.485, -0.025, 0.125]
    VIAL_CARRIER_POS = [0.55, 0, 0.0025]
    MAX_DIST = 0.025
    MAX_HEIGHT = 0.06

    def select_action(self, state):
        super().select_action(state)

        # Get the current position and velocities
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]

        # Compute the distance in X-Y plane from the gripper to the vial and carrier 
        dist_vial = ((left_end_effector_pose[0] - self.VIAL_POS[0])**2 + (left_end_effector_pose[1] - self.VIAL_POS[1])**2)**0.5
        dist_carrier = ((left_end_effector_pose[0] - self.VIAL_CARRIER_POS[0])**2 + (left_end_effector_pose[1] - self.VIAL_CARRIER_POS[1])**2)**0.5

        # Task failed
        if dist_vial > self.MAX_DIST or left_end_effector_pose[2] < 0.06:
            return [0.0] * 7

        # Task solved
        if dist_carrier < self.MAX_DIST and left_end_effector_pose[2] <= self.MAX_HEIGHT:
            return [0.0] * 7

        # Lower the gripper with a constant speed
        action = [0.0]*6 + [-0.01]

        return action



class TaskHelper:
    @staticmethod
    def calculate_distance(point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    @staticmethod
    def is_vial_in_range(vial_position, carrier_position, range_limit):
        return TaskHelper.calculate_distance(vial_position, carrier_position) <= range_limit



class VialInsertion_9(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        left_end_effector_pose = state[:6]  # x, y, z positions and euler angles
        left_end_effector_vel = state[6:12]  # velocities
        left_gripper = state[12]
        left_gripper_vel = state[13]  # Velocity of opening or closing
        objects_state = state[14:]

        vial_position = objects_state[:3]
        vial_touch = any(map(lambda touch: touch > 0, objects_state[3:5]))

        action = left_end_effector_vel + [left_gripper_vel]

        # Check vial's distance from loading position
        vial_dist = ((vial_position[0] - 0.485)**2 + (vial_position[1] -  (-0.025))**2) ** 0.5

        # Check if the vial is safely gripped
        if not vial_touch:
            print("Task failed. Fingers lost contact with the vial.")
            action[6] = -1  # Close the gripper
        elif vial_dist > 0.025:
            print(f"Task failed. Vial dist={vial_dist} > 0.025")
            return action
        elif vial_position[2] > 0.06:
            action[2] = -0.01  # Lower the vial
            print("Lowering the vial...")
        else:
            print("Task succeeded.")
            action = [0.0] * 7

        return action



class CentrifugeInsertion_0(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        
        goal_pose = [0.55, 0.154, 0.08]
        desired_distance = 0.04
        vial_position = state[:3]

        distance = ((vial_position[0]-goal_pose[0])**2 + (vial_position[1]-goal_pose[1])**2)**0.5

        if vial_position[2] < goal_pose[2]:
            print("Failed - Vial moved up")
            self.failed = True 

        elif distance > desired_distance:
            print("Failed - The vial's distance is more than 0.04m from its loading position")
            self.failed = True

        else:
            # Move end effector down slowly
            left_end_effector_vel = [0, 0, -0.01] + [0]*3 # [0]*3 for the orientation velocities
            left_gripper_vel = 0 # Keep gripper closed

            right_end_effector_vel = [0]*6
            right_gripper_vel = 0 # Keep gripper closed

            action = left_end_effector_vel + [left_gripper_vel] + right_end_effector_vel + [right_gripper_vel]
        
        if vial_position[2] <= goal_pose[2] and not self.failed:
            print("Succeeded task - The vial is still touched by both gripper fingers, and the vial is at 0.08m height or below.")

        return action
 


class CentrifugeInsertion_1(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        # Parse state
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] 
        objects_state = state[14:] 

        vial_position = objects_state[:3]
        centrifuge_position = objects_state[3:6]

        # Calculate distance metrics
        vertical_dist = vial_position[2] - centrifuge_position[2] 
        horizontal_dist = ((vial_position[0] - centrifuge_position[0]) ** 2 +
                           (vial_position[1] - centrifuge_position[1]) ** 2) ** 0.5

        # If vial is not yet at the centrifuge height and hasn't moved off horizontally, lower the vial
        if vertical_dist > 0.08 and horizontal_dist < 0.04:
            action = [-0.01, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0]  # Only moves in z-direction 

        # If vial is at correct height but moved off horizontally, correct the horizontal position
        elif vertical_dist <= 0.08 and horizontal_dist >= 0.04:
            if (vial_position[0] - centrifuge_position[0]) > 0:
                action = [0.0, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0]  # Moves in y-direction
            else:
                action = [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]  # Moves in y-direction

        # If vial moved off in both directions, correct the vertical position first
        elif vertical_dist > 0.08 and horizontal_dist >= 0.04:
            action = [-0.01, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0]  # Only moves in z-direction 

        # If vial is in correct position and hasn't lost contact, do not move
        else:
            action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return action



import numpy as np
from itertools import islice

class CentrifugeInsertion_2(CodePolicyLeft):
    def __init__(self):
        super(CentrifugeInsertion_2, self).__init__()

    def select_action(self, state):
        vial_position = np.array([0.57, 0.16, 0.18])
        centrifuge_position = np.array([0.55, 0.154, 0.0025])

        left_end_effector_pose = np.array(state[:6])
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] 

        objects_state = state[14:] 

        desired_ee_pose = np.array(vial_position)
        desired_ee_pose[2] = 0.08

        desired_ee_vel = (desired_ee_pose - left_end_effector_pose[:3]) * 0.1

        action = np.zeros(14)
        
        if np.linalg.norm(desired_ee_pose[:2] - left_end_effector_pose[:2]) > 0.04:
            print("Task failed: XY-distance from initial position is more than 0.04m")
            return action

        elif left_end_effector_pose[2] < desired_ee_pose[2] and left_gripper < 0.1:
            action[:6] = desired_ee_vel
            action[6] = 0.0
        elif left_gripper > 0.0:
            action[6] = -0.1 

        return action



import numpy as np

class CentrifugeInsertion_3(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.goal_height = 0.08
        self.load_position = np.array([0.57, 0.16, 0.18])
        self.distance_tolerance = 0.04
        self.fingers_touching_vial = True
        self.vial_in_place = False

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper_vel = state[13]
        objects_state = state[14:]
        
        current_vial_position = objects_state[:3]

        # Calculate distance in x-y plane from loading position
        distance = np.linalg.norm(current_vial_position[:2] - self.load_position[:2])

        # Check if the vial is in place
        if current_vial_position[2] <= self.goal_height and distance <= self.distance_tolerance:
            self.vial_in_place = True

        if not self.vial_in_place:
            # Create default action
            action = [0.0] * 7
            
            # If the effector is too high, move it down
            if left_end_effector_pose[2] > self.goal_height:
                action[2] = -1.0

            # If the vial is not in position, adjust in x-y plane
            if distance > self.distance_tolerance:
              dx = self.load_position[0] - current_vial_position[0]
              dy = self.load_position[1] - current_vial_position[1]
              action[0] = dx
              action[1] = dy
              
        elif self.fingers_touching_vial:
            # Keep the effector still, but keep the gripper closed
            action = [0.0] * 7
            action[6] = left_gripper_vel =  -1.0
        else:
            # Fail the task if the fingers lose contact with the vial
            raise Exception('Task failed: Lost contact with the vial')

        # Setting right arm velocities to zero
        action.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return action



import numpy as np

class CentrifugeInsertion_4(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.vial_position = np.array([0.57, 0.16, 0.18]) # initial vial position
        self.centrifuge_position = np.array([0.55, 0.154, 0.0025]) # centrifuge position
        self.vial_inserted = False

    def select_action(self, state):
        # define the state variables
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        objects_state = state[14:]

        if not self.vial_inserted:
            # if vial is not inserted, move down towards the centrifuge
            action = np.array([0, 0, -0.1, 0, 0, 0, 0]) # slowly move down in z direction
        else:
            # if vial is inserted, keep the robot still
            action = np.array([0, 0, 0, 0, 0, 0, 0]) # stay still

        if left_end_effector_pose[2] <= self.centrifuge_position[2] + 0.08: # when vial reaches centrifuge
            self.vial_inserted = True

        # keep right end effector and gripper velocity to zero
        right_end_effector_vel = [0.0] * 6
        right_gripper_vel = 0.0

        # define the action to be returned
        action = np.concatenate((action, right_end_effector_vel, [right_gripper_vel]))

        return action




class CentrifugeInsertion_5(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        # End effector's and objects' states
        left_end_effector_pose = state[:6]
        left_gripper = state[12]
        vial_position = state[14:17]
        centrifuge_position = state[20:23]

        # Calculate the distances
        vertical_distance_to_centrifuge = left_end_effector_pose[2] - centrifuge_position[2]
        horizontal_distance_to_centrifuge = ((left_end_effector_pose[0]-centrifuge_position[0])**2
                                            +(left_end_effector_pose[1]-centrifuge_position[1])**2)**0.5

        # Prepare the action vector (vx, vy, vz, roll, pitch, yaw, gripper_vel)
        action = [0.0] * 7

        # Control the gripper to hold the vial tight
        if left_gripper > 0.02:
            action[6] = -1.0 # close the gripper
        else:
            action[6] = 0.0  # hold still

        # Control the end-effector to insert the vial by lowering it
        if vertical_distance_to_centrifuge > 0.08 and horizontal_distance_to_centrifuge < 0.04:
            action[2] = -0.02 # lower the end-effector
        elif vertical_distance_to_centrifuge <= 0.08 and horizontal_distance_to_centrifuge >= 0.04:
            action[2] = 0.02 # lift the end-effector
        else:
            action[2] = 0 # hold the height constant

        # Try to maintain the initial position in the x-y plane
        if horizontal_distance_to_centrifuge > 0.04:
            action[0] = -(left_end_effector_pose[0] - vial_position[0]) # control in x-axis
            action[1] = -(left_end_effector_pose[1] - vial_position[1]) # control in y-axis

        return action



class CentrifugeInsertion_6(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.vial_start_pos = [0.57, 0.16, 0.18]
        self.centrifuge_pos = [0.55, 0.154, 0.0025]
        self.holding_vial = True

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        objects_state = state[14:]

        action = [0] * 14
        vial_pos = self.vial_start_pos

        if self.holding_vial:
            if left_end_effector_pose[2] > self.centrifuge_pos[2] + 0.08:
                # move down towards the centrifuge
                action[2] = -0.01
            else:
                # adjust x, y position to place vial exactly over the centrifuge
                x_diff = self.centrifuge_pos[0] - left_end_effector_pose[0]
                y_diff = self.centrifuge_pos[1] - left_end_effector_pose[1]
                action[0], action[1] = x_diff, y_diff

                # check if vial is properly placed
                if abs(x_diff) <= 0.04 and abs(y_diff) <= 0.04:
                    self.holding_vial = False

            if left_gripper <= 0:
                # need to hold the vial tighter
                action[6] = -0.1
            else:
                action[6] = 0
        else:
            action[6] = 0.1  # Open the gripper after placing vial

        return action



import numpy as np

class CentrifugeInsertion_7(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        # Goal Positions
        vial_pos = [0.57, 0.16, 0.18]
        centrifuge_pos = [0.55, 0.154, 0.0025]

        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        objects_state = state[14:]

        #---- Step 1: Lowering the vial to the centrifuge ----#
        # Compute positional error
        pos_error = np.array(vial_pos) - np.array(centrifuge_pos)
        
        # Compute control command --> proportional (P) controller
        action = -0.1 * pos_error 

        #---- Step 2: Ensure gripper stays closed ----#
        action[6] = 0.0 if left_gripper > 0 else -1.0

        return np.concatenate((action, [0.0]*7))    # keep right_end_effector_vel and right_gripper_vel to zero



import numpy as np

class CentrifugeInsertion_8(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        # Extract required state attributes
        left_end_effector_pose = state[:6]
        objects_state = state[14:]
        vial_pos = objects_state[:3]
        centrifuge_pos = objects_state[7:10]

        # Calculate the required movement in x, y, z directions
        dx = centrifuge_pos[0] - vial_pos[0]
        dy = centrifuge_pos[1] - vial_pos[1]
        dz = centrifuge_pos[2] - vial_pos[2]

        # Ensure the vial only moves straight down and not in x and y directions
        if abs(dx) > 0.04 or abs(dy) > 0.04:
            return [0.0]*7

        # Proceed if vial has not reached the required height
        if dz > 0.08:
            down_speed = -0.01 
            action = [0, 0, down_speed, 0.0, 0.0, 0.0, 0.0]  

        # Stop if vial is at required height
        elif dz <= 0.08:
            action = [0.0] * 7 

        # Move the left-end effector while keeping the right end effector and gripper velocity to zero
        return action + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]



class CentrifugeInsertion_9(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        
        # Vial position from the object state
        vial_pos = state[14:17]
        
        # Get the x-y plane distance
        xy_distance = ((vial_pos[0] - 0.57)**2 + (vial_pos[1] - 0.16)**2)**0.5
        
        # If no contact with the vial then fail
        if left_gripper == 0.0: 
            raise Exception("Lost contact with the vial.")
        
        # If far away in the x-y plane then fail
        if xy_distance > 0.04:
            raise Exception("Lost alignment with the loading position.")
        
        # If the vial has reached the required height then keep the robot's hand still
        if vial_pos[2] <= 0.08:
            action = [0.0] * 7
        else:
            # Else move the robot hand downwards, while maintaining constant horizontal velocities
            action = [left_end_effector_vel[0], left_end_effector_vel[1], -1, 0, 0, 0, left_gripper_vel]
        
        return action



import numpy as np

class CubeStack_0(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_pos_init = [0.492, 0.0832282365, 0.146841436]
        self.red_cube_pos_init = [0.58833315, 0.06057621, 0.09636923]

    def select_action(self, state):
        blue_cube_state = state[14:21] #assuming the state of the blue cube starts at index 14
        red_cube_state = state[21:28] #assuming the state of the red cube starts at index 21
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] 

        blue_cube_pos = blue_cube_state[:3]
        red_cube_pos = red_cube_state[:3]

        # Using a simple proportional control to move the blue cube over the red cube
        obj_disp = np.array(self.red_cube_pos_init) - np.array(blue_cube_pos)
        obj_disp[2] = self.blue_cube_pos_init[2] - blue_cube_pos[2] # The z control is different

        err_disp = np.array(red_cube_pos) - np.array(self.red_cube_pos_init)
        blue_red_disp = np.array(blue_cube_pos) - np.array(red_cube_pos)

        # Distance in x, y, |Difference in z|, |Diff of position with initial state|
        conditions = [np.linalg.norm(blue_red_disp[:2]), abs(blue_red_disp[2]), np.linalg.norm(err_disp)]

        # If the conditions are met, the task is completed
        if (conditions[0] < 0.005) and (conditions[1] <= 0.0255) and (conditions[2] < 0.005):
            return [0.0] * 7

        if conditions[2] >= 0.005:
            return [0.0] * 7

        action = obj_disp * 0.1 # Simple Proportional control
        action = np.append(action, 0) # adding zero velocity for the gripper, to keep it in contact

        return action.tolist()



class CubeStack_1(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_initial_position = [0.492, 0.0832282365, 0.146841436]
        self.red_cube_initial_position = [0.58833315, 0.06057621, 0.09636923]
        self.target_position = [0.58833315, 0.06057621, 0.09636923 + 0.025]
    
    def select_action(self, state): 
        # Get the current position of blue and red cube
        left_end_effector_pose = state[:6]
        blue_cube_position = state[14:17]
        red_cube_position = state[24:27]
        
        # Get the velocities
        left_end_effector_vel = state[6:12]
        left_gripper_vel = state[13] 

        # Condition to check if blue cube is at the target position
        is_blue_on_target = abs(blue_cube_position[0] - self.target_position[0]) < 0.005 and \
                            abs(blue_cube_position[1] - self.target_position[1]) < 0.005 and \
                            abs(blue_cube_position[2] - self.target_position[2]) <= 0.0255
        
        # Condition to check if red cube is at the initial position
        is_red_at_initial = abs(red_cube_position[0] - self.red_cube_initial_position[0]) < 0.005 and \
                            abs(red_cube_position[1] - self.red_cube_initial_position[1]) < 0.005 and \
                            abs(red_cube_position[2] - self.red_cube_initial_position[2]) < 0.005

        # Conditions to check if the gripper has lost contact with blue cube or 
        # the red cube has moved from its initial position
        has_gripper_lost_contact = left_gripper_vel > 0 or \
                                   abs(blue_cube_position[2] - left_end_effector_pose[2]) > 0.03
        has_red_cube_moved = not is_red_at_initial

        # If the task is failed 
        if has_gripper_lost_contact or has_red_cube_moved:
            print("Task Failed!")
            action = [0.0] * 7

        # If the blue cube is on target and red cube is at initial position
        elif is_blue_on_target and is_red_at_initial:
            print("Task Completed")
            action = [0.0] * 7

        else: 
            # Determine velocities to move the blue cube towards the target position
            x_vel = (self.target_position[0] - blue_cube_position[0]) * 0.01
            y_vel = (self.target_position[1] - blue_cube_position[1]) * 0.01
            z_vel = (self.target_position[2] - blue_cube_position[2]) * 0.01
            action = [x_vel, y_vel, z_vel, 0, 0, 0, 0]
        
        # return action
        return action




import numpy as np

class CubeStack_2(CodePolicyLeft):

    def __init__(self):
        super().__init__()

        # initial cube positions
        self.blue_cube_init_pos = np.array([0.492, 0.0832282365, 0.146841436])
        self.red_cube_init_pos = np.array([0.58833315, 0.06057621, 0.09636923])

        # end positions
        self.blue_cube_end_pos = self.red_cube_init_pos.copy()
        self.blue_cube_end_pos[2] += 0.025  # height difference between two cubes
        self.red_cube_end_pos = self.red_cube_init_pos.copy()

        # action velocities
        self.position_velocity = 0.01
        self.orientation_velocity = 0.01
        self.gripper_velocity = 0

    def select_action(self, state):
        blue_cube_pos = state[:3]  # replace this line to get blue cube's position
        blue_cube_ori = state[3:6]  # replace this line to get blue cube's orientation
        
        red_cube_pos = state[6:9]  # replace this line to get red cube's position
        red_cube_ori = state[9:12]  # replace this line to get red cube's orientation

        # calculate desired velocities to move blue cube over the red cube
        position_vel = np.sign(self.blue_cube_end_pos - blue_cube_pos) * self.position_velocity
        orientation_vel = np.sign([0, 0, 0] - blue_cube_ori) * self.orientation_velocity

        action = np.hstack((position_vel, orientation_vel, self.gripper_velocity))

        return action




import math
import numpy as np

class CubeStack_3(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        LOADING_RED_CUBE_POS = np.array([0.58833315, 0.06057621, 0.09636923])
        tolerance = 0.005
        tolerance_height = 0.0255
        blue_cube_pose = state[14:20] # position and orientation of the blue cube
        red_cube_pose = state[20:26] # position and orientation of the red cube

        action = [0.0] * 7

        # Compute the Euclidean distance in the x-y plane between the blue and red cubes
        xy_distance = np.linalg.norm(blue_cube_pose[0:2] - red_cube_pose[0:2])
        # Compute the absolute difference in height (z-value) between the blue and red cubes
        height_difference = abs(blue_cube_pose[2] - red_cube_pose[2])

        # Check if the blue cube is close to the red cube on the x-y plane and with the correct height, 
        # and if the red cube is roughly in its loading position
        if (xy_distance < tolerance) and (height_difference <= tolerance_height) and \
                (np.linalg.norm(LOADING_RED_CUBE_POS - red_cube_pose[0:3]) < tolerance):
            action[:3] = 0.0
            action[3:6] = 0.0
            action[6] = 0.0
        else:
            action[:3] = -blue_cube_pose[:3] + red_cube_pose[:3] + np.array([0, 0, 0.025])
            action[3:6] = [-math.pi/2, 0, 0]
            action[6] = max(-1.0, min(1.0, blue_cube_pose[3] - 0.5))

        return action



import numpy as np

class CubeStack_4(CodePolicyLeft):
    def __init__(self, blue_cube_init, red_cube_init):
        super().__init__()
        self.blue_cube_init = np.array(blue_cube_init[:3])
        self.red_cube_init = np.array(red_cube_init[:3])

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_gripper = state[12]
        objects_state = state[14:]  

        # split object states into separate variables
        blue_cube_state = objects_state[:7]
        red_cube_state = objects_state[7:14]

        # Get the cube positions  
        blue_cube_pos = np.array(blue_cube_state[:3])
        red_cube_pos = np.array(red_cube_state[:3])

        # Compute the desired position
        desired_pos = red_cube_pos.copy()
        desired_pos[-1] += 0.026  # 0.0255 (cube height ) + 0.0005 (some tolerance)

        # Compute the error
        pos_error = desired_pos - blue_cube_pos

        # check all conditions for task completion and failure
        xy_distance_blue_red = np.linalg.norm(blue_cube_pos[:2] - red_cube_pos[:2])
        height_difference_blue_red = np.abs(blue_cube_pos[2] - red_cube_pos[2])
        red_cube_displacement = np.linalg.norm(red_cube_pos - self.red_cube_init)
       
        if left_gripper < 0.5 or xy_distance_blue_red >= 0.005 or red_cube_displacement >= 0.005:
            return [0.0]*7, True, False  # task failure

        elif xy_distance_blue_red < 0.005 and height_difference_blue_red <= 0.0255:
            return [0.0]*7, True, True  # task success
        
        else:
            # Use a simple P controller to compute the velocities
            kp = 1.0
            action = kp * pos_error
            return np.hstack([action, 0.0]), False, False  # keep holding the blue cube



import numpy as np

class CubeStack_5(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_init_pos = np.array([0.492, 0.0832282365, 0.146841436]) 
        self.red_cube_init_pos = np.array([0.58833315, 0.06057621, 0.09636923]) 
        self.right_end_effector_vel = np.array([0.0, 0.0, 0.0]) 
        self.right_gripper_vel = 0.0 
        
    def select_action(self, state):
        blue_cube = state[:6]
        blue_cube_pos = np.array(blue_cube[:3])
        red_cube = state[6:12]
        red_cube_pos = np.array(red_cube[:3])
        dist = np.linalg.norm(blue_cube_pos - red_cube_pos)
        
        if not self.cubes_are_close(blue_cube_pos, red_cube_pos):
            action = self.move_towards_red(blue_cube_end_effector_vel, red_cube)
        elif not self.cubes_are_combined(blue_cube_pos, red_cube_pos):
            action = self.move_on_top(blue_cube_end_effector_vel, red_cube)
        else:
            action = self.keep_hold(blue_cube_end_effector_vel)

        return action

    def cubes_are_close(self, blue_cube_pos, red_cube_pos):
        dist = np.linalg.norm(blue_cube_pos[:2] - red_cube_pos[:2])
        height_diff = abs(blue_cube_pos[2] - red_cube_pos[2])
        red_cube_dist = np.linalg.norm(red_cube_pos - self.red_cube_init_pos)
        return dist < 0.005 and height_diff <= 0.0255 and red_cube_dist < 0.005

    def move_towards_red(self, blue_cube_end_effector_vel, red_cube):
        action = (red_cube[:3] - blue_cube_end_effector_vel[:3]) * 0.1 
        action = np.append(action, np.array([0.0, 0.0, 0.0, 0.0]))
        return np.concatenate((action, np.array([0.0])), axis=0)

    def move_on_top(self, blue_cube_end_effector_vel, red_cube):
        action = np.array([0.0, 0.0, (red_cube[2] - blue_cube_end_effector_vel[2]) * 0.1])
        action = np.append(action, np.array([0.0, 0.0, 0.0, 0.0]))
        return np.concatenate((action, np.array([0.0])), axis=0)

    def keep_hold(self, blue_cube_end_effector_vel):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return action



import numpy as np

class CubeStack_6(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_pos = np.array([0.492, 0.0832282365, 0.146841436])
        self.red_cube_pos = np.array([0.58833315, 0.06057621, 0.09636923])
        self.loading_pos = self.red_cube_pos
        self.opening_pos = 1 # gripper opening position
        self.closing_pos = -1 # gripper closing position
        self.threshold = 0.005 # threshold distance, in meters
        self.height_diff = 0.0255 # height difference, in meters

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        objects_state = state[14:]

        # The suggestive action assumes that the blue cube is always at the first position in the objects_state
        blue_cube_pose = objects_state[:7] 
        red_cube_pose = objects_state[7:14]
        
        action = [0.0] * 7  # initialize action to keep bot still

        # store x,y,z position for each cube
        blue_cube_xy = blue_cube_pose[:2]
        red_cube_xy = red_cube_pose[:2]

        distance = np.sqrt(np.sum((blue_cube_xy - red_cube_xy)**2))
        
        if distance < self.threshold and np.abs(blue_cube_pose[2] - red_cube_pose[2]) <= self.height_diff and np.abs(self.red_cube_pos - self.loading_pos) < self.threshold:
            # Task is complete, keep bot still
            action = [0.0] * 7
        elif blue_cube_pose[13] != self.closing_pos or np.abs(red_cube_pose[:2] - self.loading_pos[:2]) >= self.threshold:
            # If blue cube is dropped or red cube is moved, task is failed
            print('Task Failed')
        else:
            # If task is not complete, move blue cube on top of the red cube
            action = blue_cube_xy - red_cube_xy
            
        return action



import numpy as np

class CubeStack_7(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_start_pos = np.array([0.492, 0.0832282365, 0.146841436])
        self.red_cube_pos = np.array([0.58833315, 0.06057621, 0.09636923])
        self.xy_tolerance = 0.005  # in meters
        self.z_tolerance = 0.0255  # in meters
        self.gripper_holding_force = -0.5 # force to hold the blue cube 

    def select_action(self, state):
        left_end_effector_pose = np.array(state[:3]) # excluding orientation 
        left_end_effector_vel = np.array(state[3:6])
        left_gripper = state[6]
        left_gripper_vel = state[7]
        objects_state = state[8:]
        
        # define a speed limit
        speed_limit = 0.1 # meters per second

        # calculate current distance to the target position
        target_position = self.red_cube_pos.copy()
        target_position[2] += 0.025 # putting the blue cube on top of the red cube
        distance = target_position - left_end_effector_pose 
        
        #dmaintain the grip on the blue cube
        if(left_gripper <= 0):
            left_gripper_vel = self.gripper_holding_force 
        else:
            left_gripper_vel = 0

        # check the distance condition and move the effector
        if np.abs(distance[0]) <= self.xy_tolerance and np.abs(distance[1]) <= self.xy_tolerance:
            left_end_effector_vel[0] = 0
            left_end_effector_vel[1] = 0
        else:           
            left_end_effector_vel[:2] = np.clip(distance[:2], -speed_limit, speed_limit)

        if np.abs(distance[2]) <= self.z_tolerance:
            left_end_effector_vel[2] = 0
        else:           
            left_end_effector_vel[2] = np.clip(distance[2], -speed_limit, speed_limit)

        return np.concatenate((left_end_effector_vel,left_gripper_vel))



import numpy as np

class CubeStack_8(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.blue_cube_init_position = np.array([0.492, 0.0832282365, 0.146841436])
        self.red_cube_init_position = np.array([0.58833315, 0.06057621, 0.09636923])
        self.task_failed = False

    def select_action(self, state):
        """
        Modify this method as per robot's state space
        """
        gripper_position = state[:3]
        left_end_effector_vel = state[6:9]
        left_gripper = state[12]
        left_gripper_vel = state[13]

        # Get blue and red cube state
        blue_cube_state = state[14:21]
        red_cube_state = state[21:28]

        blue_cube_pos = np.array(blue_cube_state[:3])
        red_cube_pos = np.array(red_cube_state[:3])

        objects_state = state[28:]  # If empty then there are no objects in sim

        # Check task failure conditions
        if np.linalg.norm(blue_cube_pos - gripper_position) > 0.005 or \
                np.linalg.norm(red_cube_pos - self.red_cube_init_position) > 0.005:
            self.task_failed = True

        # Strategy: Manipulate the blue cube towards the red cube
        target_pos = self.red_cube_init_position.copy()
        target_pos[2] += 0.0255  # Adjust for cube height

        direction_to_target = target_pos - gripper_position
        direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)

        action = np.zeros(14)  # Initialize action
        action[:3] = direction_to_target * 0.01  # Move towards the target

        # Reach to the target condition
        if np.linalg.norm(blue_cube_pos - target_pos) < 0.005 and \
                np.abs(blue_cube_pos[2] - target_pos[2]) <= 0.0255:
            action[:3] = 0  # Stop moving

        return action



import numpy as np

class CubeStack_9(CodePolicyLeft):
    def __init__(self):
        super().__init__()

        # Define initial position of blue and red cube
        self.blue_cube_pos = np.array([0.492, 0.0832282365, 0.146841436])
        self.red_cube_pos = np.array([0.58833315, 0.06057621, 0.09636923])

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        objects_state = state[14:]
        
        # Define blue and red cube state 
        blue_cube_state = objects_state[:7]
        red_cube_state = objects_state[7:14]
        
        # Check if blue_cube_state or red_cube_state empty, if so, return still action
        if blue_cube_state.size == 0 or red_cube_state.size == 0:
            return [0.0] * 7

        # Calculate the distance between the two cubes in the x-y plane
        xy_plane_dist = np.linalg.norm(blue_cube_state[:2] - red_cube_state[:2])

        # Calculate the absolute difference between the two cubes height
        height_diff = np.abs(blue_cube_state[2] - red_cube_state[2])

        # Calculate the distance of the red cube from its loading position.
        red_cube_loading_dist = np.linalg.norm(red_cube_state[:3] - self.red_cube_pos)

        # Task failed conditions
        if (left_gripper <= 0) or (red_cube_loading_dist >= 0.005) or (xy_plane_dist >= np.linalg.norm(self.blue_cube_pos[:2] - self.red_cube_pos[:2])):
            return [0.0] * 7

        # Task completed conditions
        if (xy_plane_dist < 0.005) and (height_diff <= 0.0255):
            return [0.0] * 7

        # Manipulate blue cube towards red cube
        action = red_cube_state[:3] - blue_cube_state[:3]
        action = np.append(action, [0.0, 0.0, 0.0])
        action = np.append(action, left_gripper)

        return action



class CubesTowerSlide_0(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.target_y = 0.1
        self.threshold = 0.01
        self.contact_distance = 0.05

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] 
        blue_cube_state, red_cube_state = state[14:], state[28:]

        blue_cube_pos = blue_cube_state[:3]
        red_cube_pos = red_cube_state[:3]

        objects_distance = left_end_effector_pos[:3] - red_cube_pos
        objects_distance_xy = objects_distance[:2]
        
        if np.linalg.norm(objects_distance_xy) > self.threshold:
            return [0.]*7  # fail task if distance between cubes is too large

        if np.linalg.norm(objects_distance) > self.contact_distance:
            return [0.]*7  # fail task if distance between end effector and red cube is too large
            
        if (left_end_effector_pose[1] <= self.target_y) or (left_gripper == 1):
            action = [0., -0.1, 0., 0., 0., 0., 0.]  # slide the red cube to the right by 0.1 meters
        else:
            action = [0.] * 6 + [-1.]  # let go of the blue cube

        return action        



import numpy as np
from scipy.spatial import distance

class CubesTowerSlide_1(CodePolicyLeft):
    def __init__(self):
        super().__init__()

        # Initial position of the red cube
        self.initial_red_cube_position = [0.588111651, 0.0601549914, 0.095612514]

        # Loading position of the cubes
        self.loading_position = [0.59, 0.061, 0.120946143]

        # Target position to slide the cubes towards
        self.target_position = [self.loading_position[0], self.loading_position[1] - 0.1, self.loading_position[2]]

        # Position and orientation indices
        self.red_cube_position_idx = [14, 15, 16]
        self.blue_cube_position_idx = [19, 20, 21]

        # Failure criteria
        self.max_cube_dist = 0.01
        self.max_end_effector_red_cube_dist = 0.05

    def select_action(self, state):
        # Get current position of the red cube
        red_cube_position = state[self.red_cube_position_idx]

        # Check distance between cubes
        blue_cube_position = state[self.blue_cube_position_idx]
        cube_dist = distance.euclidean(red_cube_position, blue_cube_position)

        if cube_dist > self.max_cube_dist:
            print("Task failed, x-y distance between cubes is > 0.01 meters.")
            return None

        # Check distance between the end effector and the red cube
        end_effector_red_cube_dist = distance.euclidean(red_cube_position, state[:3])
        if end_effector_red_cube_dist > self.max_end_effector_red_cube_dist:
            print("Task failed, distance between end effector and the red cube is > 0.05 meters.")
            return None

        # If cubes have reached the target position, stop moving
        if distance.euclidean(red_cube_position, self.target_position) < 0.01:
            print("Task solved, both cubes moved to the right of 0.1 meters from their loading position.")
            return [0.0] * 7

        left_gripper = state[12]
        if left_gripper > 0:
            # Let go of the blue cube
            left_gripper_vel = -1
        else:
            # Move towards the red cube
            action = np.array(red_cube_position) - np.array(self.target_position)
            action = np.append(action, 0)  # Gripper should remain stationary

            # Normalize the action to be in the range [-1, 1]
            action = np.clip(action, a_min=-1, a_max=1)
            
            return action.tolist()



import numpy as np

class CubesTowerSlide_2(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.target_offset = [-0.1, 0, 0]  # Desired offset in y axis
        self.pos_tolerance = 0.01  # after moving the cubes, ensure they are at least this close
        self.hold_cube = True  # Flag controlling cube holding status

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]

        # Blue cube and Red cube states
        blue_cube_state = state[14:20]
        red_cube_state = state[20:26]

        # Compute the offset in y-direction from the current position
        y_offset = left_end_effector_pose[1] - red_cube_state[1]

        if self.hold_cube and np.linalg.norm(y_offset) < 0.05:
            # If holding the blue cube and within 5cm of the red cube, release the blue cube
            action = np.zeros(6) #keep velocities as zero
            action[5] = 0.1  #increase vel to let go off the blue cube
            self.hold_cube = False  # Update the flag

        elif not self.hold_cube:
            if np.abs(red_cube_state[1] - 0.1) > 0.01 or np.abs(blue_cube_state[1] - red_cube_state[1]) > 0.01:
                # If the cubes have been moved to the right but their distance is more than 1cm
                # Move just the left end effector in the y direction
                action = np.zeros(6)
                action[1] = -0.1
        else:
            # Fail the task and keep the robot still if the cube holding or moving conditions are not met
            action = [0.0] * 6

        return action



import numpy as np

class CubesTowerSlide_3(CodePolicyLeft):

    def select_action(self, state):
        # the blue cube's position
        blue_cube_pos = np.array([0.59, 0.061, 0.120946143])

        # Red cube's position
        red_cube_pos = np.array([0.588111651, 0.0601549914, 0.095612514]) 

        # end effector's position
        end_effector_pos = np.array(state[:3])
        
        # The robot is going to make the blue cube move aside
        # checking if it is holding the blue cube
        if np.linalg.norm(blue_cube_pos - end_effector_pos) <= 1e-3:
            # Let go of the cube
            action = [0]*6 + [-1]
        else:
            # checking if it is close to pushing the red cube
            if np.linalg.norm(red_cube_pos - end_effector_pos) <= 1e-3:
                # Here, we calculate the coordinate of the movable position
                movable_pos = red_cube_pos + np.array([0, -0.1, 0])
                # Calculate the direction and speed for the bimanual robot to move
                # Here, we use a basic proportional controller for simplicity
                kp = 1
                action = kp*(movable_pos - end_effector_pos).tolist() + [0]
            else:
                # Move to target location above the red cube
                target_pos = red_cube_pos + np.array([0, 0.01, 0.025])
                action = np.append(kp*(target_pos - end_effector_pos), [0]).tolist()

        return action



import numpy as np

class CubesTowerSlide_4(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.target_pos = [0, -0.1, 0]
        self.reference_pos = None
        self.moving = False

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        # Convert pose to numpy array for easier manipulation
        left_end_effector_pose = np.array(left_end_effector_pose)
        red_cube_pos = np.array(state[14:17])

        if self.reference_pos is None:
            self.reference_pos = red_cube_pos.copy()
        
        if not self.moving:
            if np.abs(left_end_effector_pose[1] - self.reference_pos[1]) <= 0.01:
                left_gripper_vel = -1.0 # close gripper
                self.moving = True
            else:
                # Approach the cube first
                left_end_effector_vel = (red_cube_pos - left_end_effector_pose[:3]) * 10
                left_gripper_vel = 0.0
        else:
            if np.abs(left_end_effector_pose[1] - (self.reference_pos[1] - 0.1)) <= 0.01:
                left_gripper_vel = 1.0 # open gripper
                self.moving = False
            else:
                # Move cube to the right
                direction = (self.target_pos - left_end_effector_pose[:3])
                left_end_effector_vel = direction * 10
                left_gripper_vel = 0.0

        # Pack the action
        action = np.concatenate([left_end_effector_vel, left_gripper_vel, np.array([0.0]*6), [0.0]])
        return action



class CubesTowerSlide_5(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.initial_blue_cube_position = [0.59, 0.061, 0.120946143]
        self.initial_red_cube_position = [0.588111651, 0.0601549914, 0.095612514]

    def select_action(self, state):
        blue_cube_position = state[14:17]
        blue_cube_orientation = state[17:21]
        red_cube_position = state[21:24]
        red_cube_orientation = state[24:28]

        # Distance between blue and red cubes
        cube_distance = ((blue_cube_position[0] - red_cube_position[0]) ** 2 + (blue_cube_position[1] - red_cube_position[1]) ** 2) ** 0.5
        
        # Distance from target position
        target_distance = red_cube_position[1] - (self.initial_red_cube_position[1] - 0.1)
        
        # Check conditions for task failure
        if cube_distance > 0.01 or target_distance > 0.05:
            return [0.0] * 7

        # Check condition for task completion
        elif red_cube_position[1] <= (self.initial_red_cube_position[1] - 0.1) and blue_cube_position[1] <= (self.initial_blue_cube_position[1] - 0.1):
            return [0.0] * 7

        # Proceed with the task
        else:
            # Gripper velocity - open if holding the blue cube 
            left_gripper_vel = -1 if blue_cube_position[0] > 0 else 1
	    
            # Move gripper in the negative y direction
            left_end_effector_vel = [0.0, -0.1, 0.0, 0.0, 0.0, 0.0]
        
            action = left_end_effector_vel + [left_gripper_vel]
            
            return action



class CubesTowerSlide_6(CodePolicyLeft):

    def __init__(self):
        super().__init__()
        self.move_started = False
        self.move_achieved = False

    def select_action(self, state):
        blue_cube_pos = state[objects_state.index("blue_cube") + 2 : objects_state.index("blue_cube") + 5]
        red_cube_pos = state[objects_state.index("red_cube") + 2 : objects_state.index("red_cube") + 5]
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_end_effector_gripper_vel = state[13]
        distance_to_red_cube = (np.array(left_end_effector_pose[:3]) - np.array(red_cube_pos)).length()

        if self.move_achieved:
            action = [0.0] * 7
        else:
            if not self.move_started:
                if distance_to_red_cube < 0.05 and left_gripper > 0.5:
                    action = [0.0] * 7
                    action[12] = -1.0
                    if left_gripper == 0.0:
                        self.move_started = True
                else:
                    action = self.navigate_to_red_cube(state)
            else:
                if red_cube_pos[1] > 0.1 and abs(blue_cube_pos[1] - red_cube_pos[1]) < 0.01:
                    self.move_achieved = True
                    action = [0.0] * 7
                else:
                    action = [0.0] * 7
                    action[8] = -1.0

        return action

    def navigate_to_red_cube(self, state):
        red_cube_pos = state[objects_state.index("red_cube") + 2 : objects_state.index("red_cube") + 5]
        action = [0.0] * 7
        delta = np.array(red_cube_pos) - np.array(state[:3])
        action[6:9] = delta
        return action



class CubesTowerSlide_7(CodePolicyLeft):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        # Update the left_end_effector_pose and left_end_effector_vel
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]

        # Define the objects' states
        blue_cube_state = state[14:21]
        red_cube_state = state[21:28]

        # Get the positions and orientations of the cubes
        blue_cube_pos = blue_cube_state[:3]
        red_cube_pos = red_cube_state[:3]

        # Get the y-distance between the end effector and the red cube
        y_dist_effect_red_cube = left_end_effector_pose[1] - red_cube_pos[1]

        # Here, the task is to move the tower to the right (towards negative y-direction) 
        # Therefore, if the y-distance is greater than 0.05 meters, the end effector moves towards the cube
        if y_dist_effect_red_cube > 0.05:
            left_end_effector_vel[1] = -0.1

        # If y-distance between the end effector and red cube is within the desired range
        # Move both cubes to the right by imposing a negative velocity in y direction
        elif abs(blue_cube_pos[1]-red_cube_pos[1]) <= 0.01 and red_cube_pos[1] > -0.1:
            left_end_effector_vel[1] = -0.1

        # If both conditions do not meet, stop moving
        else:
            left_end_effector_vel[1] = 0.0  
        
        # In all cases, the gripper should open to let go of the blue cube. 
        # The velocity is negative since "<0 is for closing, > 0 for opening, must be in [-1, 1]"
        left_gripper_vel = -1.0

        # Calculate the action required from the robot arm
        action = left_end_effector_vel + [left_gripper_vel] + [0]*6  # the rest values are 0

        return action



class CubesTowerSlide_8(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.distance_to_move = 0.1
        self.red_cube_significant_pos = None

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        objects_state = state[14:]

        blue_cube_pose = objects_state[0:7]
        red_cube_pose = objects_state[7:14]

        # Let go of the blue cube
        if blue_cube_pose[0] == 0.59:
            left_end_effector_vel = [0,0,0,0,0,0] 
            left_gripper_vel = -1 
        else:
            # Find out the position of the red cube when the blue cube left.
            if self.red_cube_significant_pos is None:
                self.red_cube_significant_pos = red_cube_pose
            # Move the tower cubes to the right 
            elif red_cube_pose[1] > (self.red_cube_significant_pos[1] - self.distance_to_move):
                try:
                    distance_to_red_cube = math.sqrt(
                        (left_end_effector_pose[0]-red_cube_pose[0])**2 + 
                        (left_end_effector_pose[1]-red_cube_pose[1])**2 + 
                        (left_end_effector_pose[2]-red_cube_pose[2])**2)
                    if distance_to_red_cube > 0.05: 
                        raise Exception("Task failed - Distance between end effector and the red cube is > 0.05 meters")
                    else:
                        left_end_effector_vel = [0,-1,0,0,0,0] 
                        left_gripper_vel = 0 
                except Exception as e:
                    print(e)
            else:
                # If the cubes are moved to the right of 0.1 meters from their loading position, keep the robot still
                try:
                    inter_cube_distance = math.sqrt(
                        (red_cube_pose[0]-blue_cube_pose[0])**2 + 
                        (red_cube_pose[1]-blue_cube_pose[1])**2)
                    if inter_cube_distance > 0.01: 
                        raise Exception("Task failed - x-y distance between the cubes is > 0.01 meters")
                    else:
                        left_end_effector_vel = [0,0,0,0,0,0] 
                        left_gripper_vel = 0 
                except Exception as e:
                    print(e)

        # Keep right end effector velocities and gripper velocity to zero           
        action = list(left_end_effector_vel) + [left_gripper_vel] + [0.0]*7 

        return action



import numpy as np

class CubesTowerSlide_9(CodePolicyLeft):
    def __init__(self):
        super().__init__()
        self.target_position = np.array([0., -0.1, 0.])
        self.max_distance = 0.05
        self.blue_cube_state = None
        self.red_cube_state = None

    def set_geometry(self, blue_cube, red_cube):
        self.blue_cube_state = blue_cube
        self.red_cube_state = red_cube

    def calculate_distance(self, state1, state2):
        return np.linalg.norm(np.array(state1[:3]) - np.array(state2[:3]))

    def select_action(self, state):
        action = [0.0] * 7
        if self.blue_cube_state is not None and self.red_cube_state is not None:
            if self.calculate_distance(self.blue_cube_state, self.red_cube_state) > 0.01:
                return action 

            grip_velocity = -1 if state[12] > 0 else 0
            action[-1] = grip_velocity

            if grip_velocity == 0:
                if abs(state[1] - self.target_position[1]) > self.max_distance:
                    move_to_target_velocity = -1 if state[1] > self.target_position[1] else 1
                    action[1] = move_to_target_velocity
                else:
                    action[1] = 0

        return action



class BimanualHandover_0(CodePolicy):
    def __init__(self):
        super().__init__()
        self.left_gripper_contact_counter = 0

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        right_end_effector_pose = state[14:20]
        right_end_effector_vel = state[20:26]
        right_gripper = state[26]
        right_gripper_vel = state[27]
        objects_state = state[27:]

        box_position = objects_state[0:3]

        # calculate distance between the end effectors
        diff = [left_end_effector_pose[i] - right_end_effector_pose[i] for i in range(3)]
        distance = (diff[0]**2 + diff[1]**2 + diff[2]**2)**0.5

        # if the box is not in contact with any of the grippers or the distance is 
        # more than 0.4 meters, signal task failure
        if left_gripper == 0 and right_gripper == 0 or distance > 0.4:
            action = [0] * 14
            print("Task failed")
            return np.array(action)

        # if the box is in contact with the left gripper, increase the 
        # contact_counter
        if left_gripper:
            self.left_gripper_contact_counter += 1

        # if the left gripper has not been in contact with the box for the last 50 
        # time steps and the box is in contact with the right gripper, signal task 
        # success
        if self.left_gripper_contact_counter >= 50 and right_gripper:
            action = [0] * 14
            print("Task accomplished")
            return np.array(action)

        # Otherwise, move right gripper closer to the box, grasp the box with the right 
        # gripper, and then release the box from the left gripper.
        action = [0.0] * 14
        action[6] = -0.1 if distance > 0.05 else 0
        action[7] = +1.0 if distance <= 0.05 else 0
        action[0] = -1.0 if self.left_gripper_contact_counter >= 50 else 0

        return np.array(action)



class BimanualHandover_1(CodePolicy):
    def __init__(self):
        super().__init__()
        self.steps_since_left_hand_release = 0

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] 
        right_end_effector_pose = state[14:20]
        right_end_effector_vel = state[20:26]
        right_gripper = state[26]
        right_gripper_vel = state[27]  

        box_state = state[28:35]  # assuming the box is the first object after the robot state
        box_position = box_state[:3]

        distance_between_end_effectors = np.linalg.norm(np.array(right_end_effector_pose[:3]) -
                                                        np.array(left_end_effector_pose[:3]))

        # In order to capture enough states, we will simply mimic the policy of current state.
        action = left_end_effector_vel.tolist()
        action.append(left_gripper_vel)
        action.extend(right_end_effector_vel.tolist())
        action.append(right_gripper_vel)

        # If the robot had grasped the box in the previous step and the current distance between them is <0.4, 
        # then the robot should maintain this state or close the gap between the box and the right gripper, also 
        # the robot can now start to loose the grip of the left gripper.
        if self.steps_since_left_hand_release > 0 and distance_between_end_effectors < 0.4:
            action[6:12] = (np.array(box_position) - np.array(right_end_effector_pose[:3])).tolist()
            action[13] = -0.1   # slowly open left gripper
            self.steps_since_left_hand_release += 1

        # If the above condition is not met yet, The robot should be closing in on the box until the distance is less than 0.4.
        elif distance_between_end_effectors > 0.4:
            action[6:12] = (np.array(box_position) - np.array(right_end_effector_pose[:3])).tolist()
        else:
            action[13] = -1  # gripper cannot hold the box 
        
        # If the task succeed, we will reset the epoch counter for next time.
        if self.steps_since_left_hand_release >= 50:
            self.steps_since_left_hand_release = 0
            
        # If no contact with the box, the task will be considered as failed.
        if left_gripper == 0 and right_gripper == 0:
            action[6:8] = [0.0]*6   # stop right hand when task failed
            action[13] = 1  # open left gripper when task failed
            print("Task failed: there is no contact with the box.")
        
        # If the distance between end effectors is too high, the task is considered as failed.
        if distance_between_end_effectors > 0.4:
            action[:12] = [0.0]*12  # stop both hands when task failed
            action[13] = 1  # open left gripper when task failed
            print("Task failed: Distance between end effectors is too large.")
        
        return np.array(action)



import numpy as np

class BimanualHandover_2(CodePolicy):
    def __init__(self):
       super().__init__()  
       self.steps_since_handover = 0

    def select_action(self, state):
        # Parse state
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] 
        right_end_effector_pose = state[14:20]
        right_end_effector_vel = state[20:26]
        right_gripper = state[26]
        right_gripper_vel = state[27]
        objects_state = state[27:]

        # Calculate distance between grippers
        gripper_distance = np.linalg.norm(np.array(left_end_effector_pose[:3]) - np.array(right_end_effector_pose[:3]))

        # Handover completed condition
        both_grippers_touching = left_gripper > 0 and right_gripper > 0
        handover_complete = self.steps_since_handover >= 50

        # Create movement policy
        action = np.zeros(14)
        if gripper_distance > 0.4:
            action[6:12] = -1  # Move right hand closer
        elif not handover_complete and both_grippers_touching:
            action[8] = -1  # Close right gripper
            action[4] = 1  # Open left gripper
            self.steps_since_handover += 1
        elif handover_complete:
            action[8] = 0  # Stop moving right gripper
            action[4] = 0  # Stop moving left gripper

        return np.array(action)



import numpy as np

class BimanualHandover_3(CodePolicy):
    def __init__(self):
        super().__init__()
        self.distance_threshold = 0.4
        self.timesteps_without_contact = 0

    def select_action(self, state):
        left_end_effector_pose = np.array(state[:6])
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        right_end_effector_pose = np.array(state[14:20])
        right_end_effector_vel = state[20:26]
        right_gripper = state[26]
        right_gripper_vel = state[27]  
        objects_state = state[27:]  

        box_pos = np.array([0.545, 0.105, 0.24])
        distance_to_box_right = np.linalg.norm(right_end_effector_pose[:3] - box_pos)

        if distance_to_box_right > self.distance_threshold:
            print("Task failed: distance to box from right gripper too high.")
            return []

        if left_gripper == 0 and right_gripper == 0:
            print("Task failed: box is not grasped.")
            return []

        if left_gripper == 0:
            self.timesteps_without_contact += 1
        else:
            self.timesteps_without_contact = 0

        if self.timesteps_without_contact >= 50 and right_gripper == 1:
            print("Task solved!")
            return []

        action = [0.0]*14  
        
        if right_gripper == 0:
            right_gripper_vel = 1.0

        if right_gripper == 1 and self.timesteps_without_contact >= 49:
            left_gripper_vel = 1.0

        action[0:6] = left_end_effector_vel
        action[6] = left_gripper_vel
        action[7:13] = right_end_effector_vel
        action[13] = right_gripper_vel

        return np.array(action)




class BimanualHandover_4(CodePolicy):
    def __init__(self):
        super().__init__()
        self.left_gripper_contact_steps = 0

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        right_end_effector_pose = state[14:20]
        right_end_effector_vel = state[20:26]
        right_gripper = state[26]
        right_gripper_vel = state[27]

        box_pos = [0.545, 0.105, 0.24]

        action = [0]*14

        # Check distance between grippers
        distance = sum([(a - b)**2 for a, b in zip(left_end_effector_pose[:3], right_end_effector_pose[:3])]) ** 0.5

        # Move right gripper towards the box
        if distance > 0.4:
            action[7:10] = [a_b for a_b in zip(right_end_effector_pose[:3], box_pos)]

        # Close right gripper on box
        if distance < 0.1 and right_gripper < 0.1:
            action[13] = -1

        # Open left gripper if right gripper has box
        if right_gripper > 0.9 and self.left_gripper_contact_steps > 50:
            action[6] = 1

        # Keep track of left gripper contact with box 
        if left_gripper > 0.9:
            self.left_gripper_contact_steps += 1
        else:
            self.left_gripper_contact_steps = 0

        # Assign vel values for both grippers
        action[:6] = left_end_effector_vel
        action[6:13] = right_end_effector_vel

        return np.array(action)



class BimanualHandover_5(CodePolicy):
    def __init__(self):
        super().__init__()
        self.gripper_distance = 0.4
        self.max_detach_timesteps = 50
        self.detach_timesteps = 0

    def compute_action(self, state):
        left_gripper_pose = state[:6]
        right_gripper_pose = state[14:20]

        # Check gripper distance
        distance = np.linalg.norm(np.array(right_gripper_pose[:3]) - np.array(left_gripper_pose[:3]))

        if distance > self.gripper_distance:
            print('Fail: The distance between the end effectors is more than 0.4 meters.')
            return None

        if state[12]:  # if left gripper is still in contact with the box
            self.detach_timesteps = 0
            action = [0.0] * 14
            action[6:12] = (np.array(right_gripper_pose[:3]) - np.array(left_gripper_pose[:3])).tolist()  # move right gripper towards box

            if state[26]:  # if right gripper is touching the box
                action[27] = -1  # close right gripper
        else:
            self.detach_timesteps += 1
            action = [0.0] * 14  # keep still

            if self.detach_timesteps >= self.max_detach_timesteps:
                print('Success: The box has been successfully transferred.')
                return None

        return np.array(action)



class BimanualHandover_6(CodePolicy):
    def __init__(self):
        super().__init__()
        self.count = 0 # count for tracking end-effector distance
        self.left_released = False

    def select_action(self, state):

        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] 
        right_end_effector_pose = state[14:20]
        right_end_effector_vel = state[20:26]
        right_gripper = state[26]
        right_gripper_vel = state[27]
        objects_state = state[27:]

        # Steps:
        # 1. Move right gripper towards object 
        # 2. Grasp box with right gripper
        # 3. Release box with left gripper

        end_effector_distance = ((right_end_effector_pose[0]-left_end_effector_pose[0])**2 +
        (right_end_effector_pose[1]-left_end_effector_pose[1])**2 +
        (right_end_effector_pose[2]-left_end_effector_pose[2])**2 )**0.5 

        # Move right gripper close to the box
        if end_effector_distance > 0.4:
            action = right_end_effector_vel - 0.01

        # If Right gripper close to box, grasp it
        elif right_gripper == 0 and not self.left_released:
            right_gripper_vel = -1 #closing

        # Release box from left gripper
        elif left_gripper == 1:
            left_gripper_vel = 1 #opening
            self.left_released = True

        if self.left_released:
            self.count += 1

        if self.count > 50 and right_gripper == 1:
            print("Task completed successfully")

        # If the box is not grasped by either hand or the hands moved apart more than 0.4m
        elif (right_gripper == 0 and left_gripper == 0) or end_effector_distance > 0.4:
            print("Task Failed!")

        action = left_end_effector_vel.tolist() + [left_gripper_vel] + right_end_effector_vel.tolist() + [right_gripper_vel]


        return np.array(action)



import numpy as np

class BimanualHandover_7(CodePolicy):
    def __init__(self):
        super().__init__()
        self.time_steps_since_release = 0
        self.tolerance = 0.02  # tolerance for the distance between grippers and box
        self.distance_threshold = 0.4  # threshold for the distance between end effectors

    def select_action(self, state):
        box_pose = state[27:30]  # assuming the box's position is the next after the grippers
        
        left_end_effector_pose = state[:6]
        left_end_effector_distance = np.linalg.norm(np.array(left_end_effector_pose[:3]) - np.array(box_pose))
        
        right_end_effector_pose = state[14:20]
        right_end_effector_distance = np.linalg.norm(np.array(right_end_effector_pose[:3]) - np.array(box_pose))
        
        end_effectors_distance = np.linalg.norm(np.array(left_end_effector_pose[:3])-np.array(right_end_effector_pose[:3]))
        
        if right_end_effector_distance < self.tolerance:
            gripper_action = [-1.0, 1.0]  # Close the right gripper
        elif left_end_effector_distance < self.tolerance and self.time_steps_since_release >= 50:
            gripper_action = [1.0, -1.0]  # Open the left gripper
        elif end_effectors_distance > self.distance_threshold:
            gripper_action = [0.0, 0.0]  # Neither open nor close grippers
        else:
            gripper_action = [0.0, -1.0]  # Close right gripper, remain left gripper
        
        self.time_steps_since_release += 1
        
        action = [0.0] * 12 + gripper_action
        
        return np.array(action)



from math import sqrt

END_EFFECTOR_POSE_SIZE = 6
END_EFFECTOR_VEL_SIZE = 6
GRIPPER_SIZE = 1
GRIPPER_VEL_SIZE = 1
ACTION_SIZE = 14

class BimanualHandover_8(CodePolicy):
    def __init__(self):
        super().__init__()

    def calculate_box_distance_and_orientation(self, pose1, pose2):
        distance = sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pose1, pose2)))
        orientation = [0, 0, 0]  # Return orientation assuming flat surface with no rotation
        return distance, orientation

    def calculate_gripper_velocity(self, state):
        return 0  # Assuming no velocity needed as per the task scenario

    def select_action(self, state):
        left_end_effector_pose = state[:END_EFFECTOR_POSE_SIZE]
        left_end_effector_vel = state[END_EFFECTOR_POSE_SIZE:END_EFFECTOR_POSE_SIZE*2]
        left_gripper = state[END_EFFECTOR_POSE_SIZE*2:END_EFFECTOR_POSE_SIZE*2+GRIPPER_SIZE]
        left_gripper_vel = state[END_EFFECTOR_POSE_SIZE*2+GRIPPER_SIZE:END_EFFECTOR_POSE_SIZE*2+GRIPPER_SIZE+GRIPPER_VEL_SIZE] 

        right_end_effector_pose = state[END_EFFECTOR_POSE_SIZE*2+GRIPPER_SIZE+GRIPPER_VEL_SIZE:END_EFFECTOR_POSE_SIZE*3+GRIPPER_SIZE+GRIPPER_VEL_SIZE]
        right_end_effector_vel = state[END_EFFECTOR_POSE_SIZE*3+GRIPPER_SIZE+GRIPPER_VEL_SIZE:END_EFFECTOR_POSE_SIZE*4+GRIPPER_SIZE+GRIPPER_VEL_SIZE]
        right_gripper = state[END_EFFECTOR_POSE_SIZE*4+GRIPPER_SIZE+GRIPPER_VEL_SIZE:END_EFFECTOR_POSE_SIZE*4+GRIPPER_SIZE*2+GRIPPER_VEL_SIZE]
        right_gripper_vel = state[END_EFFECTOR_POSE_SIZE*4+GRIPPER_SIZE*2+GRIPPER_VEL_SIZE:END_EFFECTOR_POSE_SIZE*4+GRIPPER_SIZE*2+GRIPPER_VEL_SIZE*2]

        objects_state = state[END_EFFECTOR_POSE_SIZE*4+GRIPPER_SIZE*2+GRIPPER_VEL_SIZE*2:]

        # Calculate the box distance and orientation from the gripper (should use real robot and object state - assumed here)
        box_distance, box_orientation = self.calculate_box_distance_and_orientation(left_end_effector_pose, right_end_effector_pose)

        # Calculate the gripper velocity for grasping box (should use box distance and orientation - assumed here)
        gripper_velocity = self.calculate_gripper_velocity(state)

        # Formulate the action
        action = [0.0]*ACTION_SIZE
        action[6] = gripper_velocity  # Assuming we need to move the left gripper
        action[20] = -gripper_velocity  # Assuming we need to move the right gripper to the opposite direction

        return np.array(action)



# Your policy class
class BimanualHandover_9(CodePolicy):
    def __init__(self):
        super().__init__()
        self.no_touch_counter = 0

    def select_action(self, state):
        # Extract state variables
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        right_end_effector_pose = state[14:20]
        right_end_effector_vel = state[20:26]
        box_pose = state[27:33]    # assuming box position is in state array
        
        # Calculate the distance between end effectors
        dist = np.linalg.norm(np.array(left_end_effector_pose[:3]) - np.array(right_end_effector_pose[:3]))

        # If distance > limit, move right end effectors towards the left one. Else, grasp the box with right gripper
        if dist > 0.4:
            action = right_end_effector_vel - 0.01 * (right_end_effector_pose - left_end_effector_pose)
            action = np.append(action, 0)  # leave right gripper at current state
        else:
            action = np.zeros(7)  # stop moving right end effector
            action[6] = -1  # close right gripper

        # If box is grasped by right gripper and not yet released by left, release it. Keep the box otherwise.
        if abs(state[26] - 1) < 0.01:  # right gripper is closed
            self.no_touch_counter += 1
            if self.no_touch_counter > 50:
                action = np.append(action[:6], 0)  # stop moving left end effector
                action = np.append(action, 1)  # open left gripper
        else:   
            self.no_touch_counter = 0
            action = np.append(action[:6], -1)    # keep closing left gripper
            action = np.append(action, 0) # stop moving left end effector
        
        return np.array(action)



import numpy as np

class CubeInCup_0(CodePolicy):
    def __init__(self):
        super().__init__()
        
    def select_action(self, state):
        # Continuing from the parent class State
        cube_position, cup_position = state[27:30], state[30:33]
        
        # Reset action to 0
        action = np.zeros(14)
        
        # Calculate the distance between cube and cup
        dist_cube_cup = np.linalg.norm(np.array(cup_position) - np.array(cube_position))
        
        # Calculate the relative position of cube with respect to the cup
        cube_relative_pos = np.array(cube_position) - np.array(cup_position)
        
        # fail conditions
        if not (-1 <= state[13] <= 1) or not (-1 <= state[27] <= 1):
            print("Task Failed: Gripper finger not in contact with the cup.")
            return action
        if dist_cube_cup > 0.2:
            print("Task Failed: Distance between left end effector and the cube is more than 0.2 meter.")
            return action
            
        # implement the action based on the state 
        if dist_cube_cup > 0.025:
            # Close left gripper (holding the cube) and move towards the cup
            action[6:12] = cube_relative_pos
            action[12] = -1
        else:
            # Open left gripper to drop the cube into the cup
            action[12] = 1
            print("Task Solved: The cube is placed into the cup.")
        
        return action



import numpy as np
from scipy.spatial import distance

class CubeInCup_1(CodePolicy):

    def __init__(self):
        self.cube_position = np.array([0.545, 0.155, 0.235])
        self.cup_position = np.array([0.55, -0.12, 0.19])
        self.max_cube_distance = 0.2
        super().__init__()

    def distance_between(self, pose1, pose2):
        return distance.euclidean(pose1, pose2)

    def task_completed(self, state):
        return (self.distance_between(state[:3], self.cube_position) <= 0.025 and
                self.distance_between(state[14:17], self.cup_position) <= 0.025 and
                state[13] < 0 and state[27] < 0)

    def move_towards(self, point, velocity):
        action = list(point) + list(velocity) + [0 if state[12] > 0.5 else -1]
        return action + [0.0] * 7

    def select_action(self, state):
        left_effector_pos = state[:3]
        right_effector_pos = state[14:17]

        right_distance_to_cup = self.distance_between(right_effector_pos, self.cup_position)
        left_distance_to_cup = self.distance_between(left_effector_pos, self.cup_position)

        left_distance_to_cube = self.distance_between(left_effector_pos, self.cube_position)

        if self.task_completed(state):
            return [0.0] * 14

        if left_distance_to_cube > 0.2:
            target = self.cube_position if left_effector_pos[2] > 0.2 else self.cup_position
            return self.move_towards(target, state[6:9])
        
        if right_distance_to_cup > 0.025:
            return [0.0] * 7 + self.move_towards(self.cup_position, state[20:23])

        return [0.0] * 14



import numpy as np

class CubeInCup_2(CodePolicy):
  def __init__(self):
    super().__init__()

  def select_action(self, state):
    cup_position = np.array([0.55, -0.12, 0.19])
    cube_position = np.array([0.545, 0.155, 0.235])
    left_end_effector_pose = np.array(state[:6])
    left_end_effector_vel = np.array(state[6:12])
    left_gripper = state[12]
    left_gripper_vel = state[13]
    right_end_effector_pose = np.array(state[14:20])
    right_end_effector_vel = np.array(state[20:26])
    right_gripper = state[26]
    right_gripper_vel = state[27]

    if np.linalg.norm(left_end_effector_pose[:-3] - cube_position) > 0.2:
      return [0] * 14 # end the episode

    if right_gripper == 0:
      return [0] * 14 # end the episode

    if np.linalg.norm(cube_position - cup_position) <= 0.025:
      return [0] * 14 # task completed successfully

    # Define desired velocities to move the cube toward the cup
    move_toward_cup_vel = (cup_position - left_end_effector_pose[:-3]) * 0.1
    move_toward_cup_vel = np.hstack([move_toward_cup_vel, [0, 0, 0]]) # keep orientation

    left_end_effector_vel = move_toward_cup_vel
    right_end_effector_vel = np.array([0, 0, 0, 0, 0, 0]) # keep the right gripper static
    left_gripper_vel = 0 # keep the left gripper closed
    right_gripper_vel = 0 # keep the right gripper closed

    action = np.hstack([left_end_effector_vel, left_gripper_vel, right_end_effector_vel, right_gripper_vel])
    return action



import numpy as np

class CubeInCup_3(CodePolicy):
    
    def __init__(self):
        super(CubeInCup_3, self).__init__()
        self.cube_position = np.array([0.545, 0.155, 0.235])
        self.cup_position = np.array([0.55, -0.12, 0.19])
        self.max_distance_cube_effector = 0.2
        self.max_distance_cube_cup = 0.025
    

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_gripper = state[12]
        right_end_effector_pose = state[14:20]
        right_gripper = state[26]

        cube_position = self.cube_position
        cup_position = self.cup_position
    
        # Check if the task has failed
        if right_gripper < 0 or np.linalg.norm(left_end_effector_pose[:3] - cube_position) > self.max_distance_cube_effector:
            print("Task Failed.")
            return None 

        # Check if the task is already solved
        if right_gripper > 0 and np.linalg.norm(cup_position - cube_position) <= self.max_distance_cube_cup:
            print("Task Solved.")
            return None 

        # Move the left end effector toward the cube
        if np.linalg.norm(left_end_effector_pose[:3] - cube_position) > 0.005:
            left_end_effector_target = cube_position 
        else: # Position the cube above the cup 
            left_end_effector_target = np.array([cup_position[0], cup_position[1], left_end_effector_pose[2]])
        
        left_end_effector_vel = (left_end_effector_target - left_end_effector_pose[:3]) * 0.1

        # Keep the right end effector stationary, hold cup
        right_end_effector_vel = np.zeros(6)
        right_gripper_vel = 0

        # Ensure the left gripper is open or closed as appropriate
        if np.linalg.norm(left_end_effector_pose[:3] - cube_position) <= 0.005:
            left_gripper_vel = -1.0 # Close gripper if near the cube
        else:
            left_gripper_vel = 1.0 # Otherwise open it to avoid dragging the cube

        action = np.concatenate([left_end_effector_vel, [left_gripper_vel], right_end_effector_vel, [right_gripper_vel]])
        return action.tolist()
  


import numpy as np

class CubeInCup_4(CodePolicy):

  cube_position = np.array([0.545, 0.155, 0.235]) # Target cube position
  cup_position = np.array([0.55, -0.12, 0.19]) # Target cup position
  distance_thresh = 0.025 # Distance threshold for cube to cup
  distance_fail = 0.2 # Fail case distance for left end effector to cube

  def select_action(self, state):
    # Parse state
    left_end_effector_pose = np.array(state[:6])
    left_end_effector_vel = np.array(state[6:12])
    left_gripper = state[12]
    left_gripper_vel = state[13]
    right_end_effector_pose = np.array(state[14:20])
    right_end_effector_vel = np.array(state[20:26])
    right_gripper = state[26]
    right_gripper_vel = state[27]
    objects_state = state[27:]

    # Condition for task failure
    end_effector_cube_distance = np.linalg.norm(left_end_effector_pose[:3] - self.cube_position)
    if(end_effector_cube_distance > self.distance_fail):
      return None # task failed

    # Condition for task success
    left_effector_to_cube = self.cube_position - left_end_effector_pose[:3] # Vector pointing from left end effector to cube
    cube_to_cup = self.cup_position - self.cube_position # Vector pointing from cube to cup
    
    if np.linalg.norm(left_effector_to_cube-cube_to_cup) <= self.distance_thresh:
      return [0.0] * 14 # Task complete, maintain current state

    # Compute control action    
    # Simple control scheme: if distance larger than threshold, move towards target; if closer than threshold, stop moving
    desired_left_pose = self.cube_position + (self.cup_position - self.cube_position) / 2
    left_effector_velocity = 0.1 * (desired_left_pose - left_end_effector_pose[:3])
    left_action = np.concatenate((left_effector_velocity, [0,0,0], [left_gripper_vel]))
    right_action = np.concatenate((right_end_effector_vel, [0,0,0], [right_gripper_vel]))

    return np.hstack((left_action, right_action)).tolist()



import numpy as np

class CubeInCup_5(CodePolicy):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        cube_position = np.array([0.545, 0.155, 0.235])
        cup_position = np.array([0.55, -0.12, 0.19])
        left_end_effector_pose = np.array(state[:6])
        left_end_effector_vel = np.array(state[6:12])
        left_gripper = state[12]
        left_gripper_vel = state[13]
        right_end_effector_pose = np.array(state[14:20])
        right_end_effector_vel = np.array(state[20:26])
        right_gripper = state[26]
        right_gripper_vel = state[27]
    
        action = [0.0] * 14

        distance_from_cube = np.linalg.norm(left_end_effector_pose - cube_position)
        distance_from_cup = np.linalg.norm(cube_position - cup_position)

        if distance_from_cube > 0.2:  # Check if the task failed
            print("Task Failed: The cube is too far from the left end effector.")
        elif not (left_gripper < 0 or right_gripper < 0):  # Check if any gripper is not in contact with the objects
            print("Task Failed: No finger of the right gripper is in contact with the cup.")
        elif distance_from_cup <= 0.025:  # check if the task is completed
            print("Task Completed: The cube is inside the cup.")
        else:  # manipulation policy
            action[:3] = (cup_position - left_end_effector_pose + [0, 0, 0.025])  # move to above the cup
            action[3:6] = [0, 0, -1]  # rotate the gripper to face downwards
            action[6] = -1  # close the gripper to lift the cube
            action[7:] = right_end_effector_vel + [right_gripper_vel]  # keep the right grip static

        return action



import numpy as np


class CubeInCup_6(CodePolicy):
    def __init__(self):
        super().__init__()
        self.target_cube_position = np.array([0.545, 0.155, 0.235])
        self.target_cup_position = np.array([0.55, -0.12, 0.19])

    def select_action(self, state):
        super().select_action(state)
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        right_end_effector_pose = state[14:20]
        right_end_effector_vel = state[20:26]

        cube_pos_error = self.target_cube_position - np.array(left_end_effector_pose[:3])
        cube_orient_error = self.target_cube_position - np.array(left_end_effector_pose[3:])

        cube_vel = left_end_effector_vel[:6] + 0.1 * np.concatenate((cube_pos_error, cube_orient_error), axis=0)

        # Keep the cup still
        cup_pos_error = np.zeros(3)
        cup_orient_error = np.zeros(3)

        cup_vel = right_end_effector_vel[:6] + 0.1 * np.concatenate((cup_pos_error, cup_orient_error), axis=0)

        # keep the grippers in their current state
        gripper_vel = [state[13], state[27]]

        action = np.concatenate((cube_vel, cup_vel, gripper_vel), axis=0)

        return action



import numpy as np

class CubeInCup_7(CodePolicy):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        # Retrieve state information
        left_end_effector_pose = state[:6]
        right_end_effector_pose = state[14:20]
        left_gripper = state[12]
        right_gripper = state[26]
        
        # Define positions of the cube and cup
        cube_pos, cup_pos = np.array([0.545, 0.155, 0.235]), np.array([0.55, -0.12, 0.19])

        # Calculate distance of grippers from objects
        left_distance = np.linalg.norm(np.array(left_end_effector_pose[:3]) - cube_pos)
        right_distance = np.linalg.norm(np.array(right_end_effector_pose[:3]) - cup_pos)

        # Initialize default action
        action = [0.0]*4
        
        # If the left gripper is too far from the cube or the right gripper detaches from the cup, fail and remain stationary
        if right_distance > 0.025 or left_distance > 0.2 :
            return action
        
        # Calculate direction vector to move cube towards cup
        direction = cup_pos - cube_pos        

        # If the cube is not yet in the cup, manipulate the left gripper to move towards the cup
        if left_distance > 0.025:
            action[:3] = direction
            action[-2] = 0.0 # Do not open or close the left gripper
        else:
            action[-2] = -1.0 # Close the left gripper
            
        return action



import numpy as np

class CubeInCup_8(CodePolicy):

    def __init__(self):
        super().__init__()

    def select_action(self, state):
        # Object's positions
        cube_pos = np.array([0.545, 0.155, 0.235]) 
        cup_pos = np.array([0.55, -0.12, 0.19]) 

        # current left gripper pos
        left_gripper_pos = np.array(state[:3])

        # current right gripper pos
        right_gripper_pos = np.array(state[14:17])
        
        # If the distance from the left gripper (holding the cube)
        # to the cube location is more than 0.2 meter, task is failed
        if np.linalg.norm(left_gripper_pos - cube_pos) > 0.2:
            print("Task failed")
            return []

        # If the right gripper is not in contact with the cup, task failed
        if state[26] < 0:
            print("Task failed")
            return []
        
        # control policy to move left gripper (holding the cube) towards the cup
        left_control = ((cup_pos - cube_pos) - left_gripper_pos) * 0.01 
        # keep right gripper still
        right_control = np.zeros(7)

        # Combine control signals and keep grippers closed
        action = np.concatenate((left_control, [-1], right_control, [-1]))

        # If the cube is close enough to the cup, task is solved
        if np.linalg.norm((cup_pos - cube_pos) - left_gripper_pos) <= 0.025:
            print("Task solved")
            return []

        return action.tolist()



import numpy as np
from scipy.spatial import distance

class CubeInCup_9(CodePolicy):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        # Unpack the state properties
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] 
        right_end_effector_pose = state[14:20]
        right_end_effector_vel = state[20:26]
        right_gripper = state[26]
        right_gripper_vel = state[27]
        objects_state = state[27:]

        # define cube and cup position
        cube_position = np.array([0.545, 0.155, 0.235])
        cup_position = np.array([0.55, -0.12, 0.19])

        # Calculate the distance between cube and cup and left end effector and cube.
        cube_cup_distance = distance.euclidean(cube_position, cup_position)
        left_effector_cube_distance = distance.euclidean(left_end_effector_pose[:3], cube_position)

        # Check if the task is failed or not.
        if right_gripper_vel < 0 or left_effector_cube_distance > 0.2:
            action = [0.0] * 14  # keep robot still
            print('Task failed.')
        elif cube_cup_distance <= 0.025:
            action = [0.0] * 14  # keep robot still
            print('Task solved.')
        else:
            # Keep the velocities constant for the end effectors
            # but close the left and right grippers accordingly.
            left_gripper_vel = -1 if left_effector_cube_distance > 0.025 else 0
            right_gripper_vel = 1 if cube_cup_distance > 0.025 else 0
            action = left_end_effector_vel.tolist() + [left_gripper_vel] + \
                     right_end_effector_vel.tolist() + [right_gripper_vel]

        return action




import numpy as np

class BimanualPenInTape_0(CodePolicy):

  def __init__(self):
    super().__init__()
    self.pen_position = np.array([0.515, -0.025, 0.37])
    self.tape_position = np.array([0.51, -0.025, 0.19])
    self.maximum_distance = 0.025
    self.successful_distance = 0.005

  def select_action(self, state):
    diff = state[14:17] - state[:3]    # Calculate the difference in position of the end effectors

    if np.linalg.norm(diff) > self.maximum_distance:     # If they are too far apart

      action = np.concatenate([diff/2, np.zeros(5), -diff/2, np.zeros(5)]) # Move the end effectors towards each other

    elif np.linalg.norm(diff) > self.successful_distance:    # If they are almost at the correct distance

      action = np.zeros(14)  # Stay in place
    else:

      action = np.zeros(14)   # Stay in place

    return action

  def check_task_success_failure(self, state):
    pen_position = state[14:17]
    tape_position = state[:3]

    if np.linalg.norm(pen_position - tape_position) <= self.successful_distance:
      return "Task is successful"
    
    elif (not state[12] or not state[26] or np.linalg.norm(pen_position - tape_position) > self.maximum_distance):
      return "Task has failed"
    else:
      return "Task in progress"



class BimanualPenInTape_1(CodePolicy):
    def __init__(self):
        super().__init__()

    def select_action(self, state):

        # Object states
        tape_pos = state[28:31]
        tape_ori = state[31:34]

        pen_pos = state[34:37]

        # Calculating the distance between the tape hole and the pen
        diff_pos = np.array(tape_pos) - np.array(pen_pos)
        distance = np.linalg.norm(diff_pos)
        
        # Checking if the pen is within a acceptable distance from the tape hole
        if distance < 0.005:
            action = [0.0] * 14 # task is solved, no need to move
        else:
            
            if np.linalg.norm(diff_pos[:2]) > 0.025: # the pen is too far in the horizontal plane, task failed.
                raise ValueError("Pen is too far from the tape")
        
            # Moving towards the direction of the hole
            direction = diff_pos / distance            
            action = list(direction * 0.005) + [0.0] * 11 # assuming that only small movements are needed
            
            # Making sure that the grippers keep holding the objects
            action[6] = np.clip(state[13], -0.01, 0.01) # adjust left gripper's velocity to maintain grip
            action[13] = np.clip(state[27], -0.01, 0.01) # adjust right gripper's velocity to maintain grip

        return np.array(action)
 


class BimanualPenInTape_2(CodePolicy):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        """
        Decides the action to perform given the state of the robot.
        """
        # Setup variables for pen, tape, and distances
        pen_pose = state[14:17]
        tape_pose = state[27:30]
        dist = np.linalg.norm(np.array(tape_pose[:2]) - np.array(pen_pose[:2]))
        close_dist = 0.005
        fail_dist = 0.025

        action = [0.0] * 14  # Action to keep the robot still

        # If the distance between pen and tape is too large or contact with pen or tape is lost
        if dist > fail_dist or state[12] == 0 or state[26] == 0:
            print("Fail condition met: either distance is too large or contact is lost.")
            return action  # Perform no action, i.e., keep the robot still

        # If the pen is close enough to the tape
        if dist <= close_dist:
            print("Task solved: pen has been inserted into the tape.")
            return action  # Perform no action, i.e., keep the robot still

        # If pen is not yet inserted into tape, and not too far or too close
        else:
            print("Inserting pen into tape...")
            # Your action here
            
        return action



import numpy as np

class BimanualPenInTape_3(CodePolicy):
    def __init__(self):
        super(BimanualPenInTape_3, self).__init__()
        self.maximum_distance = 0.025
        self.target_distance = 0.005
        self.tape_hole_radius = 0.0125  # obtained from xml file
        self.pen_radius = 0.006  # obtained from xml file
    
    def select_action(self, state):
        left_end_effector_pose = np.array(state[:6])
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        right_end_effector_pose = np.array(state[14:20])
        right_end_effector_vel = state[20:26]
        right_gripper = state[26]
        right_gripper_vel = state[27]
        objects_state = state[27:]

        relative_distance = np.linalg.norm(left_end_effector_pose - right_end_effector_pose)
        
        if relative_distance > self.maximum_distance:
            # Pinpoint the failure state and stop operation.
            print("Task Failed: Distance between PEN and TAPE is more than 0.025 meters.")
            return np.array([0.0]*14)
        
        actionable_distance = relative_distance - (self.tape_hole_radius + self.pen_radius)
        if actionable_distance <= self.target_distance:
            print("Task Solved: Pen has been successfully inserted into the hole of the tape.")
            return np.array([0.0]*14)
        
        # Compute the desired velocity based on the relative distance and distance to the target
        desired_velocity = (actionable_distance - self.target_distance) *0.1
        
        # Limitation check to handle gripper velocity constraints.
        left_gripper_vel = np.clip(desired_velocity, -1, 1)
        right_gripper_vel = -np.clip(desired_velocity, -1, 1)

        action = list(left_end_effector_vel) 
        action.append(left_gripper_vel) 
        action += list(right_end_effector_vel) 
        action.append(right_gripper_vel)

        return np.array(action)



import math

class BimanualPenInTape_4(CodePolicy):
    def __init__(self):
        super().__init__()

    def calculate_distance(self, object1, object2):
        # This method calculates the euclidean distance between two objects
        return math.sqrt((object2[0] - object1[0])**2 + (object2[1] - object1[1])**2 + (object2[2] - object1[2])**2)

    def select_action(self, state):
        tape_position = state[14:17]
        pen_position = state[:3]

        # Calculate distance between pen and tape
        distance = self.calculate_distance(pen_position, tape_position)

        # Check the conditions for task failure and success
        if distance > 0.025:
            print('Task Failed: The distance between the objects in the horizontal plane is more than 0.025 meters.')
            return
        elif distance <= 0.005:
            print('Task Solved: Both objects are grasped, and the pen is at a distance from the tape of 0.005 or less.')
            return

        # Calculate velocity for end effectors
        left_end_effector_vel = state[3:6]
        right_end_effector_vel = state[17:20]
        if distance > 0.01:
            # If distance is large, increase the velocity of end_effectors to approach each other
            left_end_effector_vel = [v + 0.1 for v in left_end_effector_vel]
            right_end_effector_vel = [v - 0.1 for v in right_end_effector_vel]
        elif distance < 0.01:
            # If distance is small, decrease the velocity of end_effectors to avoid collision
            left_end_effector_vel = [v - 0.1 for v in left_end_effector_vel]
            right_end_effector_vel = [v + 0.1 for v in right_end_effector_vel]

        action = left_end_effector_vel + [state[12]] + [state[13]] + right_end_effector_vel + [state[26]] + [state[27]]
        return action


class BimanualPenInTape_5(CodePolicy):

    def __init__(self):
        super().__init__()

    def select_action(self, state):
        # We define the necessary variables
        tape_position = np.array([0.51, -0.025, 0.19])
        pen_position = np.array([0.515, -0.025, 0.37])
        left_end_effector_pose = np.array(state[:6])
        right_end_effector_pose = np.array(state[14:20])

        # Calculate the required position and orientation changes for each end effector
        left_required_pose = np.concatenate((pen_position - left_end_effector_pose[:3], -left_end_effector_pose[3:]))
        right_required_pose = np.concatenate((tape_position - right_end_effector_pose[:3], -right_end_effector_pose[3:]))
        
        # We calculate the distance between effector poses to objects
        distance_to_pen = np.linalg.norm(left_end_effector_pose[:3] - pen_position)
        distance_to_tape = np.linalg.norm(right_end_effector_pose[:3] - tape_position)

        # We map distances to velocity by a linear map of [-1, 1]
        left_action = np.concatenate((np.clip(distance_to_pen*2 - 1, -1, 1), left_required_pose[3:]))
        right_action = np.concatenate((np.clip(distance_to_tape*2 - 1, -1, 1), right_required_pose[3:]))
        
        # we make sure grippers remain closed
        gripper_action = [state[12], state[26]]

        # Concatenate all actions
        action = np.concatenate((left_action, gripper_action, right_action))

        return action



class BimanualPenInTape_6(CodePolicy):
    def __init__(self):
        super().__init__()
        self.pen_position = [0.515, -0.025, 0.37]
        self.tape_position = [0.51, -0.025, 0.19] 
        self.task_solved = False

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        right_end_effector_pose = state[14:20]
        right_end_effector_vel = state[20:26]

        # Assign a velocity to make the left and right effectors move towards their respective objects
        left_end_effector_vel = np.subtract(self.pen_position, left_end_effector_pose[:3])
        right_end_effector_vel = np.subtract(self.tape_position, right_end_effector_pose[:3])

        # Closing the left and right grippers to hold their respective objects
        left_gripper_vel = state[13] = -1
        right_gripper_vel = state[27] = -1

        distance_between_objects = np.linalg.norm(np.subtract(self.pen_position, self.tape_position))
        if distance_between_objects <= 0.005:
            self.task_solved = True
            return [0.0] * 14  # Stop moving if task is solved
        elif distance_between_objects > 0.025 or not self._is_gripping(state):
            return None  # Task failed
        else:
            return list(left_end_effector_vel) + [left_gripper_vel] + list(right_end_effector_vel) + [right_gripper_vel]

    def _is_gripping(self, state):
        # Check if the robot is gripping the objects
        left_gripper = state[12]
        right_gripper = state[26]
        return left_gripper < 0 and right_gripper < 0


class BimanualPenInTape_7(CodePolicy):
    def __init__(self):
        super().__init__()
        
    def select_action(self, state):
        # Unpack state variables
        left_end_effector_pose = np.array(state[:6])
        left_end_effector_vel = np.array(state[6:12])
        right_end_effector_pose = np.array(state[14:20])
        right_end_effector_vel = np.array(state[20:26])

        # Seperate object states
        tape_position = np.array(state[27:30])
        tape_orientation = state[30:33]        
        pen_position = np.array(state[33:36])
        pen_orientation = state[36:39]

        # Position controllers
        left_position_ctrl = self.position_controller(pen_position, left_end_effector_pose[:3])
        right_position_ctrl = self.position_controller(tape_position, right_end_effector_pose[:3])

        # Velocity damping
        left_velocity_damping = -0.5 * left_end_effector_vel
        right_velocity_damping = -0.5 * right_end_effector_vel

        # Sum of position control and damping
        left_action = left_position_ctrl + left_velocity_damping
        right_action = right_position_ctrl + right_velocity_damping

        # Pack into action array
        action = np.concatenate([left_action, [0.0], right_action, [0.0]])

        return action

    def position_controller(self, target_position, current_position):
        # Proportional position control
        error = target_position - current_position
        control = 1 * error
        return control


class BimanualPenInTape_8(CodePolicy):
    def __init__(self):
        super().__init__()

    def compute_distance(self, pos1, pos2):
        return np.sqrt(np.sum((np.array(pos1) - np.array(pos2))**2))

    def select_action(self, state):
        # Assigning state variables
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        right_end_effector_pose = state[14:20]
        right_end_effector_vel = state[20:26]
        right_gripper = state[26]
        right_gripper_vel = state[27]
        objects_state = state[28:]

        pen_position = objects_state[:3]
        tape_position = objects_state[7:10]

        # Compute the distance between the pen and the tape
        dist_pen_tape = self.compute_distance(left_end_effector_pose[:2], right_end_effector_pose[:2])

        action = np.zeros_like(state)

        # Check if the robot is holding the objects properly
        if left_gripper <= 0 or right_gripper <= 0:
            # If not, close the grippers
            action[13] = -1
            action[27] = -1

        # Check if the pen is at a distance from the tape of 0.005 or less
        elif dist_pen_tape <= 0.005:
            # If so, the task is solved and the robot should remain still
            action = [0.0] * 14

        # If the distance between the objects in the horizontal plane is more than 0.025 meters,
        # Move the left hand towards the right using a simple control policy 
        # proportional to the distance.
        elif dist_pen_tape > 0.025:
            action[6:12] = (np.array(right_end_effector_pose[:2]) - np.array(left_end_effector_pose[:2])) * 0.1

        return action


class BimanualPenInTape_9(CodePolicy):
    def __init__(self):
        super().__init__()
        # Initialize pen and tape as empty arrays
        self.pen = np.array([])
        self.tape = np.array([])
        # Initialize successful state as False
        self.success = False
        
    def select_action(self, state):
        self.pen = np.array(state[14:20])
        self.tape = np.array(state[:6])
        
        # The success condition is when the absolute difference between pen and tape 
        # in the first two dimensions (x and y) are less than 0.005, and the third dimension (z) is less than 0.025 
        condition1 = np.abs(self.pen - self.tape) <= np.array([0.005, 0.005, 0.025, 0, 0, 0])
        
        # The condition for failure is no contact between the pen and the left gripper
        condition2 = state[12] == 0 
        # The condition for failure is no contact between the tape and the right gripper
        condition3 = state[26] == 0
        
        # If success condition is met, set success state as True and stop the robot 
        if all(condition1):
            self.success = True
            action = [0.0] * 14 # Keeps the robot still
        # If any failure condition is met, stop the simulation (assumed by the policy)
        elif condition2 or condition3:
            action = [0.0] * 14 # Keeps the robot still
        # If none of the above conditions are met, continue with the task 
        else:
            action = (self.pen - self.tape).tolist() + [0, 0]   # Moves the Left gripper towards the pen
            action += self.pen_vel.tolist() + [-1]              # Keeps same velocity of the Left gripper and tries to close it 
            action += (self.tape - self.pen).tolist() + [0, 0]  # Moves the Right gripper towards the tape
            action += self.tape_vel.tolist() + [-1]             # Keeps same velocity of the Right gripper and tries to close it
            
        return action





import numpy as np

class BimanualBoxLift_0(CodePolicy):
    def __init__(self, state):
        super().__init__()

        self.target_box = {'position': [0.55, 0.0, 0.105],
                           'orientation': [0, 0, 0, 1]}
        self.joint_goal = [0.0] * 14

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        right_end_effector_pose = state[14:20]

        left_distance  = np.linalg.norm(np.array(left_end_effector_pose[:3])  - np.array(self.target_box['position']))
        right_distance = np.linalg.norm(np.array(right_end_effector_pose[:3]) - np.array(self.target_box['position']))

        if left_distance > 0.2 or right_distance > 0.2:
            return 'Task failed: End effector is too far from the box.'
        elif np.all(np.array(self.joint_goal) == np.array(state[12:26])):
            if (self.target_box['position'][2] + 0.1) <= np.minimum(left_end_effector_pose[2], right_end_effector_pose[2]):
                return 'Task solved: The box has been successfully lifted 0.1m.'
            else:
                return 'Task partially solved: All four fingers are touching the box but it has not been lifted properly.'
        else:
            action = np.linspace(0, -1, 14).tolist()    # Close all fingers of both grippers at a steady rate
            return np.array(action)




import numpy as np

class BimanualBoxLift_1(CodePolicy):
    def __init__(self):
        super().__init__()
        self.spaghetti_box_pos = np.array([0.55, 0.0, 0.105])
        self.spaghetti_box_size = np.array([0.029, 0.49, 0.048])
        self.spaghetti_box_height = self.spaghetti_box_pos[2] + self.spaghetti_box_size[2] / 2.0

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        right_end_effector_pose = state[14:20]
        
        task_solved = False
        action = [0.0] * 14
        
        left_distance = np.linalg.norm(self.spaghetti_box_pos[:2] - np.array(left_end_effector_pose[:2]))
        right_distance = np.linalg.norm(self.spaghetti_box_pos[:2] - np.array(right_end_effector_pose[:2]))
        
        # Closing grippers if end effectors are close enough to the box
        if left_distance < 0.1 and right_distance < 0.1:
            action[6] = -1.0 
            action[13] = -1.0  
          
        # Lifting when all fingers are touching the box
        if left_end_effector_pose[2] - self.spaghetti_box_height < 0.01 and right_end_effector_pose[2] - self.spaghetti_box_height < 0.01:
            action[4] = 0.1 
            action[11] = 0.1 
          
        # Task solved
        if left_end_effector_pose[2] - self.spaghetti_box_height > 0.1 and right_end_effector_pose[2] - self.spaghetti_box_height > 0.1:
            task_solved = True
      
        # Task failed
        if left_distance > 0.2 or right_distance > 0.2:
            action = [0.0] * 14
            
        return np.array(action), task_solved



import numpy as np

class BimanualBoxLift_2(CodePolicy):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        action = super().select_action(state)

        # Define the box properties
        spaghetti_box_pos = np.array([0.55, 0.0, 0.105])
        min_distance = 0.2
        max_lift_height = 0.1

        # Get gripper states
        left_end_effector_pose = np.array(state[:6])
        right_end_effector_pose = np.array(state[14:20])

        # Calculate distance from the grippers to the box
        dist_left = np.linalg.norm(left_end_effector_pose[:3] - spaghetti_box_pos)
        dist_right = np.linalg.norm(right_end_effector_pose[:3] - spaghetti_box_pos)

        # If grippers are too far away, move them towards the box. 
        # Else, move towards the lift position
        if dist_left > min_distance or dist_right > min_distance:
            action[:3] = spaghetti_box_pos - left_end_effector_pose[:3]
            action[7:10] = spaghetti_box_pos - right_end_effector_pose[:3]
        else:
            action[:3] = np.array([0.0, 0.0, max_lift_height])
            action[7:10] = np.array([0.0, 0.0, max_lift_height])

        # Make sure the grippers are closed or close them if they aren't
        action[6] = -1.0 if state[12] >= 0.0 else 0.0 # Left gripper vel
        action[13] = -1.0 if state[26] >= 0.0 else 0.0 # Right gripper vel

        return np.array(action)



import numpy as np
class BimanualBoxLift_3(CodePolicy):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        left_end_effector_pose = np.array(state[:6])
        left_end_effector_vel = np.array(state[6:12])
        right_end_effector_pose = np.array(state[14:20])
        right_end_effector_vel = np.array(state[20:26])

        spaghetti_box_position = np.array([0.55, 0.0, 0.105])

        # Calculating the distance between the end effectors and the spaghetti box
        left_distance = np.linalg.norm(left_end_effector_pose - spaghetti_box_position)
        right_distance = np.linalg.norm(right_end_effector_pose - spaghetti_box_position)

        if left_distance > 0.2 or right_distance > 0.2:
            print("Task failed. One of the end effectors is further from the spaghetti box than 0.2 meters.")
            action = [0.0] * 14
            return np.array(action)

        # Set velocities towards the spaghetti box
        left_end_effector_vel = (spaghetti_box_position - left_end_effector_pose) * 0.1
        right_end_effector_vel = (spaghetti_box_position - right_end_effector_pose) * 0.1

        # Set grippers to close
        left_gripper_vel = -1.0
        right_gripper_vel = -1.0

        action = np.concatenate((left_end_effector_vel, [left_gripper_vel], right_end_effector_vel, [right_gripper_vel]))

        return action



import numpy as np

class BimanualBoxLift_4(CodePolicy):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        super().select_action(state)

        # assuming the position of the spaghetti box is constant and given in the problem statement
        spaghetti_box_pose = np.array([0.55, 0, 0.105])

        left_end_effector_pose = np.array(state[:6])
        right_end_effector_pose = np.array(state[14:20])
        
        # calculate distance from the end effectors to the box
        distance_left = np.linalg.norm(left_end_effector_pose[:3] - spaghetti_box_pose)
        distance_right = np.linalg.norm(right_end_effector_pose[:3] - spaghetti_box_pose)

        # check if either end effector is too far from the box
        if distance_left > 0.2 or distance_right > 0.2:
            return "Task failed because distance is too far from the box."
        
        # calculate velocity needed to reach the box
        left_end_effector_vel = (spaghetti_box_pose - left_end_effector_pose[:3]) / distance_left
        right_end_effector_vel = (spaghetti_box_pose - right_end_effector_pose[:3]) / distance_right

        # check if both fingers are touching the box
        if state[12] < 0.1 or state[26] < 0.1:
            # if not, move fingers to touch the box
            left_gripper_vel = -1 # close left gripper
            right_gripper_vel = -1 # close right gripper
        else:
            # if yes, lift the box
            spaghetti_box_pose[2] += 0.1
            # recalculate velocity needed to reach the new position of the box
            left_end_effector_vel = (spaghetti_box_pose - left_end_effector_pose[:3]) / distance_left
            right_end_effector_vel = (spaghetti_box_pose - right_end_effector_pose[:3]) / distance_right
            left_gripper_vel = 0
            right_gripper_vel = 0
            
        action = np.concatenate([left_end_effector_vel, [left_gripper_vel], right_end_effector_vel, [right_gripper_vel]])

        return action



import numpy as np

BOX_POSITION = np.array([0.55, 0.0, 0.105])

class BimanualBoxLift_5(CodePolicy):
    def __init__(self):
        super().__init__()
        self.target_height = BOX_POSITION[2] + 0.1  # target height to lift the box

    def select_action(self, state):
        
        # Get current robot state
        left_end_effector_pose = state[:6]
        right_end_effector_pose = state[14:20]
        
        # Calculate distance to the box
        box_distance = {
            "left": np.linalg.norm(left_end_effector_pose[:3] - BOX_POSITION),
            "right": np.linalg.norm(right_end_effector_pose[:3] - BOX_POSITION),
        }

        # Verify that we are not too far from the box
        if max(box_distance.values()) > 0.2:
            print("Task failed: End effector is too far from the box.")
            return np.array([0.0] * 14)

        # Check if we are holding the box and it is lifted to the target height
        if state[12] + state[26] == 2 and max(left_end_effector_pose[2], right_end_effector_pose[2]) > self.target_height:
            print("Task solved: Box lifted successfully.")
            return np.array([0.0] * 14)

        # If we are touching the box with both grippers
        if state[12] + state[26] == 2:
            print("Partially solved: Holding the box. Now lift it.")
            action = np.array([0, 0, 0.1, 0, 0, 0, 0.0, 0, 0, 0.1, 0, 0, 0, 0])
        else:
            # Otherwise, move fingers to grasp the box
            print(f"Moving fingers towards the box. Distance: {box_distance}")
            action = BOX_POSITION - left_end_effector_pose[:3] 
            action = np.concatenate([action, [1.0], BOX_POSITION - right_end_effector_pose[:3], [1.0]])

        return action



import numpy as np

class CodePolicy:
    def __init__(self):
        super().__init__()

    # To be overridden by a subclass.
    def select_action(self, state):
        pass


class BimanualBoxLift_6(CodePolicy):
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        # unpack the state information
        spaghetti_box = np.array([0.55, 0.0, 0.105])
        left_end_effector_pose = np.array(state[:6])
        right_end_effector_pose = np.array(state[14:20])

        # compute the distance to the spaghetti box
        left_distance = np.linalg.norm(left_end_effector_pose[:3] - spaghetti_box)
        right_distance = np.linalg.norm(right_end_effector_pose[:3] - spaghetti_box)

        # initialize the action array
        action = np.zeros(28)  # change with your state size

        if max(left_distance, right_distance) > 0.2: # check if either end effector is further than 0.2 meters
            return action  # doing nothing to keep the robot still, assuming the task is failed 

        # close the graspers if they have reached the spaghetti box
        if max(left_distance, right_distance) < 0.02:  # assuming 0.02 as a very near distance to the box
            action[13] = -1.  # closing left griper
            action[27] = -1.  # closing right gripper

        # compute the desired z-position for lifting the spaghetti box
        target_height = spaghetti_box[2] + 0.1

        # update the action if the box is not lifted enough
        if max(left_end_effector_pose[2], right_end_effector_pose[2]) < target_height:
            action[6:12] = [0.,0.,1.,0.,0.,0.]  # lift up the left gripper
            action[20:26] = [0.,0.,1.,0.,0.,0.]  # lift up the right gripper

        return action



import numpy as np

class CodePolicy:
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        pass  # this code as provided, has been omitted for brevity

# Desired policy class
class BimanualBoxLift_7(CodePolicy):
    def __init__(self):
        super().__init__()
        self.target_pos = [0.55, 0.0, 0.105]    # Position of the spaghetti box
        self.grasping_done = False   # Flag to check if the box has been grasped or not
        self.lift_height = 0.1       # Lifting height of the box

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        right_end_effector_pose = state[14:20]

        # Define target velocity based on the current position of effectors and the target position
        target_vel = (np.array(self.target_pos) - np.array(left_end_effector_pose[:3])) * 0.1
        target_vel_right = (np.array(self.target_pos) - np.array(right_end_effector_pose[:3])) * 0.1

        # Determine the action depending on the position of end-effectors
        if np.linalg.norm(left_end_effector_pose[:3] - self.target_pos) > 0.2 or np.linalg.norm(right_end_effector_pose[:3] - self.target_pos) > 0.2:
            return "Task Failed: Either end effector is further from the spaghetti box than 0.2 meters."
        elif not self.grasping_done:
            action = list(target_vel) + [-1] + list(target_vel_right) + [-1]   # Move towards the box and try to grasp
            if np.linalg.norm(np.array(left_end_effector_pose[:3]) - np.array(self.target_pos)) < 0.01 and np.linalg.norm(np.array(right_end_effector_pose[:3]) - np.array(self.target_pos)) < 0.01:
                self.grasping_done = True    # If both the effectors are close enough to the box, consider grasping done
        elif left_end_effector_pose[2] < (self.target_pos[2] + self.lift_height):
            action = [0, 0, 0.1, 0, 0, 0, -1, 0, 0, 0.1, 0, 0, 0, -1]   # If the box has been grasped, try to lift it
        else:
            return "Task Succesful: The spaghetti box is lifted of 0.1m over its loading height, while all fingers are in contact with it."

        return np.array(action)



import numpy as np

class CodePolicy:
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        pass

class BimanualBoxLift_8(CodePolicy):
    def __init__(self):
        super().__init__()
        self.spaghetti_box_pos = np.array([0.55, 0.0, 0.105]) 
        self.spaghetti_box_size = np.array([0.029, 0.49, 0.048])
        self.action_grip = -1  # -1 is for closing
    
    def select_action(self, state):
        # Splitting state into parts
        left_end_effector_pose = np.array(state[:6])
        right_end_effector_pose = np.array(state[14:20])

        # Measure distance to the spaghetti box
        left_distance = np.linalg.norm(self.spaghetti_box_pos - left_end_effector_pose[:3])
        right_distance = np.linalg.norm(self.spaghetti_box_pos - right_end_effector_pose[:3])

        # Initialize action with zeros
        action = np.zeros(14)

        if max(left_distance, right_distance) > 0.2:
            # Task failed, halt the robot
            return action

        if min(left_distance, right_distance) > 0.0001:
            # Move end effectors to the spaghetti box if they are not touching it
            action[:3] = (self.spaghetti_box_pos - left_end_effector_pose[:3]) * 10  # KP controller for left arm
            action[7:10] = (self.spaghetti_box_pos - right_end_effector_pose[:3]) * 10  # KP controller for right arm
        else:
            # If the spaghetti box was lifted to the required height, hold the state
            if self.spaghetti_box_pos[2] >= 0.205:
                return action

            # Both end effectors touching the box, grip and lift it up
            action[3] = self.action_grip  # close left gripper
            action[10] = self.action_grip  # close right gripper
            action[:3] = [0, 0, 1]  # lift up with left arm
            action[7:10] = [0, 0, 1]  # lift up with right arm

        return action



import numpy as np
from scipy.spatial.transform import Rotation as R

# Base class
class CodePolicy:
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        pass


class BimanualBoxLift_9(CodePolicy):
    def __init__(self):
        super().__init__()
        self.task_solved = False
        self.box_pos = np.array([0.55, 0.0, 0.105])

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13]
        right_end_effector_pose = state[14:20]
        right_end_effector_vel = state[20:26]
        right_gripper = state[26]
        right_gripper_vel = state[27] 
        objects_state = state[27:]

        # calculate the position error of both arms' end effector.
        box_error_left = self.box_pos - left_end_effector_pose[:3]
        box_error_right = self.box_pos - right_end_effector_pose[:3]

        # if either arm is too far from the box, the task is failed.
        if np.linalg.norm(box_error_left) > 0.2 or np.linalg.norm(box_error_right) > 0.2 :
            self.task_solved = False
            return np.array([0.0]*14)

        # when the robot arms are close enough, start to grasp the box.
        if np.linalg.norm(box_error_left) < 0.02 and np.linalg.norm(box_error_right) < 0.02:
            action = np.concatenate((left_end_effector_vel, [-1], right_end_effector_vel, [-1])) # start to grasp

            # check if the box has been lifted for more than 0.1m and fingers are in contact with box
            if (objects_state[2] - self.box_pos[2]) > 0.1 and left_gripper*right_gripper < 0:
                self.task_solved = True
            return action

        # adjust the position of the arms towards the box.
        kp = 1.0  # proportional control gain
        kr = 0.1   # rotational control gain
        desired_left_velocity = kp * box_error_left
        desired_right_velocity = kp * box_error_right
        desired_action = np.concatenate((desired_left_velocity, [0], desired_right_velocity, [0]))

        return desired_action
