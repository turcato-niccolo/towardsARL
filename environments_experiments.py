from environments import *
prompts = {}    # Data structures to save the LLM prompts
dicts = {}      # and config dict

def dits_reward(w, q, v, alpha):
    """
        General distance-based reward function.

        Adapted from
        S. Levine, N. Wagener, P. Abbeel, "Learning Contact-Rich Manipulation Skills with Guided Policy Search,"
        in International Conference on Robotics and Automation (ICRA), 2015

        https://arxiv.org/pdf/1501.05611
    """
    return lambda d: -(w * (d**2) + q + v * np.log(d**2 + alpha))

def get_dist_reward(d_0, A=10):
    """
        Get the distance based reward function, such that r(d_0) = 0, and r(0)=A.
    """
    v = 1
    alpha = 10 ** (-5)
    w = 100
    q = dits_reward(w, 0, v, alpha)(d_0)

    amp = dits_reward(w, q, v, alpha)(0)

    v = A * v / amp
    q = A * q / amp
    w = A * w / amp

    return dits_reward(w, q, v, alpha)



prompts["VialInsertion"] = ("The robot is holding a vial with its gripper. "
                            "Lower the vial and insert it into the vial carrier, which is just below the vial.) "
                            "Consider the task solved when the vial is still touched by both gripper fingers, and the vial is at 0.06m height or below. "
                            "Consider the task failed if the fingers loose contact with the vial, or the vial's distance in the x-y plane from its loading position is more than 0.025m.")
dicts["VialInsertion"] = "vial_insertion.yaml"

class VialInsertion(YumiLeftEnv_objects_in_scene):
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(VialInsertion, self).__init__(render_mode, env_config_dict, control_mode)
        self.max_end_effector_angular_vel = 0.1

    def reward_fun(self, observation, action):
        vial_position = np.array(sim_utils.getObjPose(self.objects_id['vial']))[:3]
        load_position = np.array(self.objects_dict['vial']['load_pos'])
        
        horizontal_distance = np.linalg.norm(vial_position[:2] - load_position[:2])
        
        finger_1, finger_2, _ = self.check_fingers_touching(self.objects_id['vial'])
        vial_height = vial_position[2]

        rewards_dict = {
            'horizontal_distance': -horizontal_distance,
            'vial_height': -vial_height if vial_height > 0.06 else 0.0,
            'finger_grip': 1.0 if finger_1 and finger_2 else 0.0
        }
        
        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        vial_pose = np.array(sim_utils.getObjPose(self.objects_id['vial']))[:3]
        load_pose = np.array(self.objects_dict['vial']['load_pos'])
        finger_1, finger_2, _ = self.check_fingers_touching(self.objects_id['vial'])

        task_solved = vial_pose[2] <= 0.06 and np.linalg.norm(vial_pose[:2] - load_pose[:2]) <= 0.025 and finger_1 and finger_2
        return {'task_solved': task_solved}

    def termination_condition(self):
        vial_pose = np.array(sim_utils.getObjPose(self.objects_id['vial']))[:3]
        load_pose = np.array(self.objects_dict['vial']['load_pos'])
        finger_1, finger_2, _ = self.check_fingers_touching(self.objects_id['vial'])

        return np.linalg.norm(vial_pose[:2] - load_pose[:2]) > 0.025 or not finger_1 or not finger_2

class VialInsertion_baseline(VialInsertion):
    """
        Baseline with only distance based reward
        Adapted from
        S. Levine, N. Wagener, P. Abbeel, "Learning Contact-Rich Manipulation Skills with Guided Policy Search,"
        in International Conference on Robotics and Automation (ICRA), 2015
    """
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(VialInsertion_baseline, self).__init__(render_mode, env_config_dict, control_mode)

        load_position = np.array(self.objects_dict['vial']['load_pos'])
        target_position = copy.deepcopy(load_position)
        target_position[2] = 0.06

        d_0 = np.linalg.norm(target_position[:3] - load_position[:3])

        self.r = get_dist_reward(d_0)
        self.vial_target_position = target_position

    def reward_fun(self, observation, action):
        vial_position = np.array(sim_utils.getObjPose(self.objects_id['vial']))[:3]

        d = np.linalg.norm(vial_position[:3] - self.vial_target_position[:3])

        reward = self.r(d)
        rewards_dict = {"dist_reward": reward}

        return reward, rewards_dict

prompts["VialGrasp"] = ("The robot has just inserted a vial in a vial carrier, the gripper is very close to the vial. "
                        "Grasp the vial, and lift out of the vial carrier, avoid to touch the vial carrier itself. "
                        "Consider the task completed when the vial is lifted of 0.1m above loading height. "
                        "Consider the task failed if the fingers loose contact with the vial, or the vial's distance in the x-y plane from its loading position is more than 0.025m.")
dicts["VialGrasp"] = "vial_grasp.yaml"

class VialGrasp(YumiLeftEnv_objects_in_scene):
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(VialGrasp, self).__init__(render_mode, env_config_dict, control_mode)
        self.max_end_effector_angular_vel = 0.1

    def reward_fun(self, observation, action):
        rewards_dict = {}

        vial_pose = np.array(sim_utils.getObjPose(self.objects_id['vial']))
        vial_carrier_pose = np.array(sim_utils.getObjPose(self.objects_id['vial_carrier']))
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))

        distance_to_vial_in_xy = np.linalg.norm(vial_pose[:2] - end_effector_left_pose[:2])

        height_diff = vial_pose[2] - self.objects_dict['vial']['load_pos'][2]

        finger_1, finger_2, total_contact_force = self.check_fingers_touching(self.objects_id['vial'])

        vial_not_moved = float(
            np.linalg.norm(vial_pose[:2] - np.array(self.objects_dict['vial']['load_pos'][:2])) < 0.025)

        rewards_dict = {
            "vial_height_reward": height_diff if height_diff > 0 else 0,
            "finger_contacts_reward": 1 * finger_1 + 1 * finger_2,  # encourage grasping the vial
            "distance_to_vial_penalty": -distance_to_vial_in_xy,  # encourage getting close to the vial
            "not_moved_reward": 10 * vial_not_moved
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        vial_pose = sim_utils.getObjPose(self.objects_id['vial'])
        task_solved = vial_pose[2] > self.objects_dict['vial']['load_pos'][2] + 0.1
        return {'task_solved': task_solved}

    def termination_condition(self):
        vial_pose = np.array(sim_utils.getObjPose(self.objects_id['vial']))
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))

        finger_1, finger_2, _ = self.check_fingers_touching(self.objects_id['vial'])

        vial_moved = np.linalg.norm(vial_pose[:2] - np.array(self.objects_dict['vial']['load_pos'][:2])) > 0.025

        return not (finger_1 or finger_2) or vial_moved

class VialGrasp_baseline(VialGrasp):
    """
        Baseline with only distance based reward
        Adapted from
        S. Levine, N. Wagener, P. Abbeel, "Learning Contact-Rich Manipulation Skills with Guided Policy Search,"
        in International Conference on Robotics and Automation (ICRA), 2015
    """
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(VialGrasp_baseline, self).__init__(render_mode, env_config_dict, control_mode)

        load_position = np.array(self.objects_dict['vial']['load_pos'])
        target_position = copy.deepcopy(load_position)
        target_position[2] += 0.1

        d_0 = np.linalg.norm(target_position[:3] - load_position[:3])

        self.r = get_dist_reward(d_0)
        self.vial_target_position = target_position

    def reward_fun(self, observation, action):
        vial_position = np.array(sim_utils.getObjPose(self.objects_id['vial']))[:3]

        d = np.linalg.norm(vial_position[:3] - self.vial_target_position[:3])

        reward = self.r(d)
        rewards_dict = {"dist_reward": reward}

        return reward, rewards_dict

prompts["CubeSlide"] = ("The robot gripper is close to the blue cube, touch it with the gripper fingers and push it close to the red cube. "
                        "Consider the task solved if the distance between the cubes is less than 0.04 meters. "
                        "Consider the task failed if the distance of the end effector from the blue cube is more than 0.1m.")
dicts["CubeSlide"] = "two_cubes.yaml"

class CubeSlide(YumiLeftEnv_objects_in_scene):
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(CubeSlide, self).__init__(render_mode, env_config_dict, control_mode)
        self.max_end_effector_angular_vel = 0.1

    def reward_fun(self, observation, action):
        rewards_dict = {}
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))
        blue_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))
        red_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['red_cube']))
        
        dist_blue = np.linalg.norm(np.array(blue_cube_pose[:3]) - np.array(end_effector_left_pose[:3]))
        dist_red_blue = np.linalg.norm(np.array(red_cube_pose[:3]) - np.array(blue_cube_pose[:3]))
        
        finger_1, finger_2, total_contact_force = self.check_fingers_touching(self.objects_id['blue_cube'])
        rewards_dict["dist_blue"] = -dist_blue
        rewards_dict["dist_red_blue"] = -dist_red_blue
        rewards_dict["finger_contacts_reward"] = 1 * finger_1 + 1 * finger_2
        
        info = self._get_info()   
        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        blue_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))
        red_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['red_cube']))
        dist_red_blue = np.linalg.norm(np.array(red_cube_pose[:3]) - np.array(blue_cube_pose[:3]))
        
        task_solved = dist_red_blue < 0.04
        return {'task_solved': task_solved}
      
    def termination_condition(self):
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))
        blue_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))
        dist_blue = np.linalg.norm(np.array(blue_cube_pose[:3]) - np.array(end_effector_left_pose[:3]))
        
        return dist_blue > 0.1

class CubeSlide_baseline(CubeSlide):
    """
        Baseline with only distance based reward
        Adapted from
        S. Levine, N. Wagener, P. Abbeel, "Learning Contact-Rich Manipulation Skills with Guided Policy Search,"
        in International Conference on Robotics and Automation (ICRA), 2015
    """
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(CubeSlide_baseline, self).__init__(render_mode, env_config_dict, control_mode)

        load_position = np.array(self.objects_dict['blue_cube']['load_pos'])
        target_position = np.array(self.objects_dict['red_cube']['load_pos'])
        # distance_vector = load_position - target_position
        # distance_vector /= np.linalg.norm(distance_vector) # normalized
        # target_position = target_position + 0.04 * distance_vector

        self.d_0 = np.linalg.norm(target_position[:3] - load_position[:3])

        self.r = get_dist_reward(self.d_0)
        self.cube_target_position = target_position

    def reward_fun(self, observation, action):
        cube_position = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))[:3]

        d = np.linalg.norm(cube_position[:3] - self.cube_target_position[:3])

        x = np.maximum(np.abs(d) * (np.abs(d)-0.04)/(self.d_0-0.04), 0.0) # Adapted to cap when d=0.04

        reward = self.r(x)
        # print("reward:", reward)

        rewards_dict = {"dist_reward": reward}

        return reward, rewards_dict



prompts["CubePick"] = ("The robot gripper is close to a blue cube and a red cube. Grasp the blue cube with both fingers, then move the blue cube upwards of 0.05 meters. "
                        "Consider the task completed when the blue cube is lifted by 0.05m over its loading height. "
                        "Consider the task failed if the distance of the end effector from the blue cube is more than 0.1m or the red cube is moved from its loading position of 0.005m or more.")
dicts["CubePick"] = "two_cubes_closer.yaml"

class CubePick(YumiLeftEnv_objects_in_scene):
    def reward_fun(self, observation, action):
        rewards_dict = {}  # Defines the reward components

        # getting the poses both the cubes and left end effector
        blue_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))
        red_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['red_cube']))
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))

        # calculating distances
        distance_blue = np.linalg.norm(np.array(blue_cube_pose[:3]) - np.array(end_effector_left_pose[:3]))
        distance_red = np.linalg.norm(np.array(red_cube_pose[:3]) - self.objects_dict['red_cube']['load_pos'])

        # calculating height difference of the blue cube from the initial position
        height_difference_blue = blue_cube_pose[2] - self.objects_dict['blue_cube']['load_pos'][2]

        # finger contact with blue cube
        finger_1_blue, finger_2_blue, _ = self.check_fingers_touching(self.objects_id['blue_cube'])

        rewards_dict = {
            "negative_distance_blue_cube_reward": - distance_blue,  # encourage getting close to the blue cube
            "negative_distance_red_cube_penalty": - distance_red if distance_red > 0.005 else 0,
            # negative reward if red cube is moved
            "finger_contacts_reward": 3 * (finger_1_blue + finger_2_blue),  # encourage finger contact with blue cube
            "blue_cube_height_reward": height_difference_blue if blue_cube_pose[2] >
                                                                 self.objects_dict['blue_cube']['load_pos'][2] else 0,
            # reward if blue cube is lifted
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        blue_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))
        red_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['red_cube']))
        is_task_solved = blue_cube_pose[2] > self.objects_dict['blue_cube']['load_pos'][
            2] + 0.05  # task is solved if the blue cube is lifted over 0.05m
        return {'task_solved': is_task_solved}

    def termination_condition(self):
        blue_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))
        red_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['red_cube']))
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))
        distance_blue = np.linalg.norm(np.array(blue_cube_pose[:3]) - np.array(end_effector_left_pose[:3]))
        distance_red = np.linalg.norm(np.array(red_cube_pose[:3]) - self.objects_dict['red_cube']['load_pos'])

        # Conditions for termination: distance from blue cube > 0.1 or red cube is moved by more than 0.005m
        return distance_blue > 0.1 or distance_red > 0.005

class CubePick_baseline(CubePick):
    """
        Baseline with only distance based reward
        Adapted from
        S. Levine, N. Wagener, P. Abbeel, "Learning Contact-Rich Manipulation Skills with Guided Policy Search,"
        in International Conference on Robotics and Automation (ICRA), 2015
    """
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(CubePick_baseline, self).__init__(render_mode, env_config_dict, control_mode)

        load_position = np.array(self.objects_dict['blue_cube']['load_pos'])
        target_position = np.array(self.objects_dict['blue_cube']['load_pos'])
        target_position[2] += 0.05

        d_0 = np.linalg.norm(target_position[2] - load_position[2])

        self.r = get_dist_reward(d_0)
        self.cube_target_position = target_position

    def reward_fun(self, observation, action):
        cube_position = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))[:3]

        d = np.linalg.norm(cube_position[2] - self.cube_target_position[2])

        reward = self.r(d)
        rewards_dict = {"dist_reward": reward}

        return reward, rewards_dict



prompts["CentrifugeInsertion"] = ("The robot is holding a vial with its gripper. Lower the vial and insert it into the lab centrifuge, which is just below the vial. "
                       "Consider the task solved when the vial is still touched by both gripper fingers, and the vial is at 0.08m height or below. "
                       "Consider the task failed if the fingers loose contact with the vial, or the vial's distance in the x-y plane from its loading position is more than 0.04m.")
dicts["CentrifugeInsertion"] = "centrifuge_insertion.yaml"

class CentrifugeInsertion(YumiLeftEnv_objects_in_scene):
    def reward_fun(self, observation, action):
        vial_id = self.objects_id['vial']
        vial_pose = np.array(sim_utils.getObjPose(vial_id))
        centrifuge_id = self.objects_id['centrifuge']
        distance = np.linalg.norm(np.array(vial_pose[:2]) - np.array(self.objects_dict['vial']['load_pos'][:2]))

        finger_1, finger_2, total_contact_force = self.check_fingers_touching(vial_id)

        rewards_dict = {
            "finger_contacts_reward":  1 * finger_1 + 1 * finger_2,
            "vial_height_penalty":  -10 * vial_pose[2] ,
            "vial_xy_movement_penalty": -10 * distance
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        vial_id = self.objects_id['vial']
        vial_pose = np.array(sim_utils.getObjPose(vial_id))

        success_condition = vial_pose[2] <= 0.08
        failure_condition = not self.termination_condition()

        return {'task_solved': success_condition and not failure_condition}

    def termination_condition(self):
        vial_id = self.objects_id['vial']
        vial_pose = np.array(sim_utils.getObjPose(vial_id))
        initial_vial_pos = np.array(self.objects_dict['vial']['load_pos'])

        fail_cond_xy_distance = np.linalg.norm(np.array(vial_pose[:2]) - np.array(initial_vial_pos[:2])) > 0.04
        fail_cond_touching = not self.check_fingers_touching(vial_id)[0] or not self.check_fingers_touching(vial_id)[1]

        return fail_cond_xy_distance or fail_cond_touching


class CentrifugeInsertion_baseline(CentrifugeInsertion):
    """
        Baseline with only distance based reward
        Adapted from
        S. Levine, N. Wagener, P. Abbeel, "Learning Contact-Rich Manipulation Skills with Guided Policy Search,"
        in International Conference on Robotics and Automation (ICRA), 2015
    """
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(CentrifugeInsertion_baseline, self).__init__(render_mode, env_config_dict, control_mode)

        load_position = np.array(self.objects_dict['vial']['load_pos'])
        target_position = copy.deepcopy(load_position)
        target_position[2] = 0.08

        d_0 = np.linalg.norm(target_position[2] - load_position[2])

        self.r = get_dist_reward(d_0)
        self.cube_target_position = target_position

    def reward_fun(self, observation, action):
        vial_position = np.array(sim_utils.getObjPose(self.objects_id['vial']))[:3]

        d = np.linalg.norm(vial_position[2] - self.cube_target_position[2])

        reward = self.r(d)
        # print("reward", reward)
        rewards_dict = {"dist_reward": reward}

        return reward, rewards_dict





prompts["CubeStack"] = ("The robot gripper is holding a blue cube, and a red cube is placed on a surface very close to the gripper. "
                        "Do not drop the clue cube, keep the gripper fingers in contact with the blue cube. Place the blue cube on top of the red cube. "
                        "Consider the task completed when the distance between the two cubes in the x-y plane is less than 0.005m, the absolute difference between the two cubes height is less or equal 0.0255m, the red cube is within 0.005m from its loading position. "
                        "Consider the task failed if the gripper looses contact with the blue cube or the red cube is moved from its loading position of 0.005m or more, or the two cubes are further than at loading time.")
dicts["CubeStack"] = "two_cubes_stack.yaml"

class CubeStack(YumiLeftEnv_objects_in_scene):
    def reward_fun(self, observation, action):
        # Initialize reward and reward components dictionary
        reward = 0.0
        rewards_dict = {}

        # Calculate the pose for the blue and red cubes
        blue_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))
        red_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['red_cube']))

        # Get the position and orientation of the left end effector
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))

        # Check if the robot's fingers are touching the blue cube
        finger_1, finger_2, total_contact_force = self.check_fingers_touching(self.objects_id['blue_cube'])

        # Compute distance between the two cubes in the x-y plane and the height difference
        xy_distance = np.linalg.norm(blue_cube_pose[:2] - red_cube_pose[:2])
        height_diff = np.abs(blue_cube_pose[2] - red_cube_pose[2])

        # Compute the distance of the red cube from its loading position
        distance_red_cube_from_load_pos = np.linalg.norm(red_cube_pose[:3] - self.objects_dict['red_cube']['load_pos'])

        # Define the reward components
        rewards_dict = {
            "distance_to_red_cube_reward": -10 * xy_distance,  # manual
            "height_diff_reward": -10 * height_diff,  # manual
            "finger_contacts_reward": int(finger_1) + int(finger_2),
            "distance_red_cube_from_load_pos_penalty": -distance_red_cube_from_load_pos if distance_red_cube_from_load_pos > 0.005 else 0
        }

        # Calculate total bonuses and shaping
        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])

        # Access the information on whether the task has been solved
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        # Calculate total reward
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        # Return reward and reward components dictionary
        return reward, rewards_dict

    def _get_info(self):
        # Calculate the pose for the blue and red cubes
        blue_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))
        red_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['red_cube']))

        # Compute distance between the two cubes in the x-y plane and the height difference
        xy_distance = np.linalg.norm(blue_cube_pose[:2] - red_cube_pose[:2])
        height_diff = np.abs(blue_cube_pose[2] - red_cube_pose[2])

        # Compute the distance of the red cube from its loading position
        distance_red_cube_from_load_pos = np.linalg.norm(
            red_cube_pose[:2] - self.objects_dict['red_cube']['load_pos'][:2])

        task_solved = xy_distance < 0.005 and height_diff <= 0.0255 and distance_red_cube_from_load_pos < 0.005
        return {'task_solved': task_solved}

    def termination_condition(self):
        # Check if the robot's fingers are touching the blue cube
        finger_1, finger_2, _ = self.check_fingers_touching(self.objects_id['blue_cube'])

        blue_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))
        red_cube_pose = np.array(sim_utils.getObjPose(self.objects_id['red_cube']))

        ditst_init = np.linalg.norm(
            np.array(self.objects_dict['red_cube']['load_pos']) - self.objects_dict['blue_cube']['load_pos'])
        dist = np.linalg.norm(blue_cube_pose[:3] - red_cube_pose[:3])

        # Consider termination if the fingers are not touching the blue cube
        return not (finger_1 and finger_2) or dist > ditst_init


class CubeStack_baseline(CubeStack):
    """
        Baseline with only distance based reward
        Adapted from
        S. Levine, N. Wagener, P. Abbeel, "Learning Contact-Rich Manipulation Skills with Guided Policy Search,"
        in International Conference on Robotics and Automation (ICRA), 2015
    """
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(CubeStack_baseline, self).__init__(render_mode, env_config_dict, control_mode)

        blue_cube_load_position = np.array(self.objects_dict['blue_cube']['load_pos'])
        red_cube_load_position = np.array(self.objects_dict['red_cube']['load_pos'])
        self.target_position = red_cube_load_position + np.array([0.0, 0.0, 0.0255])

        d_0 = np.linalg.norm(blue_cube_load_position[:3] - self.target_position[:3])

        self.r = get_dist_reward(d_0)

    def reward_fun(self, observation, action):
        cube_position = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))[:3]

        d = np.linalg.norm(cube_position[:3] - self.target_position[:3])

        reward = self.r(d)
        rewards_dict = {"dist_reward": reward}

        return reward, rewards_dict


prompts["CubesTowerSlide"] = ("The robot gripper is holding a blue cube, which is stacked on top of a red cube. "
                              "Let go of the blue cube, move the tower of cubes to the right (negative direction of the y axis) of 0.1 meters, by touching the red cube. "
                              "Avoid touching the blue cube. Consider the task solved if both cubes are moved to the right of 0.1 meters from their loading position. "
                              "Consider the task failed if the x-y distance between the cubes is > 0.01 meters, or the distance between end effector and the red cube is > 0.05 meters.")
dicts["CubesTowerSlide"] = "cubes_tower.yaml"

class CubesTowerSlide(YumiLeftEnv_objects_in_scene):

    def reward_fun(self, observation, action):
        rewards_dict = {} 

        blue_cube_pos = np.array(sim_utils.getObjPose(self.objects_id["blue_cube"]))
        red_cube_pos = np.array(sim_utils.getObjPose(self.objects_id["red_cube"]))
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))

        distance_blue_red = np.linalg.norm(blue_cube_pos[:2] - red_cube_pos[:2])
        distance_effector_red = np.linalg.norm(end_effector_left_pose[:2] - red_cube_pos[:2])

        rewards_dict["distance_blue_red_penalty"] = -1000 if distance_blue_red > 0.01 else 0
        rewards_dict["distance_effector_red_penalty"] = -1000 if distance_effector_red > 0.05 else 0
        rewards_dict["move_right_bonus"] = (
            1000 * (self.objects_dict["blue_cube"]["load_pos"][1] - blue_cube_pos[1]) +
            1000 * (self.objects_dict["red_cube"]["load_pos"][1] - red_cube_pos[1]))

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        blue_cube_pos = np.array(sim_utils.getObjPose(self.objects_id["blue_cube"]))
        red_cube_pos = np.array(sim_utils.getObjPose(self.objects_id["red_cube"]))

        distance_blue_red = np.linalg.norm(blue_cube_pos[:2] - red_cube_pos[:2])
        task_solved = (
            distance_blue_red <= 0.01 and
            self.objects_dict["blue_cube"]["load_pos"][1] - blue_cube_pos[1] > 0.1 and
            self.objects_dict["red_cube"]["load_pos"][1] - red_cube_pos[1] > 0.1)

        return {'task_solved': task_solved}

    def termination_condition(self):
        blue_cube_pos = np.array(sim_utils.getObjPose(self.objects_id["blue_cube"]))
        red_cube_pos = np.array(sim_utils.getObjPose(self.objects_id["red_cube"]))
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))

        distance_blue_red = np.linalg.norm(blue_cube_pos[:2] - red_cube_pos[:2])
        distance_effector_red = np.linalg.norm(end_effector_left_pose[:2] - red_cube_pos[:2])

        return distance_blue_red > 0.01 or distance_effector_red > 0.05

class CubesTowerSlide_baseline(CubesTowerSlide):
    """
        Baseline with only distance based reward
        Adapted from
        S. Levine, N. Wagener, P. Abbeel, "Learning Contact-Rich Manipulation Skills with Guided Policy Search,"
        in International Conference on Robotics and Automation (ICRA), 2015
    """
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(CubesTowerSlide_baseline, self).__init__(render_mode, env_config_dict, control_mode)

        blue_cube_load_position = np.array(self.objects_dict['blue_cube']['load_pos'])
        red_cube_load_position = np.array(self.objects_dict['red_cube']['load_pos'])
        self.red_cube_target_position = red_cube_load_position + np.array([0.0, -0.1, 0.0])
        self.blue_cube_target_position = red_cube_load_position + np.array([0.0, -0.1, 0.0])

        d_0_blue_cube = np.linalg.norm(blue_cube_load_position[1] - self.blue_cube_target_position[1])
        d_0_red_cube = np.linalg.norm(red_cube_load_position[1] - self.red_cube_target_position[1])

        self.r_blue_cube = get_dist_reward(d_0_blue_cube)
        self.r_red_cube = get_dist_reward(d_0_red_cube)

    def reward_fun(self, observation, action):
        blue_cube_position = np.array(sim_utils.getObjPose(self.objects_id['blue_cube']))[:3]
        red_cube_position = np.array(sim_utils.getObjPose(self.objects_id['red_cube']))[:3]

        d_blue = np.linalg.norm(blue_cube_position[1] - self.blue_cube_target_position[1])
        d_red = np.linalg.norm(red_cube_position[1] - self.red_cube_target_position[1])

        reward = 0.5 * (self.r_blue_cube(d_blue) + self.r_red_cube(d_red))
        rewards_dict = {"dist_reward": reward}

        return reward, rewards_dict


prompts["BimanualBoxLift"] = ("The robot has its left and right end effectors over a spaghetti box. "
                              "Reach the spaghetti box with both grippers and grasp it using all fingers of both grippers. "
                              "Consider the task partially solved when all four fingers are touching the box. "
                              "Consider the task solved if the spaghetti box is lifted of 0.1m over its loading height, while all fingers are in contact with it."
                              "Consider the task failed when either end effector is further from the spaghetti box than 0.2 meters.")
dicts["BimanualBoxLift"] = "bimanual_spaghetti_box.yaml"

class BimanualBoxLift(YumiEnv_objects_in_scene):

    def reward_fun(self, observation, action):
        rewards_dict = {}  # Defines the reward components

        spaghetti_box_id = self.objects_id['spaghetti_box']
        spaghetti_box_pose = np.array(sim_utils.getObjPose(spaghetti_box_id))

        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))
        end_effector_right_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_right_idx))

        left_distance = np.linalg.norm(end_effector_left_pose[:3] - spaghetti_box_pose[:3])
        right_distance = np.linalg.norm(end_effector_right_pose[:3] - spaghetti_box_pose[:3])

        left_finger_1, left_finger_2, right_finger_1, right_finger_2 = self.check_fingers_touching(spaghetti_box_id)
        rewards_dict['left_distance_reward'] = -left_distance
        rewards_dict['right_distance_reward'] = -right_distance
        rewards_dict['fingers_contacts_reward'] = 1 * (left_finger_1 + left_finger_2 + right_finger_1 + right_finger_2)
        rewards_dict['grasp_reward'] = 10 * (left_finger_1 and left_finger_2 and right_finger_1 and right_finger_2)
        h = (spaghetti_box_pose[2] - self.objects_dict['spaghetti_box']['load_pos'][2])
        rewards_dict['spaghetti_box_height_reward'] = 100 * h if h > 0 else 0

        rewards_dict['orient_reward'] = - np.linalg.norm(spaghetti_box_pose[3:])

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        spaghetti_box_id = self.objects_id['spaghetti_box']
        spaghetti_box_pose = np.array(sim_utils.getObjPose(spaghetti_box_id))
        left_finger_1, left_finger_2, right_finger_1, right_finger_2 = self.check_fingers_touching(spaghetti_box_id)

        return {'task_solved': (spaghetti_box_pose[2] > self.objects_dict['spaghetti_box']['load_pos'][2] + 0.1 and
                                left_finger_1 and left_finger_2 and right_finger_1 and right_finger_2)}
                              # and np.linalg.norm(spaghetti_box_pose[3:]) < 0.5}

    def termination_condition(self):
        spaghetti_box_id = self.objects_id['spaghetti_box']
        spaghetti_box_pose = np.array(sim_utils.getObjPose(spaghetti_box_id))

        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))
        end_effector_right_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_right_idx))

        left_distance = np.linalg.norm(end_effector_left_pose[:3] - spaghetti_box_pose[:3])
        right_distance = np.linalg.norm(end_effector_right_pose[:3] - spaghetti_box_pose[:3])

        return left_distance > 0.2 or right_distance > 0.2


class BimanualBoxLift_baseline(BimanualBoxLift):
    """
        Baseline with only distance based reward
        Adapted from
        S. Levine, N. Wagener, P. Abbeel, "Learning Contact-Rich Manipulation Skills with Guided Policy Search,"
        in International Conference on Robotics and Automation (ICRA), 2015
    """
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(BimanualBoxLift_baseline, self).__init__(render_mode, env_config_dict, control_mode)

        box_load_position = np.array(self.objects_dict['spaghetti_box']['load_pos'])
        self.box_target_position = box_load_position + np.array([0.0, 0.0, 0.1])
        d_0 = np.linalg.norm(self.box_target_position[:3] - box_load_position[:3])

        self.r = get_dist_reward(d_0)

    def reward_fun(self, observation, action):
        box_position = np.array(sim_utils.getObjPose(self.objects_id['spaghetti_box']))[:3]

        d = np.linalg.norm(box_position - self.box_target_position)

        reward = self.r(d)
        rewards_dict = {"dist_reward": reward}

        return reward, rewards_dict


prompts["BimanualPenInTape"] = ("The robot is holding a pen with its left gripper, and a tape with its right gripper. The tape has a hole in the middle. "
                                "Keep the tape grasped by the right gripper, do not loose contact between the right gripper fingers and the cup. The same for the pen and the left gripper. "
                                "Insert the pen into the hole of the tape. "
                                "Consider the task solved when both objects are grasped, and the pen is at a distance from the tape of 0.005 or less. "
                                "Consider the task failed if no finger of the right gripper is in contact with the tape, or no finger of the left gripper is in contact with the pen, or the distance between the objects in the horizontal plane is more than 0.025 meters.")
dicts["BimanualPenInTape"] = "bimanual_pen_tape.yaml"
class BimanualPenInTape(YumiEnv_objects_in_scene):
    def reward_fun(self, observation, action):
        # Define an empty dictionary for rewards
        rewards_dict = {}

        # Get pose of tape and pen
        tape_pose = np.array(sim_utils.getObjPose(self.objects_id["tape"]))
        pen_pose = np.array(sim_utils.getObjPose(self.objects_id["pen"]))

        # Get the distance between the pen and the tape
        distance = np.linalg.norm(tape_pose[:3] - pen_pose[:3])

        # Check the contacts of the fingers with tape and pen
        left_finger_1, left_finger_2, _, _ = self.check_fingers_touching(self.objects_id["pen"])
        _, _, right_finger_1, right_finger_2 = self.check_fingers_touching(self.objects_id["tape"])

        # Reward/punish different aspects of the task
        rewards_dict["distance_to_tape_reward"] = - 10 * distance
        rewards_dict["left_gripper_contacts_reward"] = 1 * left_finger_1 + 1 * left_finger_2
        rewards_dict["right_gripper_contacts_reward"] = 1 * right_finger_1 + 1 * right_finger_2

        # If distance is greater than 0.25 meters, give a negative reward
        rewards_dict["distance_penalty"] = -10 if distance > 0.25 else 0.0

        # Calculate total bonuses
        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1

        # Get task solved information and calculate total shaping
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        
        # Define reward when task is solved
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        # Sum total shaping and task solved reward
        reward = total_shaping + task_solved_reward

        # Add task solved reward to rewards dictionary
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        # Get the pose of the left and right arms
        tape_pose = np.array(sim_utils.getObjPose(self.objects_id["tape"]))
        pen_pose = np.array(sim_utils.getObjPose(self.objects_id["pen"]))

        # Check finger contacts with pen and tape
        left_finger_1, left_finger_2, _, _ = self.check_fingers_touching(self.objects_id["pen"])
        _, _, right_finger_1, right_finger_2 = self.check_fingers_touching(self.objects_id["tape"])

        # Find distance between pen and tape
        distance = np.linalg.norm(tape_pose[:3] - pen_pose[:3])

        # If all conditions are met, consider the task solved
        task_solved = (left_finger_1 and left_finger_2 and right_finger_1 and right_finger_2 and distance <= 0.005)


        return {"task_solved": task_solved}

    def termination_condition(self):
        # Get pose of tape and pen
        tape_pose = np.array(sim_utils.getObjPose(self.objects_id["tape"]))
        pen_pose = np.array(sim_utils.getObjPose(self.objects_id["pen"]))

        # Check finger contacts with pen and tape
        left_finger_1, left_finger_2, _, _ = self.check_fingers_touching(self.objects_id["pen"])
        _, _, right_finger_1, right_finger_2 = self.check_fingers_touching(self.objects_id["tape"])

        # Calculate distance between pen and tape
        distance = np.linalg.norm(tape_pose[:2] - pen_pose[:2])

        # If any condition for failure is met, terminate
        if not (left_finger_1 and left_finger_2) or not (right_finger_1 and right_finger_2) or distance > 0.025:
            return True
        
        return False


class BimanualPenInTape_baseline(BimanualPenInTape):
    """
        Baseline with only distance based reward
        Adapted from
        S. Levine, N. Wagener, P. Abbeel, "Learning Contact-Rich Manipulation Skills with Guided Policy Search,"
        in International Conference on Robotics and Automation (ICRA), 2015
    """
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(BimanualPenInTape_baseline, self).__init__(render_mode, env_config_dict, control_mode)

        pen_load_position = np.array(self.objects_dict['pen']['load_pos'])
        tape_load_position = np.array(self.objects_dict['tape']['load_pos'])

        d_0 = np.linalg.norm(pen_load_position[:3] - tape_load_position[:3])

        self.r = get_dist_reward(d_0)

    def reward_fun(self, observation, action):
        cube_position = np.array(sim_utils.getObjPose(self.objects_id['pen']))[:3]
        cup_position = np.array(sim_utils.getObjPose(self.objects_id['tape']))[:3]

        d = np.linalg.norm(cube_position[:3] - cup_position[:3])

        reward = self.r(d)
        rewards_dict = {"dist_reward": reward}

        return reward, rewards_dict

"""
    1st iteration of the Vial Handover task (bi manual)
    The prompt used has a success condition that is not correct to make the receiving gripper to actually hold the object.
"""
prompts["VialHandover_failed"] = ("The robot is holding a vial with its left gripper. Keep the vial grasped by the left gripper, do not loose contact between the left gripper fingers and the vial. "
                                  "Move the right gripper close to the vial, grasp the vial with the right gripper, and only then release the vial with the left gripper. "
                                  "Consider the task solved when the vial is grasped by both gripper fingers of the right gripper, and the left gripper's fingers are not touching it. "
                                  "Consider the task failed if no finger is in contact with the vial, or the distance between the end effectors is more than 0.3 meters.")
dicts["VialHandover_failed"] = "bimanual_vial.yaml"

class VialHandover_failed(YumiEnv_objects_in_scene):

    def reward_fun(self, observation, action):
        vial_pose = np.array(sim_utils.getObjPose(self.objects_id["vial_with_cap"]))
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))
        end_effector_right_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_right_idx))
        left_finger_1, left_finger_2, right_finger_1, right_finger_2 = self.check_fingers_touching(
            self.objects_id["vial_with_cap"])

        # if right_finger_1:
        #     print("right_finger_1")
        # if right_finger_2:
        #     print("right_finger_2")

        rewards_dict = {
            "distance_right_to_vial_reward": -np.linalg.norm(vial_pose[:3] - end_effector_right_pose[:3]),
            # encourage right hand getting close to the vial
            "distance_grippers_reward": -np.abs(
                np.linalg.norm(end_effector_left_pose[:3] - end_effector_right_pose[:3]) - 0.4),
            # encourage hands getting close
            "left_fingers_contact_reward": 1 * left_finger_1 + 1 * left_finger_2,
            # encourage left hand in contact with the vial
            "right_fingers_contact_reward": 5 * right_finger_1 + 5 * right_finger_2,
            # encourage right hand in contact with the vial
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward
        rewards_dict["task_solved_reward"] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        left_finger_1, left_finger_2, right_finger_1, right_finger_2 = self.check_fingers_touching(
            self.objects_id["vial_with_cap"])
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))
        end_effector_right_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_right_idx))

        left_grip = left_finger_1 and left_finger_2
        right_grip = right_finger_1 and right_finger_2
        okay_distance = np.linalg.norm(end_effector_left_pose[:3] - end_effector_right_pose[:3]) <= 0.4

        return {'task_solved': right_grip and not left_grip and okay_distance}

    def termination_condition(self):
        left_finger_1, left_finger_2, _, _ = self.check_fingers_touching(self.objects_id["vial_with_cap"])
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))
        end_effector_right_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_right_idx))

        return not (left_finger_1 or left_finger_2) or np.linalg.norm(
            end_effector_left_pose[:3] - end_effector_right_pose[:3]) > 0.3


"""
    Handover was repeated with a box object.
"""
prompts["BimanualHandover"] = ("The robot is holding a box with its left gripper. Keep the box grasped by the left gripper, do not loose contact between the left gripper fingers and the box. "
                               "Move the right gripper close to the box, grasp the box with the right gripper, and only then release the box with the left gripper. "
                               "Consider the task solved when the box is grasped by both gripper fingers of the right gripper, and the left gripper's fingers have not been touching it for 50 time steps. "
                               "Consider the task failed if no finger is in contact with the box, or the distance between the end effectors is more than 0.4 meters.")
dicts["BimanualHandover"] = "bimanual_small_box.yaml"

class BimanualHandover(YumiEnv_objects_in_scene):
    def reward_fun(self, observation, action):

        obj_pose = np.array(sim_utils.getObjPose(self.objects_id['box']))

        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))
        end_effector_right_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_right_idx))

        distance = np.linalg.norm(obj_pose[:3] - end_effector_right_pose[:3])

        left_finger_1, left_finger_2, right_finger_1, right_finger_2 = self.check_fingers_touching(self.objects_id['box'])

        rewards_dict = {
            "distance_to_box_reward": -distance, # Encourage getting close to the box
            "left_finger_touch_reward": 0.5 * (left_finger_1 + left_finger_2), # Reward for holding box
            "right_finger_touch_bonus": 2 * (right_finger_1 + right_finger_2), # Reward for grasping with right gripper
            "box_drop_penalty": -10 if not any([left_finger_1, left_finger_2, right_finger_1, right_finger_2]) else 0, # Penalty for dropping box
            "end_effectors_distance_penalty": -10 if distance > 0.4 else 0, # Penalty for large distance between end effectors.
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        left_finger_1, left_finger_2, right_finger_1, right_finger_2 = self.check_fingers_touching(self.objects_id['box'])
        task_solved = right_finger_1 and right_finger_2 and not any([left_finger_1, left_finger_2]) and self.memory.get('count', 0) >= 50

        return {'task_solved': task_solved}

    def termination_condition(self):
        left_finger_1, left_finger_2, right_finger_1, right_finger_2 = self.check_fingers_touching(self.objects_id['box'])
        dropped_box = not any([left_finger_1, left_finger_2, right_finger_1, right_finger_2])
        if dropped_box:
            return True
        if right_finger_1 and right_finger_2 and not any([left_finger_1, left_finger_2]):
            self.memory['count'] = self.memory.get('count', 0) + 1

        obj_pose = np.array(sim_utils.getObjPose(self.objects_id['box']))
        end_effector_right_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_right_idx))
        distance = np.linalg.norm(obj_pose[:3] - end_effector_right_pose[:3])
        if distance > 0.4:
            return True

        return False

class BimanualHandover_baseline(BimanualHandover):
    """
        Baseline with only distance based reward
        Adapted from
        S. Levine, N. Wagener, P. Abbeel, "Learning Contact-Rich Manipulation Skills with Guided Policy Search,"
        in International Conference on Robotics and Automation (ICRA), 2015
    """
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(BimanualHandover_baseline, self).__init__(render_mode, env_config_dict, control_mode)

        box_load_position = np.array(self.objects_dict['box']['load_pos'])
        end_effector_right_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_right_idx))

        d_0 = np.linalg.norm(end_effector_right_pose[:3] - box_load_position)

        self.r = get_dist_reward(d_0)

    def reward_fun(self, observation, action):
        box_position = np.array(sim_utils.getObjPose(self.objects_id['box']))[:3]
        end_effector_right_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_right_idx))

        d = np.linalg.norm(end_effector_right_pose[:3] - box_position[:3])

        _, _, right_finger_1, right_finger_2 = self.check_fingers_touching(self.objects_id['box'])


        rewards_dict = {"dist_reward": self.r(d),
                        "right_finger_touch_bonus": 5 * (right_finger_1 + right_finger_2)}
        # only exception since it is a difficult task

        reward = rewards_dict['dist_reward'] + rewards_dict['right_finger_touch_bonus']


        return reward, rewards_dict



prompts["CubeInCup"] = ("The robot is holding a cube with its left gripper, and a cup with its right gripper. "
                        "Keep the cup grasped by the right gripper, do not loose contact between the right gripper fingers and the cup. Place the cube into the cup. "
                        "Consider the task solved when the cup is grasped by the gripper fingers of the right gripper, the cube is at a distance from the cup of 0.025 or less. "
                        "Consider the task failed if no finger of the right gripper is in contact with the cup, or the distance between the left end effector and the cube is more than 0.2 meters.")
dicts["CubeInCup"] = "bimanual_cube_cup.yaml"

class CubeInCup(YumiEnv_objects_in_scene):
    def reward_fun(self, observation, action):
        rewards_dict = {}

        cube_id = self.objects_id["cube"]
        cup_id = self.objects_id["cup"]

        cube_pose = np.array(sim_utils.getObjPose(cube_id))
        cup_pose = np.array(sim_utils.getObjPose(cup_id))

        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))

        cup_cube_distance = np.linalg.norm(cube_pose[:3] - cup_pose[:3])
        left_effector_cube_distance = np.linalg.norm(cube_pose[:3] - end_effector_left_pose[:3])

        _, _, right_finger_1, right_finger_2 = self.check_fingers_touching(cup_id)

        rewards_dict = {
            "cube_cup_distance_reward": -cup_cube_distance,
            "cube_left_effector_distance_penalty": -left_effector_cube_distance,
            "gripper_contact_reward": 1 * right_finger_1 + 1 * right_finger_2
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward
        return reward, rewards_dict

    def _get_info(self):
        cube_id = self.objects_id["cube"]
        cup_id = self.objects_id["cup"]

        cube_pose = np.array(sim_utils.getObjPose(cube_id))
        cup_pose = np.array(sim_utils.getObjPose(cup_id))

        cup_cube_distance = np.linalg.norm(cube_pose[:3] - cup_pose[:3])
        _, _, right_finger_1, right_finger_2 = self.check_fingers_touching(cup_id)

        task_solved = cup_cube_distance <= 0.025 and (right_finger_1 or right_finger_2)
        return {'task_solved': task_solved}

    def termination_condition(self):
        cube_id = self.objects_id["cube"]
        cup_id = self.objects_id["cup"]
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))
        cube_pose = np.array(sim_utils.getObjPose(cube_id))
        _, _, right_finger_1, right_finger_2 = self.check_fingers_touching(cup_id)

        left_effector_cube_distance = np.linalg.norm(cube_pose[:3] - end_effector_left_pose[:3])
        gripper_cup_contact = (right_finger_1 or right_finger_2)

        return left_effector_cube_distance > 0.2 or not gripper_cup_contact


class CubeInCup_baseline(CubeInCup):
    """
        Baseline with only distance based reward
        Adapted from
        S. Levine, N. Wagener, P. Abbeel, "Learning Contact-Rich Manipulation Skills with Guided Policy Search,"
        in International Conference on Robotics and Automation (ICRA), 2015
    """
    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        super(CubeInCup_baseline, self).__init__(render_mode, env_config_dict, control_mode)

        cube_load_position = np.array(self.objects_dict['cube']['load_pos'])
        cup_load_position = np.array(self.objects_dict['cup']['load_pos'])

        d_0 = np.linalg.norm(cube_load_position[:3] - cup_load_position[:3])

        self.r = get_dist_reward(d_0)

    def reward_fun(self, observation, action):
        cube_position = np.array(sim_utils.getObjPose(self.objects_id['cube']))[:3]
        cup_position = np.array(sim_utils.getObjPose(self.objects_id['cup']))[:3]

        d = np.linalg.norm(cube_position[:3] - cup_position[:3])

        reward = self.r(d)
        rewards_dict = {"dist_reward": reward}

        return reward, rewards_dict


