import numpy as np

from environments import *


class GeneratedEnv_2DGraspEnv_0(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        info = self._get_info()

        # define variables using observations
        cube_position = observation[4:6]
        grasped = observation[6]
        gripper_position = observation[:2]

        # Calculate distance between gripper and cube
        distance_to_cube = np.linalg.norm(gripper_position - cube_position)

        rewards_dict = {
            "distance_to_cube_reward": -distance_to_cube,  # reward getting closer to the cube
            "grasp_reward": 1.0 if grasped else 0.0,  # reward gripping the cube
            "lift_reward": cube_position[1] if grasped else 0.0,
            # reward lifting the cube, assuming y is the vertical axis
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        cube_height = self.c[1]  # based on the assumption that y is the vertical axis
        return {'task_solved': cube_height >= 0.5}  # task is solved if cube height is 0.5 or above


class GeneratedEnv_2DGraspEnv_1(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        gripper_pose = observation[:2]
        cube_pose = observation[4:6]
        absolute_distance = np.linalg.norm(np.array(gripper_pose) - np.array(cube_pose))
        gripper_speed = np.linalg.norm(observation[2:4])
        grasped = observation[6]

        rewards_dict = {
            "distance_to_cube_reward": -absolute_distance,  # reward for getting closer to the cube
            "gripper_speed_penalty": -gripper_speed / 10,  # discourage unnecessary movements
            "grasping_reward": 5 * grasped,  # large reward for successfully grasping the cube
            "lifting_reward": 5 * cube_pose[1],  # reward for lifting the cube
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
        cube_pose = self.c
        return {'task_solved': cube_pose[1] >= 0.5}

    def termination_condition(self):
        cube_pose = self.c
        if cube_pose[1] < 0:  # if cube falls below a certain height, terminate the task as unsuccessful
            return True
        else:
            return super().termination_condition()  # if not, call termination_condition from the superclass


class GeneratedEnv_2DGraspEnv_2(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        rewards_dict = {}

        p = observation[:2]  # agent's position
        p_dot = observation[2:4]  # agent's velocity
        c = observation[4:6]  # cube's position
        grasped = observation[6]  # whether the cube is grasped

        # Distance between agent and cube
        distance = np.linalg.norm(np.array(p) - np.array(c))

        # Reward for reaching the cube
        rewards_dict["distance_to_cube_reward"] = -distance if not grasped else distance

        # Reward for grasping the cube
        rewards_dict["grasp_reward"] = 10.0 if grasped else -0.1

        # Reward for lifting the cube
        rewards_dict["lift_reward"] = c[1] if grasped else -c[1] * 0.1

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        task_solved = False
        cube_pose = self.c  # Cube position

        if self.grasped and cube_pose[1] >= 0.5:  # If the cube is grasped and lifted to height 0.5
            task_solved = True

        return {'task_solved': task_solved}


class GeneratedEnv_2DGraspEnv_3(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        rewards_dict = {}

        p = observation[:2]  # agent's position
        c = observation[4:6]  # cube's position
        grasped = bool(observation[6])

        distance = np.linalg.norm(p - c)  # distance between agent and cube
        height_reward = c[1]  # height of cube (encourages lifting)

        rewards_dict = {
            "distance_to_cube_reward": -distance,  # encourage getting close to the target cube
            "grasping_reward": 2 * int(grasped),  # encourage grasping of the cube
            "cube_height_reward": height_reward,  # encourage lifting of the cube
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
        cube_pose = self.c
        task_solved = cube_pose[1] > 0.5

        return {'task_solved': task_solved}


class GeneratedEnv_2DGraspEnv_4(SimplifiedGraspingEnv):
    def _get_info(self):
        return {'task_solved': self.c[1] >= 0.5}

    def reward_fun(self, observation, action):
        p = self.p
        c = self.c
        p_dot = self.p_dot
        grasped = self.grasped

        distance_to_cube = -np.linalg.norm(p - c)  # reward getting closer to the cube
        speed_penalty = -np.linalg.norm(p_dot)  # penalize movements
        grasp_reward = 10 * grasped  # reward grasping the cube
        lift_reward = 1000 * c[1]  # reward for lifting the cube up

        rewards_dict = {
            "distance_to_cube_reward": distance_to_cube,
            "speed_penalty": speed_penalty,
            "grasp_reward": grasp_reward,
            "lift_reward": lift_reward
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict


class GeneratedEnv_2DGraspEnv_5(SimplifiedGraspingEnv):

    def _get_info(self):
        cube_height = self.c[1]
        task_solved = cube_height >= 0.5
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        rewards_dict = {}

        cube_pose = self.c
        gripper_pose = self.p
        distance_to_cube = -np.linalg.norm(cube_pose - gripper_pose)

        rewards_dict['distance_to_cube_reward'] = distance_to_cube

        gripper_contact_with_cube = self.grasped
        rewards_dict['gripper_contact_with_cube_reward'] = gripper_contact_with_cube * 0.1

        cube_height = cube_pose[1]
        rewards_dict['cube_height_reward'] = cube_height * 10

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict


class GeneratedEnv_2DGraspEnv_6(SimplifiedGraspingEnv):

    def _get_info(self):
        # Task is considered solved if the cube is at height 0.5
        task_solved = self.c[1] >= 0.5
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        p = observation[:2]  # position of end effector
        c = observation[4:6]  # position of cube
        grasped = observation[6]

        rewards_dict = {}

        distance = np.linalg.norm(np.array(c[:2]) - np.array(p[:2]))
        rewards_dict["distance_to_cube"] = -distance

        # getting closer to the cube grants a bonus
        if np.linalg.norm(np.array(p[:2]) - np.array(c[:2])) <= 0.04:
            rewards_dict["close_to_cube_bonus"] = 1.0
        else:
            rewards_dict["close_to_cube_bonus"] = 0.0

        # grasping the cube grants a bonus
        if grasped:
            rewards_dict["grasping_bonus"] = 1.0
        else:
            rewards_dict["grasping_bonus"] = 0.0

        # lifting bonus, proportionate to height of cube when grasped
        if grasped:
            rewards_dict["lifting_bonus"] = c[1]
        else:
            rewards_dict["lifting_bonus"] = 0.0

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1

        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict


class GeneratedEnv_2DGraspEnv_7(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Get positions of the effector and the cube
        agent_pos = self.p
        cube_pos = self.c

        # Compute the Euclidean distance between the agent and the cube
        dist_to_cube = np.linalg.norm(np.array(agent_pos) - np.array(cube_pos))

        # Reward for distance to cube: encourage agent to get close to the cube
        rewards_dict["distance_to_cube_reward"] = -dist_to_cube

        # Reward for lifting the cube
        rewards_dict["lift_reward"] = cube_pos[1] if cube_pos[1] > 0.04 else 0.0

        # If the cube is grasped, give reward
        rewards_dict["grasping_reward"] = int(self.grasped)

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = self._max_episode_steps * total_bonuses * info['task_solved']

        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward

        rewards_dict["task_solved_reward"] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        cube_pose = self.c
        return {'task_solved': cube_pose[1] > 0.5}


class GeneratedEnv_2DGraspEnv_8(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        p, c, grasped = observation[:2], observation[4:6], observation[6]

        # distance between the agent and the cube
        dist_to_cube = np.linalg.norm(p - c)

        # whether the agent has grasped the cube or not
        grasp_reward = 1.0 if grasped else -0.1

        # reward for lifting the cube, we assume z axis is y in the current 2D setup
        lift_reward = 0.0
        if grasped:
            lift_reward = c[1]

        rewards_dict = {
            "dist_to_cube": -dist_to_cube,
            "grasp_reward": grasp_reward,
            "lift_reward": lift_reward,
        }

        info = self._get_info()
        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        """
        Returns a dictionary with information about the environment
        Must contain the boolean field task_solved
        """
        task_solved = self.grasped and (self.c[1] >= 0.5)  # task is solved when the cube is lifted to 0.5 height
        return {'task_solved': task_solved}


class GeneratedEnv_2DGraspEnv_9(SimplifiedGraspingEnv):
    def _get_info(self):
        """
        - Compute whether the task is accomplished.
        - 'task_solved' field in the info dictionary tells if the cube is lifted above 0.5 height.
        """
        info_dict = super()._get_info()
        info_dict['task_solved'] = self.c[1] >= 0.5
        return info_dict

    def reward_fun(self, observation, action):
        """
        Defines the dense reward function guided by several components:
        - Getting closer to the cube.
        - Grasping the cube.
        - Lifting the cube.
        """
        rewards_dict = {
            "distance_to_cube_reward": -np.linalg.norm(self.c - self.p),  # Negative distance (closer is better)
            "grasping_reward": 5 * self.grasped,  # Encourage the agent to grasp the cube
            "lifting_reward": 10 * self.c[1]  # Encourage the agent to lift the cube
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1

        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict


class GeneratedEnv_2DGraspEnv_NoTermReward_0(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        p = observation[:2]  # Position of the agent
        c = observation[4:6]  # Position of the cube
        grasped = observation[6]
        dist = np.linalg.norm(p - c)
        rewards_dict = {}
        rewards_dict['dist_reward'] = -dist  # encourage getting close to the cube
        rewards_dict['grasp_reward'] = 5.0 if grasped else 0.0  # Encourage grasping
        rewards_dict['lift_reward'] = c[1] + 1 if grasped else 0  # encourage lifting when grasped
        reward = sum(rewards_dict[k] for k in rewards_dict.keys())
        return reward, rewards_dict

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        task_solved = self.c[1] > 0.5
        return {'task_solved': task_solved}


class GeneratedEnv_2DGraspEnv_NoTermReward_1(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Calculate the distance to the cube
        distance_to_cube = np.linalg.norm(self.p - self.c)

        # Calculate the height of the cube. The higher the cube, the better.
        cube_height = self.c[1]

        # Define reward for reducing the distance to the cube
        rewards_dict["distance_to_cube_reward"] = -distance_to_cube

        # Cube height reward. If the cube height greater than 0.5, the task is solved.
        rewards_dict["cube_height_reward"] = cube_height if cube_height > 0.5 else -cube_height

        # Define bonus for grasping the cube
        rewards_dict["grasped_reward"] = 1.0 if self.grasped else -1.0

        # Sum all the reward components
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        # Check if the task is solved (cube height is greater than 0.5
        task_solved = self.c[1] > 0.5

        return {'task_solved': task_solved}


class GeneratedEnv_2DGraspEnv_NoTermReward_2(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        # Compute distance to the cube
        distance_to_cube = np.linalg.norm(self.p - self.c)

        # Compute reward components
        rewards_dict = {
            "distance_to_cube_reward": -distance_to_cube,
            "cube_grasped_reward": 1000.0 * int(self.grasped),
            "cube_height_reward": 1000.0 * self.c[1]
        }

        # Compute total reward
        total_reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return total_reward, rewards_dict

    def _get_info(self):
        # The task is solved when the cube is at height 0.5
        task_solved = self.c[1] >= 0.5
        return {"task_solved": task_solved}


class GeneratedEnv_2DGraspEnv_NoTermReward_3(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        rewards_dict = {}

        distance_to_cube = np.linalg.norm(observation[:2] - observation[4:6])
        is_grasping = observation[6]

        rewards_dict['distance_to_cube_reward'] = -distance_to_cube
        rewards_dict['grasping_reward'] = 1 if is_grasping else -0.1
        rewards_dict['height_reward'] = observation[4]

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        cube_height = self.c[1]

        return {'task_solved': cube_height >= 0.5}


class GeneratedEnv_2DGraspEnv_NoTermReward_4(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        rewards_dict = {}

        # reward for distance to the cube
        p = observation[:2]
        c = observation[4:6]
        distance = np.linalg.norm(np.array(p) - np.array(c))
        rewards_dict['distance_to_cube_reward'] = -distance

        # reward for grasping the cube
        grasped = bool(observation[6])
        rewards_dict['grasp_reward'] = 100 * grasped

        # reward for lifting the cube
        cube_height = c[1]
        rewards_dict['cube_height_reward'] = 100 * cube_height

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Contains a boolean field task_solved that indicates whether the cube is at 0.5 height
        """
        cube_pose = self.c[1]
        return {'task_solved': cube_pose >= 0.5}


class GeneratedEnv_2DGraspEnv_NoTermReward_5(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        """
        This function calculates reward based on the agent's performance.
        """

        info = self._get_info()

        # Shaping functions
        distance_to_cube = np.linalg.norm(observation[:2] - observation[4:6])
        height = observation[4:1]

        # Reward components
        rewards_dict = {
            "distance_to_cube": -distance_to_cube,  # Reward for being close to the cube
            "height": height if info['task_solved'] else 0.  # Reward for lifting the cube
        }

        # Total reward is the sum of the individual rewards
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        """
        This function provides additional information about the agent's current state.
        """
        cube_pose = self.c
        return {'task_solved': cube_pose[1] >= 0.5}  # Task is considered solved if the cube's height is >= 0.5


class GeneratedEnv_2DSlideEnv_0(SimplifiedGraspingEnv):
    # We will only redefine the _get_info and reward_fun methods
    # Since we do not redefine __init__, it will default to the parent class __init__

    def reward_fun(self, observation, action):
        # Cube details
        cube_pos = self.c
        cube_x_coordinate = cube_pos[0]

        # Agent details
        agent_pos = self.p
        agent_x_coordinate = agent_pos[0]

        # Distance to the cube
        distance_to_cube = np.linalg.norm(cube_pos - agent_pos)

        # Check if the cube is grasped
        cube_grasped = self.grasped

        # Defining the reward components
        rewards_dict = {
            "distance_to_cube_reward": -distance_to_cube,  # Encourage getting close to the cube
            "grasp_reward": cube_grasped * 5,  # Encourage grasping the cube
            "cube_right_reward": cube_x_coordinate * 10  # Encourage moving the cube to the right
        }

        # Task_solved_reward if the cube's x_coordinate is greater than 0.99
        info = self._get_info()
        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        # Cube details
        cube_pos = self.c
        cube_x_coordinate = cube_pos[0]

        task_solved = False
        # Check if the cube is to the right, cube's x_coordinate is greater than 0.99.
        if cube_x_coordinate > 0.99:
            task_solved = True

        return {"task_solved": task_solved}

    def termination_condition(self):
        # There is no termination condition for this task
        return False


class GeneratedEnv_2DSlideEnv_0(SimplifiedGraspingEnv):
    def reward_fun(self, obs, action):
        rewards_dict = {}

        agent_pos = obs[:2]
        cube_pos = obs[4:6]

        dist_to_cube = np.linalg.norm(np.array(agent_pos) - np.array(cube_pos))
        rewards_dict['distance_to_cube_reward'] = -dist_to_cube

        gripper_touching = obs[6]
        rewards_dict['gripper_touching'] = 1 if gripper_touching else 0

        rewards_dict['positive_x_direction'] = cube_pos[0] if cube_pos[0] > 0 else 0

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])

        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        obs = self._get_obs()
        task_solved = obs[4] > 0.99  # task is solved when the cube x coordinate is > 0.99
        return {'task_solved': task_solved}


class GeneratedEnv_2DSlideEnv_1(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        p = observation[:2]  # Position of the gripper
        c = observation[4:6]  # Position of the cube
        grasped = observation[6]  # If the gripper grasped the cube

        rewards_dict = {
            "distance_to_cube_reward": -np.linalg.norm(p - c),  # Encourage getting close to the cube
            "grasped_cube_reward": grasped,  # Encouraging keeping the cube grasped
            "moved_cube_right": max(0, action[0]) if grasped else 0  # Encouraging moving the cube to the right
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
        """
            Returns a dictionary with a boolean field task_solved, that is True when the cube's x coordinate
            is > 0.99 and False otherwise
        """
        return {'task_solved': self.c[0] > 0.99}

    def termination_condition(self):
        """
            No additional termination conditions are defined for this task as there is no failure condition
            according to the prompt instructions
        """
        return False


class GeneratedEnv_2DSlideEnv_NoTermReward_0(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        # create rewards_dict
        rewards_dict = {
            'dist_reward': -np.linalg.norm(self.c - self.p),
            'grasp_reward': float(self.grasped) * 500,
            'move_right_reward': self.c[0]
        }

        # sum rewards
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        return {'task_solved': self.c[0] > 0.99}


from environments import *


class GeneratedEnv_2DGraspEnv_NoTermReward_5(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        p, p_dot, c, grasped = observation[:2], observation[2:4], observation[4:6], observation[6]
        rewards_dict = {}

        # Calculate the distance from the effector to the cube
        distance = np.linalg.norm(p - c)

        # Calculate the reward based on distance, grasp status and height of the cube
        rewards_dict["distance_to_cube_reward"] = -distance  # Encourage getting close to the target cube
        rewards_dict["grasp_reward"] = 5 * int(grasped)  # Encourage interaction with the target cube
        rewards_dict["height_reward"] = 10 * c[1]  # Encourage lifting the cube

        # Compute the total reward
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        task_solved = float(self.c[1] >= 0.5)  # Task is solved when the cube is at 0.5 height or more
        return {'task_solved': bool(task_solved)}


from environments import *


class GeneratedEnv_2DSlideEnv_2(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        task_solved_reward = 0
        gripper_pos = observation[:2]
        gripper_vel = observation[2:4]
        cube_pos = observation[4:6]
        grasped = observation[6]

        rewards_dict = {
            "distance_to_cube_reward": -np.linalg.norm(gripper_pos[:2] - cube_pos[:2]),  # The negative distance to cube
            "gripper_speed_reward": -np.linalg.norm(gripper_vel),  # Penalty for speed
            "grasping_reward": grasped * 5,  # Encourage grasping the cube
            "cube_pos_reward": cube_pos[0] if cube_pos[0] > 0 else 0  # Reward for moving the cube to the right
        }

        info = self._get_info()
        task_solved_reward = 0
        if info['task_solved']:
            total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
            total_bonuses = total_bonuses if total_bonuses > 0 else 1
            task_solved_reward = 10 * self._max_episode_steps * total_bonuses
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        info = super()._get_info()
        cube_pos = self.c
        info['task_solved'] = cube_pos[0] > 0.99
        return info


from environments import *


class GeneratedEnv_2DSlideEnv_NoTermReward_1(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        p = observation[:2]  # position of the end effector
        c = observation[4:6]  # position of the cube
        grasped = observation[6]  # if the cube is grasped

        rewards_dict = {}

        # reward for getting close to the cube
        rewards_dict["distance_to_cube_reward"] = -np.linalg.norm(np.array(p) - np.array(c))

        # reward for grasping the cube
        rewards_dict["grasped_cube_reward"] = 5.0 if grasped else 0.0

        # reward for moving the cube to the right
        rewards_dict["cube_x_reward"] = c[0] if grasped else 0.0

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        cube_x = self.c[0]
        # the task is solved when the cube x coordinate is > 0.99
        return {'task_solved': cube_x > 0.99}


from environments import *


class GeneratedEnv_2DGraspEnv_NoTermReward_6(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Extract agent and cube positions
        p = observation[:2]
        c = observation[4:6]
        grasped = observation[6]

        # Calculate the distance between agent and cube
        distance = np.linalg.norm(np.array(c) - np.array(p))

        # Reward for getting close to the cube
        rewards_dict["distance_reward"] = 1.0 / (1.0 + distance)

        # Additional reward for lifting the cube
        if grasped and c[1] >= 0.5:
            rewards_dict["lifting_reward"] = 5.0

        # Total reward is the sum of the reward components
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        cube_height = self.c[1]
        return {'task_solved': cube_height >= 0.5}


from environments import *


class GeneratedEnv_2DSlideEnv_NoTermReward_2(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        """
        Reward function based on task description, includes partial task rewards.
        """
        info = self._get_info()

        reward_dict = {
            'grasp_cube': 1. if self.grasped else -1.,
            'distance_to_cube': -np.linalg.norm(self.p - self.c),
            'move_cube_right': self.c[0] if self.grasped else 0.
        }

        reward = sum([reward_dict[k] for k in reward_dict.keys()])

        return reward, reward_dict

    def _get_info(self):
        """
        Compute task completion status using the current state of the simulation.
        """
        task_solved = self.c[0] > 0.99
        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DSlideEnv_3(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        gripper_pos, gripper_vel, cube_pos, grasped = observation[:2], observation[2:4], observation[4:6], observation[
            6]

        info = self._get_info()

        distance_to_cube = np.linalg.norm(np.array(cube_pos) - np.array(gripper_pos))
        cube_moved_right = max(0, cube_pos[0])

        rewards_dict = {
            "distance_to_cube_reward": -distance_to_cube,
            "grasping_reward": 50 * grasped,
            "cube_moved_right_reward": 100 * cube_moved_right * int(self.grasped)
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        info_dict = super()._get_info()
        info_dict['task_solved'] = self.c[0] > 0.99
        return info_dict


from environments import *


class GeneratedEnv_2DSlideEnv_NoTermReward_3(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        p = observation[:2]
        p_dot = observation[2:4]
        c = observation[4:6]
        grasped = int(observation[6])

        # Defines the reward components
        rewards_dict = {
            "distance_to_cube_reward": -np.linalg.norm(p - c),  # encourage getting close to the cube
            "grasped_reward": 5 * grasped,  # encourage graping the cube
            "x_direction_reward": 2 * p_dot[0],  # encourage moving in positive x direction
            "y_direction_reward": -1 * abs(p_dot[1])  # discourage moving in y direction
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        task_solved = self.c[0] > 0.99
        return {'task_solved': task_solved}

    def termination_condition(self):
        return False


from environments import *


class GeneratedEnv_2DGraspEnv_NoTermReward_7(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        cube_position = observation[4:6]
        agent_position = observation[:2]
        distance_to_cube = np.linalg.norm(np.array(cube_position) - np.array(agent_position))
        is_grasped = observation[6]

        rewards_dict = {
            "distance_to_cube_reward": -distance_to_cube,  # incentivize getting closer to the cube
            "grasping_reward": 5 * is_grasped,  # incentivize grasping of the cube
            "cube_lifting_reward": 5 * cube_position[1]  # incentivize lifting the cube
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        cube_height = self.c[1]

        return {'task_solved': cube_height >= 0.5}
    # No failure condition to be defined in this case as per the prompt


from environments import *


class GeneratedEnv_2DSlideEnv_4(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):

        p = observation[:2]  # agent's position
        p_dot = observation[2:4]  # agent's velocity
        c = observation[4:6]  # cube's position
        grasped = observation[6]  # whether the cube is grasped
        info = self._get_info()
        # Reward for being close to the cube
        distance_to_cube = np.linalg.norm(np.array(p[:2]) - np.array(c[:2]))
        distance_reward = - distance_to_cube

        # Reward for grasping the cube
        if grasped and distance_to_cube < 0.04:
            grasp_reward = 3
        else:
            grasp_reward = 0

        # Reward for moving the cube to the right (positive x direction)
        if grasped and p_dot[0] > 0:
            move_cube_reward = 5
        else:
            move_cube_reward = 0

        rewards_dict = {
            "distance_reward": distance_reward,
            "grasp_reward": grasp_reward,
            "move_cube_reward": move_cube_reward,
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward
        return reward, rewards_dict

    def _get_info(self):
        c = copy.deepcopy(self.c)
        return {'task_solved': c[0] > 0.99}


from environments import *


class GeneratedEnv_2DSlideEnv_NoTermReward_4(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        info = self._get_info()

        rewards_dict = {
            "distance_to_cube_reward": -np.linalg.norm(observation[:2] - observation[4:6]),
            "grasping_reward": 5.0 * int(observation[6]),
            "cube_x_direction_reward": observation[4],
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        return {'task_solved': self.c[0] > 0.99}


from environments import *


class GeneratedEnv_2DGraspEnv_NoTermReward_8(SimplifiedGraspingEnv):
    def _get_info(self):
        cube_height = self.c[1]
        return {'task_solved': cube_height > 0.5}

    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Define reward components
        distance_to_cube_reward = -np.sqrt(np.sum(np.square(self.p - self.c)))
        cube_lifting_reward = self.c[1]
        # Encourage interaction with the cube and reward for maintaining the grip
        grasp_reward = 1.0 if self.grasped else -0.1
        # if the agent is holding the cube, give extra reward based on cube height
        lifting_reward = self.c[1] if self.grasped else 0.0

        rewards_dict = {
            "distance_to_cube_reward": distance_to_cube_reward,
            "grasp_reward": grasp_reward,
            "lifting_reward": lifting_reward
        }

        # Total reward is a sum of the components
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DSlideEnv_NoTermReward_5(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):

        rewards_dict = {}
        p, p_dot, c, grasped = observation[:2], observation[2:4], observation[4:6], observation[6]

        distance_to_cube = np.linalg.norm(p - c)
        if p[0] < 0:
            distance_to_goal = np.abs(p[0] - 0.99)
        else:
            distance_to_goal = 0

        rewards_dict["getting_closer_to_cube"] = -distance_to_cube if distance_to_cube > 0 else 0
        rewards_dict["getting_closer_to_goal"] = -distance_to_goal if distance_to_goal > 0 else 0
        rewards_dict["grasping_reward"] = 1000 if grasped else 0

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        info = super()._get_info()
        info['task_solved'] = self.c[0] > 0.99 and self.grasped
        return info

    def termination_condition(self):
        return False  # No failure condition for this task


from environments import *


class GeneratedEnv_2DGraspEnv_NoTermReward_9(SimplifiedGraspingEnv):
    def _get_info(self):
        # if c[1] (the height of the cube) reaches 0.5 and the cube is grasped, the task is solved
        return {'task_solved': self.c[1] >= 0.5 and self.grasped}

    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Reward positive y displacement of the cube
        rewards_dict["cube_height_reward"] = 1000 * self.c[1]

        # Reward getting close to the cube in 2D plane
        rewards_dict["distance_to_cube_reward"] = -np.linalg.norm(self.p - self.c[:2])

        # Reward grasping the cube
        rewards_dict["grasp_reward"] = 500 if self.grasped else 0

        # Sum up all the rewards
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DSlideEnv_5(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}
        p = observation[:2]
        c = observation[4:6]

        # Encourage getting close to the cube
        rewards_dict["distance_to_cube_reward"] = -np.linalg.norm(p - c)

        # Encourage the cube to move to the right
        rewards_dict["movement_right_reward"] = c[0]

        # Encourage grasping the cube
        rewards_dict["grasped_reward"] = 2 * observation[6]

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        cube_pose = self.c
        return {'task_solved': cube_pose[0] > 0.99}


from environments import *


class GeneratedEnv_2DSlideEnv_NoTermReward_6(SimplifiedGraspingEnv):
    def _get_info(self):
        task_solved = False
        if self.grasped and self.c[0] > 0.99:
            task_solved = True
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        reward_dict = {}
        reward_dict["distance_to_cube_reward"] = -np.linalg.norm(self.p - self.c)  # Reward getting closer to the cube
        reward_dict["grasping_reward"] = 10.0 * int(self.grasped)  # Reward if the agent has grasped the cube
        reward_dict["move_cube_right_reward"] = self.c[
                                                    0] * self.grasped  # Reward for moving the cube to the right only if the cube has been grasped

        reward = sum([reward_dict[k] for k in reward_dict.keys()])
        return reward, reward_dict


from environments import *


class GeneratedEnv_2DSlideEnv_6(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        rewards_dict = {}

        # position of agent
        p = observation[:2]
        # position of cube
        c = observation[4:6]
        # compute distance
        distance = np.linalg.norm(np.array(c) - np.array(p))
        # grasp status
        grasping = observation[6]

        rewards_dict = {
            "distance_to_cube_reward": -distance,  # encourage getting close to cube
            "grasping_cube_reward": 10 if grasping == 1 else 0,  # encourage grasping
            "cube_motion_right_reward": 4 * (c[0] - 0.0)  # encourage moving cube to the right
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
        current_state = self._get_obs()
        cube_x_position = current_state[4]
        return {'task_solved': cube_x_position > 0.99}


from environments import *


class GeneratedEnv_2DPlaceEnv_0(SimplifiedPlacingEnv):

    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Encourage agent to get close to target (0, 0)
        distance_to_target = np.linalg.norm(self.c - np.array([0.0, 0.0]))
        rewards_dict['distance_to_target_reward'] = -distance_to_target

        # Give a reward if the agent is grasping the cube
        rewards_dict['grasping_reward'] = self.grasped * 1

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        # Task is solved if cube is at 0.05 distance from (0, 0) and is grasped
        task_solved = (np.linalg.norm(self.c - np.array([0.0, 0.0])) <= 0.05) and self.grasped
        return {'task_solved': task_solved}

    def termination_condition(self):
        return not self.grasped


from environments import *


class GeneratedEnv_2DPlaceEnv_NoTermReward_0(SimplifiedPlacingEnv):

    def reward_fun(self, observation, action):
        cube_pose = self.c
        goal_pose = np.array([0.0, 0.0])

        rewards_dict = {}
        dist_to_goal = np.linalg.norm(cube_pose - goal_pose)
        grasp_reward = np.float32(self.grasped)
        rewards_dict["dist_to_goal_reward"] = -dist_to_goal  # reward getting closer to the goal
        rewards_dict["grasp_reward"] = grasp_reward  # reward for grasping the cube

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        cube_pose = self.c
        goal_pose = np.array([0.0, 0.0])
        task_solved = np.linalg.norm(cube_pose - goal_pose) < 0.05 and self.grasped
        return {'task_solved': task_solved}

    def termination_condition(self):
        # Termination condition is when the agent doesn't grasp the cube
        if not self.grasped:
            return True
        return False


from environments import *


class GeneratedEnv_2DSlideEnv_NoTermReward_7(SimplifiedGraspingEnv):

    def _get_info(self):
        task_solved = self.c[0] > 0.99  # Task is solved when the cube x coordinate is greater than 0.99
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        rewards_dict = {}
        cube_position = self.c
        agent_position = self.p

        # Positive reward for getting close to the cube
        rewards_dict["distance_reward"] = -np.linalg.norm(agent_position - cube_position)

        # Positive reward for gripping the cube
        rewards_dict["grip_reward"] = 10.0 if self.grasped else -1.0

        # Positive reward for moving cube to the right
        rewards_dict["right_movement_reward"] = 5.0 if cube_position[0] > agent_position[0] else -1.0

        # Sum up rewards from the rewards dictionary
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DSlideEnv_7(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        # Cube position
        c = np.array(observation[4:6])
        # Agent position
        p = np.array(observation[:2])
        # Gripped cube or not
        gripped = observation[6]

        # Calculation of distance
        distance = np.linalg.norm(p - c)

        # The agent should move towards cube if not already gripped
        reward_dict = {
            "move_towards_cube": -distance if not gripped else 0,
            "gripp": 0 if not gripped else 1,
            "move_positive_x": 0 if not gripped else c[0]
        }

        info = self._get_info()

        total_bonuses = sum([reward_dict[k] if reward_dict[k] > 0 else 0 for k in reward_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([reward_dict[k] for k in reward_dict.keys()])

        reward = total_shaping + task_solved_reward

        reward_dict['task_solved_reward'] = task_solved_reward

        return reward, reward_dict

    def _get_info(self):
        info = super()._get_info()

        # Task is solved if the cube x coordinate is more than 0.99
        if self.c[0] > 0.99:
            info['task_solved'] = True

        return info


from environments import *


class GeneratedEnv_2DPlaceEnv_1(SimplifiedPlacingEnv):
    def reward_fun(self, o, a):
        rewards_dict = {}

        distance_to_target = np.linalg.norm(self.c - np.array([0, 0]))
        grasped_object = self.grasped

        rewards_dict["grasp_reward"] = 10.0 if grasped_object else -1.0
        rewards_dict["distance_to_target_reward"] = -distance_to_target * 10

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        distance_to_target = np.linalg.norm(self.c - np.array([0, 0]))
        grasped_object = self.grasped
        return {'task_solved': distance_to_target <= 0.05 and grasped_object}

    def termination_condition(self):
        # Return True if the cube is not grasped which indicates a failed condition
        return not self.grasped


from environments import *


class GeneratedEnv_2DPlaceEnv_NoTermReward_1(SimplifiedPlacingEnv):

    def reward_fun(self, observation, action):
        reward_dict = {}

        # Calculate the distance from the cube to the goal (0,0)
        distance_to_goal = np.linalg.norm(self.c - np.array([0, 0]))

        # Give reward based on the distance to the goal. Closer to the goal is better
        reward_dict["distance_to_goal"] = -distance_to_goal

        # Give reward if the cube is grasped
        if self.grasped:
            reward_dict["cube_grasped"] = 1.0
        else:
            reward_dict["cube_grasped"] = -1.0

        # Calculate the total reward
        reward = sum([reward_dict[k] for k in reward_dict.keys()])

        return reward, reward_dict

    def _get_info(self):
        info = {}

        # Task is considered solved if cube is at 0.05 distance from (0,0) and is grasped
        info["task_solved"] = self.grasped and np.linalg.norm(self.c - np.array([0, 0])) <= 0.05

        return info


from environments import *


class GeneratedEnv_2DPlaceEnv_2(SimplifiedPlacingEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}

        cube_pose = np.array([self.p[0], self.p[1]])  # Cube position
        target_position = np.array([0, 0])  # We want to move the cube to (0, 0)

        distance_to_target = np.linalg.norm(target_position - cube_pose)
        rewards_dict["distance_reward"] = -distance_to_target

        grasp_bonus = 1 if self.grasped else 0
        rewards_dict["grasp_bonus"] = grasp_bonus

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        cube_pose = np.array([self.p[0], self.p[1]])
        target_position = np.array([0, 0])
        distance_to_target = np.linalg.norm(target_position - cube_pose)
        task_solved = distance_to_target < 0.05 and self.grasped  # Task is solved if the cube is within the 0.05 distance to (0,0) + grasped
        return {'task_solved': task_solved}

    def termination_condition(self):
        return not self.grasped  # End the episode if the agent is not holding the cube


from environments import *


class GeneratedEnv_2DPlaceEnv_NoTermReward_2(SimplifiedPlacingEnv):
    def __init__(self, render_mode='human'):
        super().__init__(render_mode=render_mode)

    def reward_fun(self, observation, action):
        x, y = self.c
        grasped = bool(self.grasped)
        distance = ((x - 0) ** 2 + (y - 0) ** 2) ** 0.5

        rewards_dict = {
            'distance_penalty': -distance,
            'grasp_reward': int(grasped)
        }

        if distance < 0.05 and grasped:
            rewards_dict['solved_bonus'] = 100
        else:
            rewards_dict['solved_bonus'] = 0

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        x, y = self.c
        grasped = bool(self.grasped)
        distance = ((x - 0) ** 2 + (y - 0) ** 2) ** 0.5
        info = dict()
        info['task_solved'] = (distance < 0.05 and grasped)
        return info


from environments import *


class GeneratedEnv_2DSlideEnv_8(SimplifiedGraspingEnv):
    def reward_fun(self, observation, action):
        p = observation[:2]  # Agent's position
        c = observation[4:6]  # Cube's position

        info = self._get_info()
        grasped = observation[6] > 0

        rewards_dict = {
            "distance_to_cube_reward": -np.linalg.norm(p - c),  # Encourage getting close to the cube
            "grasping_reward": 5 * grasped,  # Encourage agent to grasp the cube
            "moving_to_right_reward": 10 * (c[0] > p[0]) * grasped,  # Encourage moving the cube to the right
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward
        return reward, rewards_dict

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
        """
        task_solved = self.c[0] > 0.99
        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DPlaceEnv_3(SimplifiedPlacingEnv):

    # Override reward_fun
    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Compute distance between agent and target (0, 0)
        distance_to_target = np.linalg.norm(self.c - np.array([0., 0.]))
        rewards_dict['distance_reward'] = -distance_to_target  # Negative reward proportional to distance.

        # Reward for grasping the cube
        if self.grasped:
            rewards_dict['grasping_reward'] = 1
        else:
            rewards_dict['grasping_reward'] = -0.1  # Small penalty for not grasping the cube

        # Compute total bonuses and total shaping as in your instructions.
        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = self._max_episode_steps * total_bonuses * info['task_solved']

        # Final reward
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    # Override _get_info
    def _get_info(self):
        distance_to_target = np.linalg.norm(self.c - np.array([0., 0.]))
        task_solved = False
        if distance_to_target <= 0.05 and self.grasped:
            task_solved = True
        return {'task_solved': task_solved}

    # Define termination_condition method
    def termination_condition(self):
        if not self.grasped:  # End episode when agent does not grasp the cube
            return True
        return False


from environments import *


class GeneratedEnv_2DPlaceEnv_NoTermReward_3(SimplifiedPlacingEnv):
    def reward_fun(self, obs, action):
        rewards_dict = {}

        # check the distance to the target position
        cube_position = self.c
        target_position = np.array([0, 0])
        distance = np.linalg.norm(target_position - cube_position)

        # check if grasping
        grasp_reward = 5.0 if self.grasped else -5.0

        rewards_dict = {
            "distance_reward": -distance * 0.1,  # reward for being close to the target position
            "grasp_reward": grasp_reward,  # reward for holding the cube
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        # calculate the distance to the target position
        cube_position = self.c
        target_position = np.array([0, 0])
        distance = np.linalg.norm(target_position - cube_position)

        # check if the task is solved i.e the cube is at required distance (or closer) from the target and is grasped
        task_solved = bool(distance <= 0.05 and self.grasped)

        return {"task_solved": task_solved}


from environments import *


class GeneratedEnv_2DSlideEnv_NoTermReward_8(SimplifiedGraspingEnv):

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field 'task_solved'.
            A task should be marked solved if the cube's x-coordinate is > 0.99.
        """
        cube_coord = self.c[0]
        task_solved = cube_coord > 0.99
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        """
            Defines the reward dictionary and calculates the reward based on the current observation.

            Negative distance to the cube encourages the agent to move towards the cube.
            Position of the cube along x-axis in a positive direction is rewarded, which encourages the agent to move the cube to the right.

            The reward is the sum of all values in the reward dictionary.
        """
        reward_dict = {}

        cube_coord = self.c
        agent_coord = self.p
        distance_to_cube = -np.linalg.norm(agent_coord - cube_coord)  # negative distance to the cube

        cube_moved_right = cube_coord[0]  # reward if cube is moved to the right

        # Conditional reward upon grasping the cube
        if self.grasped:
            grasped_reward = 1.0
        else:
            grasped_reward = -0.1  # Small penalty if the agent does not grasp the cube

        reward_dict["distance_to_cube_reward"] = distance_to_cube
        reward_dict["cube_moved_right_reward"] = cube_moved_right
        reward_dict["grasped_reward"] = grasped_reward

        reward = sum([reward_dict[k] for k in reward_dict.keys()])

        return reward, reward_dict


from environments import *


class GeneratedEnv_2DPlaceEnv_4(SimplifiedPlacingEnv):

    def reward_fun(self, observation, action):
        reward_dict = {}

        cube_position = np.array([self.c[0], self.c[1]])

        # Calculate the Euclidean distance from the cube's current position to the target
        distance_to_goal = np.linalg.norm(cube_position - [0, 0])

        # Compute reward components based on environment properties and task objectives
        # For any negative reward (penalty), consider to cautiously adjust the magnitude
        reward_dict = {
            "grasp_reward": 5 if self.grasped else -1,  # encourage grasping the cube
            "distance_penalty": - 5 * distance_to_goal,  # discourage distance from the goal
        }

        total_bonuses = sum([reward_dict[k] if reward_dict[k] > 0 else 0 for k in reward_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1

        info = self._get_info()
        total_shaping = sum([reward_dict[k] for k in reward_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        reward = total_shaping + task_solved_reward

        reward_dict['task_solved_reward'] = task_solved_reward

        return reward, reward_dict

    def _get_info(self):
        """
        The task is solved if:
        - The cube is at 0.05 distance from (0, 0)
        - The cube is grasped
        """
        info = {}
        cube_position = np.array([self.c[0], self.c[1]])
        distance_to_goal = np.linalg.norm(cube_position - [0, 0])
        info["task_solved"] = distance_to_goal <= 0.05 and self.grasped
        return info

    def termination_condition(self):
        """
        The task has failed if the agent is not grasping the cube.
        """
        return not self.grasped


from environments import *


class GeneratedEnv_2DPlaceEnv_NoTermReward_4(SimplifiedPlacingEnv):

    def reward_fun(self, observation, action):
        rewards_dict = {}  # Defines the reward components
        cube_pose = np.array(self.c)
        end_effector_pose = np.array(self.p)

        distance = np.linalg.norm(cube_pose - end_effector_pose)
        distance_to_target = np.linalg.norm(cube_pose - np.array([0, 0]))

        task_solved = distance_to_target <= 0.05 and self.grasped

        rewards_dict = {
            "distance_to_cube_reward": -distance,  # encourage getting close to the cube
            "grasped_reward": self.grasped,  # encourage grasping the cube
            "distance_to_target_reward": -distance_to_target,  # encourage moving cube to target position
            "task_solved_reward": 500 * task_solved  # reward if task is solved
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        cube_pose = self.c
        distance_to_target = np.sqrt(np.sum(np.square(cube_pose - np.array([0, 0]))))
        task_solved = distance_to_target <= 0.05 and self.grasped
        return {'task_solved': task_solved}

    def termination_condition(self):
        # Task fails if agent is not grasping the cuebe
        return not self.grasped


from environments import *


class GeneratedEnv_2DPlaceEnv_NoTermReward_5(SimplifiedPlacingEnv):

    def reward_fun(self, observation, action):
        reward_dict = {}

        # get the position of the cube
        cube_pose = np.array(self.c)

        # distance to the target
        distance = np.linalg.norm(cube_pose - np.array([0, 0]))
        reward_dict["distance_reward"] = - distance if self.grasped else 0.0

        # reward for grasping the cube
        reward_dict["grasp_reward"] = 10.0 if self.grasped else 0.0

        reward = sum([reward_dict[k] for k in reward_dict.keys()])
        return reward, reward_dict

    def _get_info(self):
        cube_pose = self.c
        distance_to_goal = np.linalg.norm(cube_pose - np.array([0, 0]))

        info = {
            'task_solved': distance_to_goal <= 0.05 and self.grasped
        }

        return info


from environments import *


class GeneratedEnv_2DSlideEnv_9(SimplifiedGraspingEnv):

    def reward_fun(self, observation, action):
        info = self._get_info()

        # reward for getting close to the cube
        proximity_reward = -np.linalg.norm(self.p - self.c)

        # reward for grasping the cube
        grasp_reward = 1.0 if self.grasped else 0.0

        # reward for moving cube to the right > 0.99
        right_move_reward = 10.0 if self.c[0] > 0.99 else 0.0

        rewards_dict = {
            'proximity_reward': proximity_reward,
            'grasp_reward': grasp_reward,
            'right_move_reward': right_move_reward
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])

        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        task_solved = self.c[0] > 0.99 and self.grasped
        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DSlideEnv_NoTermReward_9(SimplifiedGraspingEnv):

    def _get_info(self):
        cube_pos_x = self.c[0]
        task_solved = cube_pos_x > 0.99
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        agent_pos = observation[:2]
        cube_pos = observation[4:6]
        grip_status = observation[6]
        x_distance_to_cube = abs(agent_pos[0] - cube_pos[0])
        y_distance_to_cube = abs(agent_pos[1] - cube_pos[1])
        cube_x_position_reward = cube_pos[0]  # rewarding cube to move to the right
        agent_grip_reward = 1.0 if grip_status else -0.5  # rewarding agent to grip the cube

        rewards_dict = {
            "x_distance_to_cube_penalty": -x_distance_to_cube,
            "y_distance_to_cube_penalty": -y_distance_to_cube,
            "cube_x_position_reward": cube_x_position_reward,
            "agent_grip_reward": agent_grip_reward
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DPlaceEnv_NoTermReward_6(SimplifiedPlacingEnv):

    def reward_fun(self, observation, action):
        reward_dict = {}

        # calculate distance from cube to the target
        distance = np.linalg.norm(self.c - np.array([0.0, 0.0]))

        # check if cube is grasped
        if not self.grasped:
            fail_grasp_reward = -100  # choose a small penalty compared to positive rewards
        else:
            fail_grasp_reward = 0

        # define components of the reward
        reward_dict["negative_distance"] = -distance  # negative distance to the target encourages agent to get closer
        reward_dict["grasp_penalty"] = fail_grasp_reward

        reward = sum([reward_dict[k] for k in reward_dict.keys()])
        return reward, reward_dict

    def _get_info(self):
        info = {}

        # calculate distance from cube to the target
        distance = np.linalg.norm(self.c - np.array([0.0, 0.0]))

        # task is solved when the cube is at 0.05 distance
        task_solved = bool(distance <= 0.05 and self.grasped)
        info["task_solved"] = task_solved

        return info

    def termination_condition(self):
        # episode fails if the agent drops the cube
        if not self.grasped:
            return True
        else:
            return False


from environments import *


class GeneratedEnv_2DPlaceEnv_5(SimplifiedPlacingEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}
        distance_to_target = np.linalg.norm(self.c - np.array([0.0, 0.0]))
        grasped = self.grasped

        rewards_dict['distance_reward'] = -distance_to_target  # encourage the agent to move the cube to the target
        rewards_dict['grasp_reward'] = 5 * grasped  # reward the agent if it grasps the cube

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward
        return reward, rewards_dict

    def _get_info(self):
        """Return a dictionary with task-specific information."""
        distance_to_target = np.linalg.norm(self.c - np.array([0.0, 0.0]))
        grasped = self.grasped
        task_solved = distance_to_target <= 0.05 and grasped
        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DPlaceEnv_NoTermReward_7(SimplifiedPlacingEnv):
    def reward_fun(self, observation, action):
        cube_pose = np.array(self.c)
        distance = np.linalg.norm(cube_pose - 0)  # distance from cube to origin

        rewards_dict = {
            "distance_reward": -distance,  # encourage getting close to the target
            "grasp_reward": 1000 * int(self.grasped),  # encourage grasping the cube
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        cube_pose = np.array(self.c)
        a_epsilon = 0.05  # acceptance distance from target
        within_target_radius = np.linalg.norm(cube_pose - 0) < a_epsilon
        task_solved = within_target_radius and self.grasped  # task is solved when cube is within target radius and grasped

        return {'task_solved': task_solved}

    def termination_condition(self):
        if not self.grasped:
            return True
        return False


from environments import *


class GeneratedEnv_2DPlaceEnv_NoTermReward_8(SimplifiedPlacingEnv):
    def reward_fun(self, obs, a):
        rewards_dict = {}

        # Calculate the distance from cube to the target [0, 0]
        distance_to_target = np.linalg.norm(np.array(self.c) - np.array([0, 0]))

        # Define rewards
        rewards_dict['distance_to_target_reward'] = -distance_to_target  # the closer to the target the better
        rewards_dict['grasping_reward'] = 1.0 if self.grasped else -0.1  # encourage grasping the cube

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        info = {}

        # Check if the task is solved
        distance_to_target = np.linalg.norm(np.array(self.c) - np.array([0, 0]))
        info['task_solved'] = (self.grasped and distance_to_target <= 0.05)

        return info

    def termination_condition(self):
        return not self.grasped  # terminate if the agent doesn't grasp the cube


from environments import *


class GeneratedEnv_2DPlaceEnv_6(SimplifiedPlacingEnv):
    def _get_info(self):
        cube_pose = np.array(self.c)
        distance_to_target = np.linalg.norm(cube_pose - np.array([0, 0]))

        task_solved = (distance_to_target <= 0.05 and self.grasped)
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        cube_pose = np.array(self.c)
        distance_to_target = np.linalg.norm(cube_pose - np.array([0, 0]))

        grasp_reward = 3.0 if self.grasped else -1.0
        distance_reward = -distance_to_target

        rewards_dict = {
            "grasp_reward": grasp_reward,
            "distance_reward": distance_reward
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def termination_condition(self):
        return not self.grasped  # terminate if the agent doesn't grasp the cube


class GeneratedEnv_2DPlaceEnv_NoTermReward_9(SimplifiedPlacingEnv):

    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Get current cube and agent positions
        cube_pose = self.c
        agent_pose = self.p

        # Calculate distance to target
        target = np.array([0., 0.])
        distance = np.linalg.norm(target - cube_pose[:2])

        # Reward for getting closer to target
        rewards_dict["distance_reward"] = -distance

        # Reward for holding the cube
        rewards_dict["grasp_reward"] = 1. if self.grasped else 0.

        # Calculate reward
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        cube_pose = self.c
        agent_pose = self.p

        # Calculate distance to target
        target = np.array([0., 0.])
        distance = np.linalg.norm(target - cube_pose[:2])

        # Determine if task is solved
        grasped = self.grasped
        close_to_target = distance < 0.05
        task_solved = close_to_target and grasped

        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DPlaceEnv_7(SimplifiedPlacingEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}  # Defines the reward components

        # Distance from cube to target position (0,0)
        distance_to_target = np.linalg.norm(np.array(self.c) - np.array([0., 0.]))

        # Checks if the cube is grasped
        is_grasped = float(self.grasped)

        rewards_dict = {
            "distance_to_target_reward": -distance_to_target,  # encourage getting close to the target position
            "grasping_reward": is_grasped,  # encourage grasping of the cube
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
        # Task considered solved if cube is within 0.05 distance to target (0, 0) and is grasped.
        task_solved = (np.linalg.norm(np.array(self.c) - np.array([0., 0.])) < 0.05) and self.grasped
        return {'task_solved': task_solved}

    def termination_condition(self):
        # Task failed if not grasping the cube
        task_failed = not self.grasped
        return task_failed


from environments import *


class GeneratedEnv_2DPlaceEnv_8(SimplifiedPlacingEnv):

    def _get_info(self):
        cube_pose = np.array(self.c)
        grasped = self.grasped

        task_solved = np.linalg.norm(cube_pose - np.array([0.0, 0.0])) < 0.05 and grasped
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        self.action = action
        rewards_dict = {}

        # Pose of the cube
        cube_pose = np.array(self.c)

        # Distance between the cube and the target location (0,0)
        distance = np.linalg.norm(cube_pose - np.array([0.0, 0.0]))

        # Grasping reward
        grasped = self.grasped

        rewards_dict = {
            "distance_to_target_reward": -distance,  # encourage getting close to the target location
            "grasping_reward": 5 * int(grasped)  # encourage grasping of the cube
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward
        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiGraspEnv_0(SimplifiedBiGraspingEnv):
    def _get_info(self):
        task_solved = self.c[1] >= 0.5
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Reward for getting close to the target cube for both agents
        cube_pos = self.c
        dist_agent_0_to_cube = np.linalg.norm(observation[:2] - cube_pos)
        dist_agent_1_to_cube = np.linalg.norm(observation[4:6] - cube_pos)
        rewards_dict['distance_to_cube_reward'] = -(dist_agent_0_to_cube + dist_agent_1_to_cube) / 2

        # Reward for grasping the cube for both agents
        rewards_dict['grasp_reward'] = 1 * (self.grasped_0 and self.grasped_1)

        # Reward for lifting the cube
        rewards_dict['lift_cube_reward'] = self.c[1]

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward
        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DPlaceEnv_9(SimplifiedPlacingEnv):

    def _get_info(self):
        euc_distance = np.linalg.norm(self.c - np.array([0.0, 0.0]))
        task_solved = bool(euc_distance <= 0.05 and self.grasped)
        return {"task_solved": task_solved}

    def reward_fun(self, observation, action):
        rewards_dict = {}

        euc_distance = np.linalg.norm(self.c - np.array([0.0, 0.0]))
        grasp_reward = 1 if self.grasped else 0

        # Reward components
        rewards_dict["distance_reward"] = -euc_distance
        rewards_dict["grasped_reward"] = grasp_reward

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward
        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiGraspEnv_NoTermReward_0(SimplifiedBiGraspingEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Calculate the distance of both grippers to the cube
        distance_0_to_cube = np.linalg.norm(observation[:2] - observation[8:10])
        distance_1_to_cube = np.linalg.norm(observation[4:6] - observation[8:10])

        # Encourage getting close to the cube
        rewards_dict["distance_to_cube_reward"] = -(distance_0_to_cube + distance_1_to_cube)

        # Encourage both graspers to grip the cube
        rewards_dict["grasping_reward"] = 500.0 * (observation[10] + observation[11])

        # Encourage lifting the cube
        rewards_dict["cube_height_reward"] = 1000 * observation[9]

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        # Check if the cube has been lifted
        task_solved = self.grasped_0 and self.grasped_1 and (self.c[1] > 0.5)
        return {'task_solved': task_solved}

    def termination_condition(self):
        """
        Task has no termination condition
        """
        return False


from environments import *


class GeneratedEnv_2DBiGraspEnv_NoTermReward_1(SimplifiedBiGraspingEnv):

    def reward_fun(self, observation, action):
        p_0 = observation[:2]
        p_1 = observation[4:6]
        cube_pos = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]

        distance_to_cube_0 = np.linalg.norm(p_0 - cube_pos)
        distance_to_cube_1 = np.linalg.norm(p_1 - cube_pos)

        reward_dict = {
            "distance_to_cube_reward_0": -distance_to_cube_0 if not grasped_0 else 0,
            "distance_to_cube_reward_1": -distance_to_cube_1 if not grasped_1 else 0,
            "grasp_bonus_0": 1.0 if grasped_0 else 0,
            "grasp_bonus_1": 1.0 if grasped_1 else 0,
            "lift_reward": cube_pos[1] if cube_pos[1] > 0.5 else 0
        }

        reward = sum([reward_dict[k] for k in reward_dict.keys()])

        return reward, reward_dict

    def _get_info(self):
        task_solved = (self.grasped_0 and self.grasped_0) and (self.c[1] >= 0.5)
        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DBiGraspEnv_1(SimplifiedBiGraspingEnv):

    def reward_fun(self, observation, action):
        p_0 = observation[:2]
        p_1 = observation[4:6]
        c = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]

        distance_0 = np.sqrt(np.sum(np.square(np.subtract(p_0, c))))
        distance_1 = np.sqrt(np.sum(np.square(np.subtract(p_1, c))))

        rewards_dict = {
            'distance_0_to_cube': -distance_0 if grasped_0 == 0 else 0,
            'distance_1_to_cube': -distance_1 if grasped_1 == 0 else 0,
            'grasp_and_lift_0': c[1] if grasped_0 > 0 else 0,
            'grasp_and_lift_1': c[1] if grasped_1 > 0 else 0
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
        task_solved = True if self.c[1] >= 0.5 else False
        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DBiGraspEnv_NoTermReward_2(SimplifiedBiGraspingEnv):
    def reward_fun(self, observation, action):
        p_0 = observation[:2]
        p_1 = observation[4:6]
        c = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]

        dist_to_cube_0 = np.linalg.norm(p_0 - c)
        dist_to_cube_1 = np.linalg.norm(p_1 - c)

        dist_decrease = self.p_0_init[0] - dist_to_cube_0 + self.p_1_init[
            0] - dist_to_cube_1  # Assuming initial distances saved in p_0 and p_1

        rewards_dict = {
            "dist_decrease": dist_decrease,  # +reward for decreasing the distance
            "grasping_0": 1 if grasped_0 else -0.1,  # +1 reward for agent 0 grasping, -0.1 for not grasping
            "grasping_1": 1 if grasped_1 else -0.1,  # +1 reward for agent 1 grasping, -0.1 for not grasping
            "lifting": c[1] if grasped_0 and grasped_1 else -0.1  # +reward for lifting, -0.1 otherwise
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        task_solved = self.c[1] >= 0.5
        return {'task_solved': task_solved}

    def termination_condition(self):
        return False


from environments import *


class GeneratedEnv_2DBiGraspEnv_2(SimplifiedBiGraspingEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}

        p_0 = observation[:2]
        p_1 = observation[4:6]
        c = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]

        # Reward for getting close to the cube
        distance_0_to_cube = np.linalg.norm(p_0 - c)
        distance_1_to_cube = np.linalg.norm(p_1 - c)
        rewards_dict["distance_to_cube_reward"] = -distance_0_to_cube - distance_1_to_cube

        # Small bonus for grasping the cube
        rewards_dict["grasp_cube_reward"] = 5 * (grasped_0 + grasped_1)

        # Reward for lifting the cube
        rewards_dict["lift_cube_reward"] = 10 * c[1] * (grasped_0 and grasped_1)

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1

        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        # If cube has reached 0.5 height, consider task as solved
        info = {'task_solved': self.c[1] >= 0.5}
        return info


from environments import *


class GeneratedEnv_2DBiGraspEnv_NoTermReward_3(SimplifiedBiGraspingEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Getting the positions of the agents and the cube
        p_0 = observation[:2]
        p_1 = observation[4:6]
        c = observation[8:10]

        # Checking if the agents have grasped the cube
        grasped_0 = int(observation[10])
        grasped_1 = int(observation[11])

        # Reward for getting close to the cube
        dist_0_to_cube = np.linalg.norm(p_0 - c)
        dist_1_to_cube = np.linalg.norm(p_1 - c)
        rewards_dict["distance_to_cube_reward"] = -0.5 * (dist_0_to_cube + dist_1_to_cube)

        # Reward for grasping the cube
        grasp_rewards = 1.0
        if grasped_0 and grasped_1:
            rewards_dict["grasp_reward"] = grasp_rewards
        else:
            rewards_dict["grasp_reward"] = -grasp_rewards

        # Reward for lifting the cube
        rewards_dict["lift_reward"] = c[1] if (c[1] >= 0.5) else 0.0

        # Summing up the rewards
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        cube_height = self.c[1]
        return {'task_solved': cube_height >= 0.5}


from environments import *


class GeneratedEnv_2DBiGraspEnv_3(SimplifiedBiGraspingEnv):

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        task_solved = self.c[1] >= 0.5
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        p_0, p_1, c = observation[:2], observation[4:6], observation[8:10]
        grasped_0, grasped_1 = observation[10], observation[11]

        # Cube distance calculation
        p_0_distance = np.linalg.norm(p_0 - c)
        p_1_distance = np.linalg.norm(p_1 - c)

        cube_touch_reward = 1.0 * (grasped_0 + grasped_1)
        cube_height_reward = 2.0 * self.c[1]
        penalty_distance = -0.001 * (p_0_distance + p_1_distance)

        rewards_dict = {
            "penalty_distance": penalty_distance,
            "cube_height_reward": cube_height_reward,
            "cube_touch_reward": cube_touch_reward
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1

        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])

        info = self._get_info()

        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiGraspEnv_NoTermReward_4(SimplifiedBiGraspingEnv):

    def reward_fun(self, observation, action):  # Defines the reward components
        p_0 = observation[:2]
        p_1 = observation[4:6]
        c = observation[8:10]

        distance_0 = np.linalg.norm(p_0 - c)
        distance_1 = np.linalg.norm(p_1 - c)

        grasped_0_reward = 1 * observation[10]
        grasped_1_reward = 1 * observation[11]

        height_reward = 1000 * c[1]

        rewards_dict = {
            "distance_to_cube_agent_0_reward": -distance_0,  # reward for getting close to the cube
            "distance_to_cube_agent_1_reward": -distance_1,  # reward for getting close to the cube
            "grasped_agent_0_reward": grasped_0_reward,  # reward for agent 0 grasping the cube
            "grasped_agent_1_reward": grasped_1_reward,  # reward for agent 1 grasping the cube
            "cube_height_reward": height_reward  # reward for lifting the cube
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        cube_height = self.c[1]
        return {'task_solved': cube_height > 0.5}


from environments import *


class GeneratedEnv_2DBiGraspEnv_4(SimplifiedBiGraspingEnv):
    def reward_fun(self, observation, action):
        # Extract the positions of agents and cube
        p_0 = observation[:2]
        p_1 = observation[4:6]
        c = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]
        # Compute the distance of each agent to the cube
        dist_0 = np.linalg.norm(p_0 - c)
        dist_1 = np.linalg.norm(p_1 - c)
        # Define reward components
        rewards_dict = {
            'dist_0': -dist_0,  # Reward for agent 0 being close to the cube
            'dist_1': -dist_1,  # Reward for agent 1 being close to the cube
            'grasp_0_reward': 5 * grasped_0,  # Reward for agent 0 grasping the cube
            'grasp_1_reward': 5 * grasped_1,  # Reward for agent 1 grasping the cube
            'cube_height_reward': 100 * max(0, c[1] - 0.02)  # Reward for lifting the cube
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
        cube_height = self.c[1]
        task_solved = cube_height > 0.5  # The task is solved if the cube is lifted above 0.5
        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DBiGraspEnv_NoTermReward_5(SimplifiedBiGraspingEnv):

    def reward_fun(self, observation, action):
        """
            Define the method here
            You are free to use any variables in the observation or none
            Implement a shaping clearly depends on the task you want to solve, consider to use the bonuses provided
        """

        p_0 = observation[:2]
        p_1 = observation[4:6]
        c = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]

        distance_0_to_cube = np.linalg.norm(np.array(p_0) - np.array(c))
        distance_1_to_cube = np.linalg.norm(np.array(p_1) - np.array(c))

        rewards_dict = {
            "distance_to_cube_reward_0": - distance_0_to_cube,
            # Negative reward for the distance to the cube for the agent 0
            "distance_to_cube_reward_1": - distance_1_to_cube,
            # Negative reward for the distance to the cube for the agent 1
            "grasping_reward_0": 1.0 * grasped_0,  # Reward for grasping the cube for the agent 0
            "grasping_reward_1": 1.0 * grasped_1,  # Reward for grasping the cube for the agent 1
            "cube_height_reward": 10.0 * c[1],  # Reward proportional to the height of the cube
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        # If the height of the cube is at least 0.5 the task is considered solved
        task_solved = (self.c[1] >= 0.5)
        return {'task_solved': task_solved}

    def termination_condition(self):
        """
            Should be False as stated by the prompt
        """
        return False


from environments import *


class GeneratedEnv_2DBiGraspEnv_5(SimplifiedBiGraspingEnv):
    def _get_info(self):
        # Task is solved when the cube is at y=0.5
        return {'task_solved': self.c[1] >= 0.5}

    def reward_fun(self, observation, action):
        # Extracting states from observation
        p_0 = observation[:2]
        p_0_dot = observation[2:4]
        p_1 = observation[4:6]
        p_1_dot = observation[6:8]
        c = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]

        # Computing distance from agents to the cube
        distance_0_to_cube = np.linalg.norm(p_0 - c)
        distance_1_to_cube = np.linalg.norm(p_1 - c)

        # Reward definitions
        rewards_dict = {
            "agent_0_distance_to_cube": - distance_0_to_cube,
            "grasp_reward_agent_0": 10 * grasped_0,
            "cube_height_reward_agent_0": 20 * c[1] if grasped_0 else 0,
            "velocity_reward_agent_0": - np.linalg.norm(p_0_dot),

            "agent_1_distance_to_cube": - distance_1_to_cube,
            "grasp_reward_agent_1": 10 * grasped_1,
            "cube_height_reward_agent_1": 20 * c[1] if grasped_1 else 0,
            "velocity_reward_agent_1": - np.linalg.norm(p_1_dot)
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward
        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiGraspEnv_NoTermReward_6(SimplifiedBiGraspingEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}
        p_0, p_0_dot, p_1, p_1_dot, c, grasped_0, grasped_1 = observation[:2], observation[2:4], observation[
                                                                                                 4:6], observation[
                                                                                                       6:8], observation[
                                                                                                             8:10], \
        observation[10], observation[11]

        # Distance between agent 0 and cube
        distance_0 = np.linalg.norm(p_0 - c)
        # Distance between agent 1 and cube
        distance_1 = np.linalg.norm(p_1 - c)

        # In this task, we encourage both agents to get closer to the cube
        rewards_dict["distance_agent_0_to_cube_reward"] = -distance_0
        rewards_dict["distance_agent_1_to_cube_reward"] = -distance_1

        # When the agent has successfully grasped the cube, reward is given
        rewards_dict["grasp_reward_agent_0"] = 0.5 * int(grasped_0)
        rewards_dict["grasp_reward_agent_1"] = 0.5 * int(grasped_1)

        # Reward for how high the cube has been lifted
        rewards_dict['cube_height_reward'] = 10 * c[1]

        # The total reward is the sum of these components
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        cube_z = self.c[1]
        # The task is considered solved if the cube has been lifted to a height above 0.5
        return {'task_solved': cube_z > 0.5}


from environments import *


class GeneratedEnv_2DBiGraspEnv_NoTermReward_7(SimplifiedBiGraspingEnv):
    def reward_fun(self, observation, action):
        p_0, p_0_dot, p_1, p_1_dot, c, grasped_0, grasped_1 = observation[:2], observation[2:4], observation[
                                                                                                 4:6], observation[
                                                                                                       6:8], observation[
                                                                                                             8:10], int(
            observation[10]), int(observation[11])
        rewards_dict = {
            "distance_to_cube_reward": -1 * np.linalg.norm(p_0 - c) - 1 * np.linalg.norm(p_1 - c),
            # encourage getting close to the cube
            "grasp_reward": 1 * grasped_0 + 1 * grasped_1,  # encourage interaction with the cube
            "lift_reward": 1000 * (c[1] - 0.0)  # encourage lifting the cube
        }
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        task_solved = (self.c[1] >= 0.5)
        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DBiGraspEnv_6(SimplifiedBiGraspingEnv):

    # Redefine the method _get_info
    def _get_info(self):
        # Check if the cube has been lifted up to the required height
        task_solved = self.c[1] >= 0.5
        # Return the result as a dictionary
        return {'task_solved': task_solved}

    # Redefine the method reward_fun
    def reward_fun(self, observation, action):
        # Extract the agents' positions, cube position, and grasping information
        p_0, p_0_dot, p_1, p_1_dot, c, grasped_0, grasped_1 = observation[:2], observation[2:4], observation[
                                                                                                 4:6], observation[
                                                                                                       6:8], observation[
                                                                                                             8:10], \
        observation[10], observation[11]

        # Calculate the distances from the agents to the cube
        dist_0_to_c = np.sqrt((p_0[0] - c[0]) ** 2 + (p_0[1] - c[1]) ** 2)
        dist_1_to_c = np.sqrt((p_1[0] - c[0]) ** 2 + (p_1[1] - c[1]) ** 2)

        # Calculate rewards based on the distance to the cube, and grasping and lifting the cube
        rewards_dict = {
            "agent_0_distance_to_cube": -dist_0_to_c,
            "agent_1_distance_to_cube": -dist_1_to_c,
            "agent_0_grasping_reward": 5 if grasped_0 else 0,
            "agent_1_grasping_reward": 5 if grasped_1 else 0,
            "lifting_reward": c[1],
        }

        # Calculate additional bonus rewards
        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1

        # Extract the task solved information from _get_info method
        info = self._get_info()
        task_solved = info['task_solved']

        # Calculate task solved rewards
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * task_solved

        # Calculate the total shaping reward
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])

        # Calculate the final reward
        reward = total_shaping + task_solved_reward

        # Add the task solved reward to the rewards dictionary
        rewards_dict['task_solved_reward'] = task_solved_reward

        # Return the final reward and rewards dictionary
        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiGraspEnv_NoTermReward_8(SimplifiedBiGraspingEnv):
    def _get_info(self):
        task_solved = self.c[1] > 0.5
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        p_0 = np.array(observation[:2])
        p_1 = np.array(observation[4:6])
        c = np.array(observation[8:10])

        distance_reward_0 = - np.linalg.norm(p_0 - c)
        distance_reward_1 = - np.linalg.norm(p_1 - c)
        cube_height_reward = self.c[1] - self.c_0[1]
        grasp_reward_0 = observation[10] * 1.0
        grasp_reward_1 = observation[11] * 1.0

        rewards_dict = {
            "distance_reward_0": distance_reward_0,
            "distance_reward_1": distance_reward_1,
            "cube_height_reward": cube_height_reward,
            "grasp_reward_0": grasp_reward_0,
            "grasp_reward_1": grasp_reward_1,
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiGraspEnv_NoTermReward_9(SimplifiedBiGraspingEnv):
    def _get_info(self):
        task_solved = self.c[1] >= 0.5
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        c = self.c
        p_0 = self.p_0
        p_1 = self.p_1

        distance_0_to_cube = np.linalg.norm(np.array(c[:2]) - np.array(p_0[:2]))
        distance_1_to_cube = np.linalg.norm(np.array(c[:2]) - np.array(p_1[:2]))

        rewards_dict = {
            "distance_0_to_cube_reward": -distance_0_to_cube,  # Encourage agent_0 getting close to the cube
            "distance_1_to_cube_reward": -distance_1_to_cube,  # Encourage agent_1 getting close to the cube
            "grasped_reward": 1 * int(self.grasped_0) + 1 * int(self.grasped_1),
            # Encourage both agents grasping the cube
            "cube_height_reward": 10 * c[1]  # Encourage lifting the cube
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiGraspEnv_7(SimplifiedBiGraspingEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}

        # get the agent positions from the observation
        p_0 = observation[:2]
        p_1 = observation[4:6]
        # get the cube position from the observation
        c = observation[8:10]

        # compute the distances for each agent to the cube
        distance_0 = np.linalg.norm(p_0 - c)
        distance_1 = np.linalg.norm(p_1 - c)

        # check if the agents have grasped the cube
        grasped_0 = observation[10] > 0
        grasped_1 = observation[11] > 0

        # compute positive rewards for grasping the cube and getting close to it
        rewards_dict["distance_to_cube_reward_0"] = -distance_0
        rewards_dict["distance_to_cube_reward_1"] = -distance_1
        rewards_dict["grasp_reward_0"] = 2 * int(grasped_0)
        rewards_dict["grasp_reward_1"] = 2 * int(grasped_1)

        # lifting reward: they must lift the cube after it's been grasped
        if grasped_0 and grasped_1:
            rewards_dict["lift_reward"] = c[1]
        else:
            rewards_dict["lift_reward"] = 0

        # compute total bonuses
        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1

        info = self._get_info()

        # compute total reward shaping and task solved reward
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        # check if the cube is above the required height
        c = self.c
        return {'task_solved': c[1] > 0.5}


from environments import *


class GeneratedEnv_2DBiGraspEnv_8(SimplifiedBiGraspingEnv):

    def reward_fun(self, obs, action):
        rewards_dict = {}

        p_0 = obs[:2]
        p_1 = obs[4:6]
        c = obs[8:10]

        # Calculate distance between agents and cube.
        distance_0_to_c = np.linalg.norm(p_0 - c)
        distance_1_to_c = np.linalg.norm(p_1 - c)

        # Increment bonus as agents get closer to cube.
        rewards_dict["distance_to_cube_reward_0"] = -distance_0_to_c
        rewards_dict["distance_to_cube_reward_1"] = -distance_1_to_c

        # Bonus for grasping cube.
        # agent 0 has grasped the cube
        rewards_dict["grasping_reward_0"] = 1 * (obs[10] == 1)
        # agent 1 has grasped the cube
        rewards_dict["grasping_reward_1"] = 1 * (obs[11] == 1)

        # Bonus for lifting cube.
        rewards_dict['cube_height_reward'] = 1 * c[1]

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1

        info = self._get_info()

        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])

        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        """
        Task solved if cube height is above 0.5
        """
        return {'task_solved': self.c[1] > 0.5}


from environments import *


class GeneratedEnv_2DBiGraspEnv_9(SimplifiedBiGraspingEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}
        # Positions of agents and cube
        pos_0, pos_1 = observation[:2], observation[4:6]
        cube_pos = observation[8:10]

        # Distance to cube for both agents
        dist_0 = np.linalg.norm(pos_0 - cube_pos)
        dist_1 = np.linalg.norm(pos_1 - cube_pos)

        # Grasping info
        grasp_0 = observation[10]
        grasp_1 = observation[11]

        # Reward based on distance to cube and grasping
        rewards_dict = {
            "distance_to_cube_reward": -dist_0 - dist_1,  # encourage getting close to the target cube
            "grasp_lift_reward": 1 * (grasp_0 + grasp_1) * cube_pos[1],  # encourage grasping and lifting the cube
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
        task_solved = self.c[
                          1] > 0.5 and self.grasped_0 and self.grasped_1  # task is solved when cube height is 0.5 and both agent grasped the cube
        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DBiHandoverEnv_0(SimplifiedBiHandoverEnv):
    def _get_info(self):
        task_solved = np.linalg.norm(np.array(self.c) - np.array([-0.5, 0.0])) <= 0.1
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        info = self._get_info()

        rewards_dict = {}

        # Compute the distances between the cube and the agents
        dist_to_cube_1 = np.linalg.norm(self.p_1 - self.c)
        dist_to_cube_0 = np.linalg.norm(self.p_0 - self.c)
        dist_to_end_pos = np.linalg.norm(self.c - np.array([-0.5, 0.0]))

        # Reward components based on the distances and the grasping status
        rewards_dict["distance_to_cube_reward_1"] = -dist_to_cube_1 if not self.grasped_1 else 0
        rewards_dict["distance_to_cube_reward_0"] = -dist_to_cube_0 if not (self.grasped_0 or self.grasped_1) else 0
        rewards_dict["distance_to_end_pos_reward_1"] = -dist_to_end_pos if self.grasped_1 else 0
        rewards_dict["distance_to_end_pos_reward_0"] = -dist_to_end_pos if self.grasped_0 and not self.grasped_1 else 0

        # Compute the total reward
        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiHandoverEnv_1(SimplifiedBiHandoverEnv):
    def reward_fun(self, observation, action):
        # Unpack variables from observation
        p_0, p_0_dot, p_1, p_1_dot, c, grasped_0, grasped_1 = observation[:2], observation[2:4], observation[
                                                                                                 4:6], observation[
                                                                                                       6:8], observation[
                                                                                                             8:10], \
        observation[10], observation[11]

        # Define reward components
        reward_dict = {}

        # Reward for decreasing the distance from the target cube
        reward_dict['distance_agent0_to_cube_reward'] = -np.linalg.norm(p_0 - c)
        reward_dict['distance_agent1_to_cube_reward'] = -np.linalg.norm(p_1 - c)
        # Reward for grasping the cube
        reward_dict['grasping_reward'] = 5 if grasped_1 == 1 or grasped_0 == 1 else 0
        # Reward for bringing the cube to the target location
        reward_dict['cube_to_target_reward'] = 100 * (1 - np.linalg.norm(c - np.array([-0.5, 0.0])))

        total_bonuses = sum([reward_dict[k] if reward_dict[k] > 0 else 0 for k in reward_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([reward_dict[k] for k in reward_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        reward_dict['task_solved_reward'] = task_solved_reward

        return reward, reward_dict

    def _get_info(self):
        c = self.c
        task_solved = np.linalg.norm(c - np.array([-0.5, 0.0])) <= 0.1
        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DBiHandoverEnv_NoTermReward_0(SimplifiedBiHandoverEnv):
    def reward_fun(self, observation, action):
        p_0 = observation[:2]
        p_0_dot = observation[2:4]
        p_1 = observation[4:6]
        p_1_dot = observation[6:8]
        c = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]

        rewards_dict = {}
        rewards_dict['distance_to_cube_0_reward'] = -np.linalg.norm(
            p_0 - c)  # encourage agent 0 getting close to the cube
        rewards_dict['distance_to_cube_1_reward'] = -np.linalg.norm(
            p_1 - c)  # encourage agent 1 getting close to the cube
        rewards_dict['grasped_cube_0_reward'] = 10 * grasped_0  # encourage agent 0 to grasp the cube
        rewards_dict['grasped_cube_1_reward'] = 10 * grasped_1  # encourage agent 1 to grasp the cube
        rewards_dict['distance_to_target_reward'] = -np.linalg.norm(
            c - np.array([-0.5, 0.0]))  # encourage cube to be at target position

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        c = self.c
        distance_to_target = np.linalg.norm(c - np.array([-0.5, 0.0]))  # distance from cube to target position
        return {'task_solved': distance_to_target <= 0.1}


from environments import *


class GeneratedEnv_2DBiHandoverEnv_NoTermReward_1(SimplifiedBiHandoverEnv):

    def reward_fun(self, observation, action):
        # extract information from observation
        p_0 = observation[:2]
        p_1 = observation[4:6]
        c = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]

        # distance from agent-0 to cube
        dist_0_to_cube = np.linalg.norm(p_0 - c)
        # distance from agent-1 to cube
        dist_1_to_cube = np.linalg.norm(p_1 - c)
        # distance from cube to the goal position
        dist_cube_to_goal = np.linalg.norm(c - np.array([-0.5, 0.0]))

        rewards_dict = {
            "dist_0_to_cube_reward": -dist_0_to_cube if not grasped_1 else 0.,
            "grasp_0_reward": 1. if grasped_0 and not grasped_1 else 0.,
            "handover_reward": 1. if grasped_1 and grasped_0 else 0.,
            "dist_1_to_cube_reward": -dist_1_to_cube if grasped_1 else 0.,
            "dist_cube_to_goal_reward": -dist_cube_to_goal
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        cube_pos = self.c
        goal_pos = np.array([-0.5, 0.0])
        task_solved = np.linalg.norm(cube_pos - goal_pos) <= 0.1

        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DBiHandoverEnv_2(SimplifiedBiHandoverEnv):
    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        task_solved = np.linalg.norm([self.c[0] - (-0.5), self.c[1] - 0.0]) <= 0.1
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        agent_0_reward = -np.linalg.norm(self.p_0 - self.c)
        agent_1_reward = -np.linalg.norm(self.p_1 - self.c)
        final_destination_reward = -np.linalg.norm([self.c[0] - (-0.5), self.c[1] - 0.0])

        bonus_agent_0_grasping = 2 if self.grasped_0 and agent_0_reward > -0.05 else 0
        bonus_agent_1_grasping = 2 if self.grasped_1 and agent_1_reward > -0.05 else 0

        rewards_dict = {"agent_0_reward": agent_0_reward, "bonus_agent_0_grasping": bonus_agent_0_grasping,
                        "agent_1_reward": agent_1_reward, "bonus_agent_1_grasping": bonus_agent_1_grasping,
                        "final_destination_reward": final_destination_reward}

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward
        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiHandoverEnv_3(SimplifiedBiHandoverEnv):

    def reward_fun(self, observation, action):
        info = self._get_info()
        rewards_dict = {}
        p_0 = observation[:2]
        p_1 = observation[4:6]
        c = observation[8:10]
        rewards_dict["agent_to_cube_reward"] = -np.linalg.norm(c - p_0)  # Encourage Agent 1 to get close to cube
        rewards_dict["cube_to_agent_reward"] = -np.linalg.norm(c - p_1)  # Encourage Agent 1 to get close to Agent 0
        rewards_dict["cube_to_goal_reward"] = -np.linalg.norm(
            c + [0.5, 0.0])  # Encourage Agent 0 to take to cube to the goal
        rewards_dict["grasping_reward"] = 10 * (int(observation[10]) == 0 and int(
            observation[11]) == 1)  # Encourage handover from Agent 1 to Agent 0
        rewards_dict["grasping_reward"] -= 0.5 * int(
            not (observation[10] == 0 and observation[11] == 1))  # Penalize if not handed-over

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        task_solved = np.linalg.norm(self.c + [0.5, 0.0]) <= 0.1
        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DBiHandoverEnv_NoTermReward_2(SimplifiedBiHandoverEnv):
    def reward_fun(self, observation, action):
        info = self._get_info()

        rewards_dict = {}

        # Reward for Agent 1 bringing the cube to Agent 0.
        p_0_c = np.linalg.norm(observation[:2] - observation[8:10])
        p_1_c = np.linalg.norm(observation[4:6] - observation[8:10])
        rewards_dict['reward_c_1_0'] = -p_1_c if not self.grasped_1 else -p_0_c

        # Giving bonus if the task is partially solved (Agent 1 hand over cube to Agent 0).
        rewards_dict['bonus'] = 100 if info.get('task_partially_solved') else 0.0

        # Reward for Agent 0 for bringing the cube to the location (-0.5, 0.0)
        rewards_dict['reward_c_0_target'] = -np.linalg.norm(observation[8:10] - np.array([-0.5, 0.0]))

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
        """
        grasped_1 = self.grasped_1
        c = copy.deepcopy(self.c)
        p_0 = copy.deepcopy(self.p_0)
        p_1 = copy.deepcopy(self.p_1)

        grasp_0 = np.linalg.norm(p_0 - c) <= 0.1
        grasp_1 = np.linalg.norm(p_1 - c) <= 0.1

        target_location = np.array([-0.5, 0.0])

        task_partially_solved = grasped_1 and not grasp_0
        task_solved = np.linalg.norm(c - target_location) <= 0.1

        return {'task_solved': task_solved, 'task_partially_solved': task_partially_solved, 'grasp_1': grasp_1}


from environments import *


class GeneratedEnv_2DBiHandoverEnv_4(SimplifiedBiHandoverEnv):
    metadata = {"render_modes": ["human", "rgb_array", "human-shadows"]}

    def reward_fun(self, observation, action):
        rewards_dict = {}

        p_0 = observation[:2]
        p_1 = observation[4:6]
        goal_location = np.array([-0.5, 0.0])
        cube_pos = observation[8:10]

        dist_p0_to_cube = np.linalg.norm(p_0 - cube_pos)
        dist_goal_to_cube = np.linalg.norm(goal_location - cube_pos)
        dist_p1_to_cube = np.linalg.norm(p_1 - cube_pos)

        grasped_0 = observation[10]
        grasped_1 = observation[11]

        # Encourage agent_1 to grasp the cube
        rewards_dict["p1_grasp_cube_reward"] = 1 if grasped_1 else -0.1
        # Encourage agent_0 to get closer to the cube
        rewards_dict["p0_get_close_to_cube_reward"] = -dist_p0_to_cube
        # Encourage agent_1 to get closer to agent_0 when grasping the cube
        rewards_dict["p1_pass_to_p0_reward"] = -dist_p1_to_cube if grasped_1 else 0
        # Encourage agent_0 to move the cube to the goal when received
        rewards_dict["p0_bring_to_goal_reward"] = -dist_goal_to_cube if grasped_0 else 0

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        task_solved = False
        cube_pos = self.c
        goal_location = np.array([-0.5, 0.0])
        dist_goal_to_cube = np.linalg.norm(goal_location - cube_pos)
        if dist_goal_to_cube <= 0.1:
            task_solved = True

        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DBiHandoverEnv_NoTermReward_3(SimplifiedBiHandoverEnv):

    def _get_info(self):
        """
            The task is solved when the cube is at a distance lower than 0.1 from the position (-0.5, 0.0)
        """
        distance_to_target = np.sqrt((self.c[0] - (-0.5)) ** 2)
        task_solved = distance_to_target <= 0.1
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        info = self._get_info()
        p_0 = observation[:2]
        p_1 = observation[4:6]
        c = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]

        rewards_dict = {}

        # The reward function should penalize the distance of Agent 1 to the cube, and Agent 0 to the target location
        # when Agent 1 is holding the cube
        rewards_dict["agent1_to_cube"] = -np.sqrt(np.sum((p_1 - c) ** 2))
        rewards_dict["agent0_to_target"] = -np.sqrt((p_0[0] - (-0.5)) ** 2 + p_0[1] ** 2) if grasped_1 else 0.0

        # When grasping the cube, the agent is encouraged
        rewards_dict["grasping_reward"] = 1.0 if (grasped_0 and not grasped_1) or (
                    not grasped_0 and grasped_1) else -1.0

        # Bonus reward for task solved
        rewards_dict["task_solved_bonus"] = 10.0 if info['task_solved'] else 0.0

        # Final reward is a sum of all the rewards
        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiHandoverEnv_5(SimplifiedBiHandoverEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}

        agent_0_pos = np.array(observation[:2])
        agent_1_pos = np.array(observation[2:4])
        cube_pos = np.array(observation[4:6])

        distance_agent_0_to_cube = np.linalg.norm(agent_0_pos - cube_pos)
        distance_agent_1_to_cube = np.linalg.norm(agent_1_pos - cube_pos)
        distance_to_goal = np.linalg.norm(cube_pos - np.array([-0.5, 0.0]))

        rewards_dict["distance_agent_0_to_cube_penalty"] = -distance_agent_0_to_cube
        rewards_dict["distance_agent_1_to_cube_penalty"] = -distance_agent_1_to_cube
        rewards_dict["progress_to_goal_reward"] = -distance_to_goal

        if observation[10] == 1:
            rewards_dict["grasping_reward"] = 1.0
            rewards_dict["progression_reward"] = -distance_agent_1_to_cube
        if observation[11] == 1:
            rewards_dict["handover_reward"] = 1.0
            rewards_dict["goal_reward"] = -10 * distance_to_goal

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        cube_pos = self.c
        distance_to_goal = np.linalg.norm(cube_pos - np.array([-0.5, 0.0]))
        info = {"task_solved": False}
        if distance_to_goal <= 0.1:
            info["task_solved"] = True
        return info


from environments import *


class GeneratedEnv_2DBiHandoverEnv_NoTermReward_4(SimplifiedBiHandoverEnv):
    def _get_info(self):
        task_solved = np.linalg.norm(self.c - np.array([-0.5, 0.0])) <= 0.1
        return {'task_solved': task_solved}

    def reward_fun(self, observation, action):
        p_0 = observation[:2]
        p_1 = observation[4:6]
        c = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]

        # Encourage agent 1 to get close to the cube
        distance_p1_to_cube = np.linalg.norm(p_1 - c)
        dist_p1_to_cube_reward = -0.1 * distance_p1_to_cube

        # Encourage agent 1 to pass the cube to agent 0
        if grasped_1 and not grasped_0:
            distance_p0_to_cube = np.linalg.norm(p_0 - c)
            dist_p0_to_cube_reward = -0.1 * distance_p0_to_cube
        else:
            dist_p0_to_cube_reward = 0

        # Encourage agent 0 to bring the cube to (-0.5, 0.0)
        distance_to_goal = np.linalg.norm(c - np.array([-0.5, 0.0]))
        dist_to_goal_reward = -distance_to_goal if grasped_0 else 0

        # Define reward components
        rewards_dict = {
            "dist_p1_to_cube_reward": dist_p1_to_cube_reward,
            "dist_p0_to_cube_reward": dist_p0_to_cube_reward,
            "dist_to_goal_reward": dist_to_goal_reward
        }

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiHandoverEnv_6(SimplifiedBiHandoverEnv):
    def _get_info(self):
        cube_pose = self.c
        return {'task_solved': np.linalg.norm(np.array(cube_pose) - np.array([-0.5, 0.])) <= 0.1}

    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Distance of agents from cube
        distance_0_to_cube = np.linalg.norm(self.p_0 - self.c)
        distance_1_to_cube = np.linalg.norm(self.p_1 - self.c)

        # Distance of cube from target
        distance_to_target = np.linalg.norm(self.c - np.array([-0.5, 0.0]))

        # Encourage Agent 0 to move cube towards target location
        rewards_dict["push_to_target_reward"] = -distance_to_target if self.grasped_0 else 0
        # Encourage Agent 1 to grasp cube from Agent 0
        rewards_dict["grasp_cube_reward"] = 10 if self.grasped_1 else -distance_1_to_cube
        # Encourage Agent 0 to get closer to cube initially
        rewards_dict["initial_grasp_reward"] = 10 if self.grasped_0 else -distance_0_to_cube

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1

        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiHandoverEnv_NoTermReward_5(SimplifiedBiHandoverEnv):
    def reward_fun(self, observation, action):
        info = self.get_info()

        rewards_dict = {}

        distance_agent1_to_cube = np.linalg.norm(observation[:2] - observation[8:10])
        distance_agent0_to_cube = np.linalg.norm(observation[4:6] - observation[8:10])
        distance_cube_to_goal = np.linalg.norm(observation[8:10] - np.array([-0.5, 0.0]))

        # Encourage agent 1 to goto cube and grasp it
        rewards_dict["agent1_distance_to_cube_reward"] = -distance_agent1_to_cube
        rewards_dict["agent1_grasping_reward"] = 100.0 * observation[11]

        # Encourage agent 0 to goto cube and grasp it from agent 1
        rewards_dict["agent0_distance_to_cube_reward"] = -distance_agent0_to_cube * info['agent1_has_cube']
        rewards_dict["agent0_grasping_reward"] = 100.0 * observation[10] * info['agent1_has_cube']

        # Encourage agent 0 to goto goal after acquiring the cube
        rewards_dict["agent0_distance_to_goal_reward"] = -distance_cube_to_goal * observation[10]

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def get_info(self):
        """
            Returns a dictionary with information about the environment
        """
        agent1_has_cube = int(self.grasped_0)  # agent1 has cube
        cube_at_goal = int(np.linalg.norm(self.c - np.array([-0.5, 0.0])) <= 0.1)  # cube is at goal
        agent0_has_cube = int(self.grasped_1)  # agent0 has cube

        return {
            'task_solved': cube_at_goal,  # task is solved
            'agent1_has_cube': agent1_has_cube,  # agent 1 has cube
            'agent0_has_cube': agent0_has_cube  # agent 0 has cube
        }

    def _get_info(self):
        cube_pose = self.c
        return {'task_solved': np.linalg.norm(np.array(cube_pose) - np.array([-0.5, 0.])) <= 0.1}


from environments import *


class GeneratedEnv_2DBiHandoverEnv_7(SimplifiedBiHandoverEnv):

    def reward_fun(self, observation, action):
        rewards_dict = {}

        # Get states from observation
        p_0 = observation[:2]
        p_0_dot = observation[2:4]
        p_1 = observation[4:6]
        p_1_dot = observation[6:8]
        c = observation[8:10]
        grasped_0 = bool(observation[10])
        grasped_1 = bool(observation[11])

        # Compute rewards
        rewards_dict["agent_0_to_cube"] = -np.linalg.norm(
            p_0 - c) if grasped_0 else 0  # encourage getting close to the cube
        rewards_dict["agent_1_to_cube"] = -np.linalg.norm(
            p_1 - c) if grasped_1 else 0  # encourage getting close to the cube

        rewards_dict["agent_0_cube_grasped"] = 5 if grasped_0 else 0  # encourage agent 0 to grasp the cube
        rewards_dict["agent_1_cube_grasped"] = 10 if grasped_1 else 0  # encourage agent 1 to grasp the cube

        rewards_dict["cube_to_target"] = -np.linalg.norm(
            c - np.array([-0.5, 0.0]))  # encourage moving cube to destination

        # Calculate the total shaping reward
        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1

        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])

        info = self._get_info()
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        rewards_dict['task_solved_reward'] = task_solved_reward

        reward = total_shaping + task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        info = super()._get_info()
        info['task_solved'] = np.linalg.norm(self.c - np.array([-0.5, 0.0])) <= 0.1
        return info


from environments import *


class GeneratedEnv_2DBiHandoverEnv_NoTermReward_6(SimplifiedBiHandoverEnv):

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        cube_dist_term = np.linalg.norm(self.c - np.array([-0.5, 0.0])) <= 0.1
        return {'task_solved': cube_dist_term}

    def reward_fun(self, observation, action):
        """
        Define the reward components and its values according the task description provided
        """
        rewards_dict = {}

        cube_to_term_dist = np.linalg.norm(self.c - np.array([-0.5, 0.0]))

        cube_to_agent0_dist = np.linalg.norm(self.c - self.p_0)
        cube_to_agent1_dist = np.linalg.norm(self.c - self.p_1)

        agent0_to_term_dist = np.linalg.norm(self.p_0 - np.array([-0.5, 0.0]))
        agent1_to_term_dist = np.linalg.norm(self.p_1 - np.array([-0.5, 0.0]))

        rewards_dict["neg_cube_to_term_dist"] = -cube_to_term_dist
        rewards_dict["neg_cube_to_agent0_dist"] = -cube_to_agent0_dist
        rewards_dict["neg_cube_to_agent1_dist"] = -cube_to_agent1_dist
        rewards_dict["neg_agent0_to_term_dist"] = -agent0_to_term_dist

        rewards_dict["grasped_0_reward"] = 100.0 * self.grasped_0
        rewards_dict["grasped_1_reward"] = 100.0 * self.grasped_1

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiHandoverEnv_NoTermReward_7(SimplifiedBiHandoverEnv):

    def reward_fun(self, observation, action):
        rewards_dict = {}

        dist_agent_1_to_cube = np.linalg.norm(observation[:2] - observation[8:10])
        dist_agent_0_to_cube = np.linalg.norm(observation[4:6] - observation[8:10])
        dist_cube_to_goal = np.linalg.norm(observation[8:10] - np.array([-0.5, 0.0]))

        rewards_dict["dist_agent_1_to_cube"] = -dist_agent_1_to_cube
        rewards_dict["dist_agent_0_to_cube"] = -dist_agent_0_to_cube
        rewards_dict["dist_cube_to_goal"] = -dist_cube_to_goal
        rewards_dict["grasping_reward_agent_1"] = 1000 if observation[10] > 0 else -100
        rewards_dict["grasping_reward_agent_0"] = 1000 if observation[11] > 0 else -100

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        info = super()._get_info()  # get info from the parent class
        dist_cube_to_goal = np.linalg.norm(self.c - np.array([-0.5, 0.0]))
        info['task_solved'] = dist_cube_to_goal <= 0.1
        return info


from environments import *


class GeneratedEnv_2DBiHandoverEnv_8(SimplifiedBiHandoverEnv):
    def reward_fun(self, observation, action):
        p_0 = observation[:2]
        p_1 = observation[4:6]
        cube = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]

        rewards_dict = {}

        # Encourage getting close to the cube
        rewards_dict["distance_to_cube_reward_0"] = -np.linalg.norm(p_0 - cube)
        rewards_dict["distance_to_cube_reward_1"] = -np.linalg.norm(p_1 - cube)

        # Encourage grasping and holding the cube
        rewards_dict["grasping_reward_0"] = 5.0 * grasped_0
        rewards_dict["grasping_reward_1"] = 5.0 * grasped_1

        # Encourage bringing the cube to goal position after grasping
        rewards_dict["goal_dist_reward"] = -np.linalg.norm(cube - np.array([-0.5, 0.0])) if grasped_1 else 0.0

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward
        return reward, rewards_dict

    def _get_info(self):
        cube_pose = self.c
        return {'task_solved': np.linalg.norm(cube_pose - np.array([-0.5, 0.0])) <= 0.1}


from environments import *


class GeneratedEnv_2DBiHandoverEnv_NoTermReward_8(SimplifiedBiHandoverEnv):
    def reward_fun(self, observation, action):
        rewards_dict = {}
        p_0 = observation[:2]
        p_1 = observation[4:6]
        c = observation[8:10]
        grasped_0 = observation[10]
        grasped_1 = observation[11]

        dist_1_to_cube = np.linalg.norm(p_1 - c)
        dist_0_to_goal = np.linalg.norm(p_0 - np.array([-0.5, 0.0]))

        rewards_dict["dist_1_to_cube"] = -dist_1_to_cube  # get agent-1 close to cube
        rewards_dict["dist_0_to_goal"] = -dist_0_to_goal  # get agent-0 close to target position

        # give reward when cube is handed from agent-1 to agent-0
        rewards_dict["handover_reward"] = 10.0 * int(grasped_0 and not grasped_1)

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])
        return reward, rewards_dict

    def _get_info(self):
        task_solved = np.linalg.norm(self.c - np.array([-0.5, 0.0])) <= 0.1  # check if task is solved
        return {'task_solved': task_solved}


from environments import *


class GeneratedEnv_2DBiHandoverEnv_9(SimplifiedBiHandoverEnv):

    def _get_info(self):
        cube_distance_from_goal = np.linalg.norm(self.c - np.array([-0.5, 0.0]))
        return {'task_solved': cube_distance_from_goal <= 0.1}

    def reward_fun(self, observation, action):
        info = self._get_info()

        # The distances of agents from the cube
        agent_0_distance = np.linalg.norm(self.p_0 - self.c)
        agent_1_distance = np.linalg.norm(self.p_1 - self.c)

        # The distance of the cube from the goal
        cube_distance_from_goal = np.linalg.norm(self.c - np.array([-0.5, 0.0]))

        rewards_dict = {
            "agent_0_distance_to_cube": -agent_0_distance,
            "agent_1_distance_to_cube": -agent_1_distance,
            "cube_distance_from_goal": -cube_distance_from_goal,
            "grasped_by_agent_0": int(self.grasped_0),
            "grasped_by_agent_1": int(self.grasped_1)
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict


from environments import *


class GeneratedEnv_2DBiHandoverEnv_NoTermReward_9(SimplifiedBiHandoverEnv):

    def reward_fun(self, observation, action):
        rewards_dict = {}

        dist_to_cube_0 = np.linalg.norm(self.p_0 - self.c)
        dist_to_cube_1 = np.linalg.norm(self.p_1 - self.c)
        dist_to_target = np.linalg.norm(self.c - np.array([-0.5, 0.0]))

        rewards_dict["distance_to_cube_reward_0"] = -dist_to_cube_0
        rewards_dict["distance_to_cube_reward_1"] = -dist_to_cube_1
        rewards_dict["cube_to_target_reward"] = -dist_to_target
        rewards_dict["grasping_reward_0"] = 1.0 if self.grasped_0 else 0.0
        rewards_dict["grasping_reward_1"] = 1.0 if self.grasped_1 else 0.0

        reward = sum([rewards_dict[k] for k in rewards_dict.keys()])

        return reward, rewards_dict

    def _get_info(self):
        dist_to_target = np.linalg.norm(self.c - np.array([-0.5, 0.0]))

        return {'task_solved': dist_to_target <= 0.1}


