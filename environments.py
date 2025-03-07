import copy

import yaml

import sim_utils
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

base_environments_list = ['YumiEnv',
                          'MobileRobotCorridorEnv',
                          'YumiLeftEnv',
                          'SimplifiedGraspingEnv',
                          'YumiLeftEnv_objects_in_scene']

time_step = 0.01
gravity_constant = -9.81
max_force = 100
yumi_urdf_path = 'robot_models/yumi_description/urdf/yumi.urdf'

cube_urdf_path = 'object_models/cube_small.urdf'


yumi_params = {
    'pos_noise': 0.005,
    'vel_noise': 0.005,
    'gripper_pos_noise': 0.001,
    'gripper_vel_noise': 0.001,
    'object_pos_noise': 0.003,
    'object_orient_noise': 0.003,
}
"""
yumi_params = {
    'pos_noise': 0.0,
    'vel_noise': 0.0,
    'gripper_pos_noise': 0.0,
    'gripper_vel_noise': 0.0,
    'object_pos_noise': 0.0,
    'object_orient_noise': 0.0,
}"""



# MobileRobotCorridorEnv
class MobileRobotCorridorEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode='human'):
        super(MobileRobotCorridorEnv, self).__init__()
        self.p_0 = np.array([0, 0])
        self.p_dot_0 = np.array([0, 0])
        self.alpha_dot_0 = 0
        self.alpha_0 = 0
        self.w = 1
        self.d = 0.2
        self.dt = 0.01

        self.objects = [[1, 1], [3, 0]]

        self.p = copy.deepcopy(self.p_0)
        self.p_dot = copy.deepcopy(self.p_0)
        self.alpha = copy.deepcopy(self.alpha_0)
        self.alpha_dot = copy.deepcopy(self.alpha_0)

        self._max_episode_steps = 1000

        self.observation_space = spaces.Box(np.array([0] + [-1] * 5), np.array([5] + [1] * 5), dtype=np.float32)
        self.observation_dim = self.observation_space.shape[0]

        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        self.action_dim = self.action_space.shape[0]
        self.max_action = 1


        self.window_size = (5 * 300, 300)  # The size of the PyGame window

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.load_simulation()
        self.reset()

    def load_simulation(self):
        if self.render_mode == 'human':
            # Initialize PyGame
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Simple Mobile Robot Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

    def _get_obs(self):
        obs = np.zeros((6,))
        obs[:2] = self.p
        obs[2] = self.alpha / (2 * np.pi)
        obs[3:5] = self.p_dot
        obs[5] = self.alpha_dot

        return copy.deepcopy(obs)

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        return {'task_solved': False}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent location
        self.p = copy.deepcopy(self.p_0)
        self.p_dot = copy.deepcopy(self.p_dot_0)
        self.alpha = copy.deepcopy(self.alpha_0)
        self.alpha_dot = copy.deepcopy(self.alpha_dot_0)

        return self._get_obs()

    def step(self, action):
        p_dot = np.clip(action[0], -1, 1) * np.array([np.cos(self.alpha), np.sin(self.alpha)])

        self.p = self.p + p_dot * self.dt  # moving max 1 m/s in each direction
        self.p_dot = p_dot

        alpha_dot = np.clip(action[1], -1, 1) * self.dt
        self.alpha = self.alpha + alpha_dot
        self.alpha_dot = alpha_dot

        self.p = np.clip(self.p, self.observation_space.low[:2], self.observation_space.high[:2])
        self.alpha = self.alpha % (2 * np.pi)

        info = self._get_info()

        if self.render_mode == 'human':
            self.render()  # Call the render method to update the display
        obs = self._get_obs()
        reward, rewards_dict = self.reward_fun(obs, action)

        return obs, reward, self.termination_condition(), rewards_dict, info

    def termination_condition(self):
        """
            Should be overwritten by subclass
        """
        # Hitting the corridor
        if np.abs(self.p[1]) == 1:
            return True

        # Hitting an object
        for obj in self.objects:
            if np.abs(self.p[0] - (obj[0]+self.d/2)) <= self.d / 2 and np.abs(self.p[1] - (obj[1]-self.w/2)) <= self.w / 2:
                return True

        return False

    def reward_fun(self, observation, action):
        info = self._get_info()

        forward_reward = self.p[0]

        rewards_dict = {'forward_reward': forward_reward}

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        task_solved_reward = self._max_episode_steps * total_bonuses * info['task_solved']

        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        reward = total_shaping + task_solved_reward
        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def render(self, mode="human"):
        self.screen.fill((255, 255, 255))  # Fill the screen with white

        # Draw the agent
        agent_pos = self._to_screen_coordinates(self.p)
        pygame.draw.circle(self.screen, (0, 0, 255), agent_pos, 10)

        R = np.array([[np.cos(self.alpha), -np.sin(self.alpha)],
                      [np.sin(self.alpha), np.cos(self.alpha)]])
        agent_top_right = self._to_screen_coordinates(self.p+np.dot(R, np.array([0.05, 0.025])))
        agent_top_left = self._to_screen_coordinates(self.p+np.dot(R, np.array([-0.05, 0.025])))
        agent_bottom_left = self._to_screen_coordinates(self.p+np.dot(R, np.array([-0.05, -0.025])))
        agent_bottom_right = self._to_screen_coordinates(self.p+np.dot(R, np.array([0.05, -0.025])))
        pygame.draw.circle(self.screen, (0, 0, 255), agent_top_right, 5)
        pygame.draw.circle(self.screen, (0, 0, 255), agent_top_left, 5)
        pygame.draw.circle(self.screen, (0, 0, 255), agent_bottom_left, 5)
        pygame.draw.circle(self.screen, (0, 0, 255), agent_bottom_right, 5)

        # Draw the target
        target_pos = self._to_screen_coordinates([5.0, 0.0])
        # print(f"Target position: {self.p_g} -> Screen coordinates: {target_pos}")
        pygame.draw.circle(self.screen, (255, 0, 0), target_pos, 10)  # Draw target as green circle

        # Draw the objects
        for obj in self.objects:
            obj_pos = self._to_screen_coordinates(obj)
            # print(f"Object position: {obj} -> Screen coordinates: {obj_pos}")
            rect_pos = pygame.Rect(obj_pos[0], obj_pos[1], self.d * self.window_size[0] / 5, self.w * self.window_size[1] / 2)
            pygame.draw.rect(self.screen, (0, 0, 0), rect_pos)  # Draw object as red rectangle

        # Update the display
        pygame.display.flip()
        self.clock.tick(200)  # Limit the frame rate to 60 FPS

    def _to_screen_coordinates(self, pos):
        """
        Convert position in environment coordinates to screen coordinates.
        Environment coordinates range from 0 to 5 in x and -1 to 1 in y.
        Screen coordinates range from 0 to window_size.
        """
        return (
            int(pos[0] / 5 * self.window_size[0]),
            int((1 - (pos[1] + 1) / 2) * self.window_size[1])
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def save_current_state(self):
        """
        :return: state representation of the simulation as a list of real numbers

        Should be overwritten by child class if said child adds entities to the simulation
        """

        return list(self.p) + list(self.p_dot) + [self.alpha] + [self.alpha_dot]

    def load_initial_state(self, state):
        """
        :param state: state in the form defined by save_current_state

        Reset the simulation to the given state, also overwrites the initial condition defined in the init method,
        therefore the reset method will reset the simulation state to the state provided to this method

        Should be overwritten by child class if said child adds entities to the simulation
        """
        self.p = copy.deepcopy(state[:2])
        self.p_dot = copy.deepcopy(state[2:4])
        self.alpha = copy.deepcopy(state[4])
        self.alpha_dot = copy.deepcopy(state[6])
# MobileRobotCorridorEnv end

# Env2D
class Env2D(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array", "human-shadows"]}

    def __init__(self, render_mode='human'):
        super(Env2D, self).__init__()
        self.p_0 = np.array([0.0, 1.0])
        self.p_dot_0 = np.array([0.0, 0.0])
        self.grasped_0 = False
        self.h = 1
        self.w = 1
        self.dt = 0.02


        self.p = copy.deepcopy(self.p_0)
        self.p_dot = copy.deepcopy(self.p_0)

        self._max_episode_steps = 1000

        self.observation_space = spaces.Box(np.array([-1] * 4), np.array([1] * 4), dtype=np.float32)
        self.observation_dim = self.observation_space.shape[0]

        self.action_space = spaces.Box(np.array([-1]*2), np.array([1]*2), dtype=np.float32)
        self.action_dim = self.action_space.shape[0]
        self.max_action = 1

        self.window_size = (600, 600)  # The size of the PyGame window

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.t_step = 0

        self.load_simulation()
        self.reset()

    def load_simulation(self):
        if self.render_mode == 'human' or self.render_mode == 'human-shadows':
            # Initialize PyGame
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("2D Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

    def _get_obs(self):
        obs = np.zeros((4,))
        obs[:2] = self.p
        obs[2:4] = self.p_dot

        return copy.deepcopy(obs)

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        return {'task_solved': np.linalg.norm(np.array(self.p)) < 0.01}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t_step = 0

        # Reset agent location
        alpha = np.random.uniform(low=-1.0, high=1.0) * np.pi

        self.p = [np.cos(alpha), np.sin(alpha)]
        self.p_dot = copy.deepcopy(self.p_dot_0)

        if self.render_mode == 'human' or self.render_mode == 'human-shadows':
            self.screen.fill((255, 255, 255))  # Fill the screen with white

        return self._get_obs()

    def step(self, action):
        self.p_dot = np.clip(action[:2], -1, 1)
        p = self.p
        self.p += self.p_dot * self.dt  # cube falls down instantly

        self.p[0] = np.clip(self.p[0], -1, 1)
        self.p[1] = np.clip(self.p[1], -1, 1)

        info = self._get_info()

        self.t_step += 1
        if self.render_mode == 'human':
            self.render(mode=self.render_mode)  # Call the render method to update the display
        if self.render_mode == 'human-shadows' and self.t_step % 20 == 0:
            self.render(mode=self.render_mode)
        obs = self._get_obs()
        reward, rewards_dict = self.reward_fun(obs, action)

        return obs, reward, self.termination_condition() or info['task_solved'], rewards_dict, info

    def termination_condition(self):
        """
            Should be overwritten by subclass
        """
        return False

    def reward_fun(self, observation, action):
        info = self._get_info()

        rewards_dict = {}

        reward = 0

        return reward, rewards_dict

    def render(self, mode="human"):
        if mode == 'human':
            self.screen.fill((255, 255, 255))  # Fill the screen with white
        # print(self.p)
        # Draw the agent
        agent_pos = self._to_screen_coordinates(self.p)
        # print(agent_pos)
        pygame.draw.circle(self.screen, (150, 255, 150), self._to_screen_coordinates([0,0]), int(self.window_size[0]/2))
        pygame.draw.circle(self.screen, (255, 255, 150), agent_pos, 10)


        pygame.image.save(self.screen, "screen.jpg")

        # Update the display
        pygame.display.flip()
        #self.clock.tick(200)  # Limit the frame rate to 60 FPS

    def _to_screen_coordinates(self, pos):
        """
        Convert position in environment coordinates to screen coordinates.
        Environment coordinates range from -1 to 1 in x and -1 to 1 in y.
        Screen coordinates range from 0 to window_size.
        """
        x_screen = int((pos[0] + 1) * (self.window_size[0] / 2))
        y_screen = int((1 - pos[1]) * (self.window_size[1] / 2))  # Flip y-axis for correct screen mapping
        return x_screen, y_screen

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def save_current_state(self):
        """
        :return: state representation of the simulation as a list of real numbers

        Should be overwritten by child class if said child adds entities to the simulation
        """

        return list(self.p) + list(self.p_dot) + list(self.c) + [self.grasped]

    def load_initial_state(self, state):
        """
        :param state: state in the form defined by save_current_state

        Reset the simulation to the given state, also overwrites the initial condition defined in the init method,
        therefore the reset method will reset the simulation state to the state provided to this method

        Should be overwritten by child class if said child adds entities to the simulation
        """
        self.p = copy.deepcopy(state[:2])
        self.p_dot = copy.deepcopy(state[2:4])
        self.c = copy.deepcopy(state[4:6])
        self.grasped = copy.deepcopy(state[6])
# Env2D end

# Env2D_plus_ten
class Env2D_plus_ten(Env2D):
    def reward_fun(self, observation, action):

        rewards_dict = {'dist': - np.linalg.norm(np.array(observation[:2])),
                        'bonus': 10}

        reward = rewards_dict['dist'] + rewards_dict['bonus']

        return reward, rewards_dict
# Env2D_plus_ten end

# Env2D_plus_one
class Env2D_plus_one(Env2D):
    def reward_fun(self, observation, action):
        rewards_dict = {'dist': - np.linalg.norm(np.array(observation[:2])),
                        'bonus': 1}

        reward = rewards_dict['dist'] + rewards_dict['bonus']

        return reward, rewards_dict
# Env2D_plus_one end

# SimplifiedGraspingEnv
class SimplifiedGraspingEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array", "human-shadows"]}

    def __init__(self, render_mode='human'):
        super(SimplifiedGraspingEnv, self).__init__()
        self.p_0 = np.array([0.0, 1.0])
        self.p_dot_0 = np.array([0.0, 0.0])
        self.grasped_0 = False
        self.h = 1
        self.w = 1
        self.dt = 0.01

        self.c_0 = np.array([0.0, 0.0])
        self.cube_init_std = 0.2

        self.p = copy.deepcopy(self.p_0)
        self.p_dot = copy.deepcopy(self.p_0)
        self.c = copy.deepcopy(self.c_0)
        self.grasped = copy.deepcopy(self.grasped_0)

        self._max_episode_steps = 1000

        self.observation_space = spaces.Box(np.array([-1] * 7), np.array([1] * 7), dtype=np.float32)
        self.observation_dim = self.observation_space.shape[0]

        self.action_space = spaces.Box(np.array([-1]*3), np.array([1]*3), dtype=np.float32)
        self.action_dim = self.action_space.shape[0]
        self.max_action = 1

        self.window_size = (2 * 300, 300)  # The size of the PyGame window

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.t_step = 0

        self.load_simulation()
        self.reset()

    def load_simulation(self):
        if self.render_mode == 'human' or self.render_mode == 'human-shadows':
            # Initialize PyGame
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Simple Grasping Robot Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

    def _get_obs(self):
        obs = np.zeros((7,))
        obs[:2] = self.p
        obs[2:4] = self.p_dot
        obs[4:6] = self.c
        obs[6] = int(self.grasped)

        return copy.deepcopy(obs)

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        return {'task_solved':False}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t_step = 0

        # Reset agent location
        self.p = copy.deepcopy(self.p_0)
        self.p_dot = copy.deepcopy(self.p_dot_0)
        self.c = copy.deepcopy(self.c_0)
        self.c[0] += np.clip(np.random.randn() * self.cube_init_std, -1, 1)
        self.grasped = copy.deepcopy(self.grasped_0)

        if self.render_mode == 'human' or self.render_mode == 'human-shadows':
            self.screen.fill((255, 255, 255))  # Fill the screen with white

        return self._get_obs()

    def step(self, action):
        p_dot = np.clip(action[:2], -1, 1)
        p = self.p
        c = self.c
        close_gripper = action[2] >= 0

        self.p += p_dot * self.dt  # cube falls down instantly

        if close_gripper and np.linalg.norm(p-c) < 0.04:  # cube is grasped
            self.c += p_dot * self.dt  # cube and gripper move at the same velocity
            self.grasped = True
        else:
            self.c[1] += -1 * self.dt
            self.grasped = False


        self.p[0] = np.clip(self.p[0], -1, 1)
        self.p[1] = np.clip(self.p[1], 0, 1)
        self.c[0] = np.clip(self.c[0], -1, 1)
        self.c[1] = np.clip(self.c[1], 0, 1)

        info = self._get_info()

        self.t_step += 1
        if self.render_mode == 'human':
            self.render(mode=self.render_mode)  # Call the render method to update the display
        if self.render_mode == 'human-shadows' and self.t_step % 20 == 0:
            self.render(mode=self.render_mode)
        obs = self._get_obs()
        reward, rewards_dict = self.reward_fun(obs, action)

        return obs, reward, self.termination_condition() or info['task_solved'], rewards_dict, info

    def termination_condition(self):
        """
            Should be overwritten by subclass
        """
        return False

    def reward_fun(self, observation, action):
        info = self._get_info()

        rewards_dict = {}

        reward = 0

        return reward, rewards_dict

    def render(self, mode="human"):
        if mode == 'human':
            self.screen.fill((255, 255, 255))  # Fill the screen with white
        # print(self.p)
        # Draw the agent
        agent_pos = self._to_screen_coordinates(self.p)
        # print(agent_pos)
        if self.grasped:
            pygame.draw.circle(self.screen, (150, 150, 255), agent_pos, 10)
        else:
            pygame.draw.circle(self.screen, (150, 255, 150), agent_pos, 10)

        # Draw the cube
        cube_pos = self._to_screen_coordinates(self.c)
        rect_pos = pygame.Rect(cube_pos[0]-self.window_size[0] / 20, cube_pos[1], self.window_size[0] / 20, self.window_size[1] / 20)
        pygame.draw.rect(self.screen, (255, 0, 0), rect_pos)  # Draw cube

        pygame.image.save(self.screen, "screen.jpg")

        # Update the display
        pygame.display.flip()
        self.clock.tick(200)  # Limit the frame rate to 60 FPS

    def _to_screen_coordinates(self, pos):
        """
        Convert position in environment coordinates to screen coordinates.
        Environment coordinates range from 0 to 5 in x and -1 to 1 in y.
        Screen coordinates range from 0 to window_size.
        """
        return (
            int((pos[0]+1) * (self.window_size[0]/2)),
            # int((1 - pos[1]) * self.window_size[1]*0.95 - self.window_size[1]*0.05),
            # int(self.window_size[1]*0.95 - pos[1] *self.window_size[1]*0.95),
            int(self.window_size[1]*0.95 * (1 - pos[1]))
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def save_current_state(self):
        """
        :return: state representation of the simulation as a list of real numbers

        Should be overwritten by child class if said child adds entities to the simulation
        """

        return list(self.p) + list(self.p_dot) + list(self.c) + [self.grasped]

    def load_initial_state(self, state):
        """
        :param state: state in the form defined by save_current_state

        Reset the simulation to the given state, also overwrites the initial condition defined in the init method,
        therefore the reset method will reset the simulation state to the state provided to this method

        Should be overwritten by child class if said child adds entities to the simulation
        """
        self.p = copy.deepcopy(state[:2])
        self.p_dot = copy.deepcopy(state[2:4])
        self.c = copy.deepcopy(state[4:6])
        self.grasped = copy.deepcopy(state[6])
# SimplifiedGraspingEnv end

# SimplifiedPlacingEnv
class SimplifiedPlacingEnv(SimplifiedGraspingEnv):
    def __init__(self, render_mode='human'):
        self.agent_init_std = 0.2
        super().__init__(render_mode=render_mode)
        self.c_0 = np.array([0.0, 1.0])
        self.cube_init_std = 0.0
        self.grasped_0 = False

    def reset(self, seed=None, options=None):
        self.t_step = 0

        # Reset agent location
        self.p = copy.deepcopy(self.p_0)
        self.p[0] = np.clip(self.p[0] + np.random.randn() * self.agent_init_std, -1, 1)
        self.p_dot = copy.deepcopy(self.p_dot_0)
        self.c = copy.deepcopy(self.p)
        self.grasped = copy.deepcopy(self.grasped_0)

        if self.render_mode == 'human' or self.render_mode == 'human-shadows':
            self.screen.fill((255, 255, 255))  # Fill the screen with white

        return self._get_obs()
# SimplifiedPlacingEnv end


# SimplifiedBiGraspingEnv
class SimplifiedBiGraspingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "human-shadows"]}

    def __init__(self, render_mode='human'):
        super(SimplifiedBiGraspingEnv, self).__init__()
        self.p_0_init = np.array([-1.0, 1.0])
        self.p_0_dot_init = np.array([0.0, 0.0])
        self.p_1_init = np.array([1.0, 1.0])
        self.p_1_dot_init = np.array([0.0, 0.0])
        self.grasped_0_init = False
        self.grasped_1_init = False
        self.h = 1
        self.w = 1
        self.dt = 0.01

        self.c_0 = np.array([0.0, 0.0])
        self.cube_init_std = 0.2

        self.p_0 = copy.deepcopy(self.p_0_init)
        self.p_0_dot = copy.deepcopy(self.p_0_dot_init)
        self.p_1 = copy.deepcopy(self.p_1_init)
        self.p_1_dot = copy.deepcopy(self.p_1_dot_init)
        self.c = copy.deepcopy(self.c_0)
        self.grasped_0 = copy.deepcopy(self.grasped_0_init)
        self.grasped_1 = copy.deepcopy(self.grasped_1_init)

        self._max_episode_steps = 1000

        self.observation_space = spaces.Box(np.array([-1] * 12), np.array([1] * 12), dtype=np.float32)
        self.observation_dim = self.observation_space.shape[0]

        self.action_space = spaces.Box(np.array([-1] * 6), np.array([1] * 6), dtype=np.float32)
        self.action_dim = self.action_space.shape[0]
        self.max_action = 1

        self.window_size = (2 * 300, 300)  # The size of the PyGame window

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.t_step = 0

        self.load_simulation()
        self.reset()

    def load_simulation(self):
        if self.render_mode == 'human' or self.render_mode == 'human-shadows':
            # Initialize PyGame
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Simple Grasping Robot Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

    def _get_obs(self):
        obs = np.zeros((12,))
        obs[:2] = self.p_0
        obs[2:4] = self.p_0_dot
        obs[4:6] = self.p_1
        obs[6:8] = self.p_1_dot
        obs[8:10] = self.c
        obs[10] = int(self.grasped_0)
        obs[10] = int(self.grasped_1)
        return copy.deepcopy(obs)

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        return {'task_solved': False}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t_step = 0

        # Reset agent location
        self.p_0 = copy.deepcopy(self.p_0_init)
        self.p_0_dot = copy.deepcopy(self.p_0_dot_init)
        self.p_1 = copy.deepcopy(self.p_1_init)
        self.p_1_dot = copy.deepcopy(self.p_1_dot_init)
        self.c = copy.deepcopy(self.c_0)
        self.grasped_0 = copy.deepcopy(self.grasped_0_init)
        self.grasped_1 = copy.deepcopy(self.grasped_1_init)
        self.c = copy.deepcopy(self.c_0)
        self.c[0] += np.clip(np.random.randn() * self.cube_init_std, -1, 1)

        if self.render_mode == 'human' or self.render_mode == 'human-shadows':
            self.screen.fill((255, 255, 255))  # Fill the screen with white

        return self._get_obs()

    def step(self, action):
        p_0_dot = np.clip(action[:2], -1, 1)
        p_1_dot = np.clip(action[2:4], -1, 1)
        p_0 = self.p_0
        p_1 = self.p_1
        c = self.c
        self.grasped_0 = action[4] >= 0 and np.linalg.norm(p_0 - c) < 0.04
        self.grasped_1 = action[5] >= 0 and np.linalg.norm(p_1 - c) < 0.04

        self.p_0 += p_0_dot * self.dt
        self.p_1 += p_1_dot * self.dt

        if self.grasped_0 and self.grasped_1:  # cube is grasped
            self.c += (p_0_dot + p_1_dot) * self.dt / 2 # cube and gripper move at the same velocity
        else:
            self.c[1] += -1 * self.dt

        self.p_0[0] = np.clip(self.p_0[0], -1, 1)
        self.p_0[1] = np.clip(self.p_0[1], 0, 1)
        self.p_1[0] = np.clip(self.p_1[0], -1, 1)
        self.p_1[1] = np.clip(self.p_1[1], 0, 1)
        self.c[0] = np.clip(self.c[0], -1, 1)
        self.c[1] = np.clip(self.c[1], 0, 1)

        info = self._get_info()

        self.t_step += 1
        if self.render_mode == 'human':
            self.render(mode=self.render_mode)  # Call the render method to update the display
        if self.render_mode == 'human-shadows' and self.t_step % 20 == 0:
            self.render(mode=self.render_mode)
        obs = self._get_obs()
        reward, rewards_dict = self.reward_fun(obs, action)

        return obs, reward, self.termination_condition() or info['task_solved'], rewards_dict, info

    def termination_condition(self):
        """
            Should be overwritten by subclass
        """
        return False

    def reward_fun(self, observation, action):
        info = self._get_info()

        rewards_dict = {}

        reward = 0

        return reward, rewards_dict

    def render(self, mode="human"):
        if mode == 'human':
            self.screen.fill((255, 255, 255))  # Fill the screen with white
        # print(self.p)
        # Draw the agent
        agent_0_pos = self._to_screen_coordinates(self.p_0)
        agent_1_pos = self._to_screen_coordinates(self.p_1)
        # print(agent_pos)
        if self.grasped_0:
            pygame.draw.circle(self.screen, (150, 150, 255), agent_0_pos, 10)
        else:
            pygame.draw.circle(self.screen, (150, 255, 150), agent_0_pos, 10)

        if self.grasped_1:
            pygame.draw.circle(self.screen, (150, 150, 255), agent_1_pos, 10)
        else:
            pygame.draw.circle(self.screen, (150, 255, 150), agent_1_pos, 10)


        # Draw the cube
        cube_pos = self._to_screen_coordinates(self.c)
        rect_pos = pygame.Rect(cube_pos[0] - self.window_size[0] / 20, cube_pos[1], self.window_size[0] / 20,
                               self.window_size[1] / 20)
        pygame.draw.rect(self.screen, (255, 0, 0), rect_pos)  # Draw cube

        pygame.image.save(self.screen, "screen.jpg")

        # Update the display
        pygame.display.flip()
        self.clock.tick(200)  # Limit the frame rate to 60 FPS

    def _to_screen_coordinates(self, pos):
        """
        Convert position in environment coordinates to screen coordinates.
        Environment coordinates range from 0 to 5 in x and -1 to 1 in y.
        Screen coordinates range from 0 to window_size.
        """
        return (
            int((pos[0] + 1) * (self.window_size[0] / 2)),
            # int((1 - pos[1]) * self.window_size[1]*0.95 - self.window_size[1]*0.05),
            # int(self.window_size[1]*0.95 - pos[1] *self.window_size[1]*0.95),
            int(self.window_size[1] * 0.95 * (1 - pos[1]))
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def save_current_state(self):
        """
        :return: state representation of the simulation as a list of real numbers

        Should be overwritten by child class if said child adds entities to the simulation
        """

        return list(self.p) + list(self.p_dot) + list(self.c) + [self.grasped]

    def load_initial_state(self, state):
        """
        :param state: state in the form defined by save_current_state

        Reset the simulation to the given state, also overwrites the initial condition defined in the init method,
        therefore the reset method will reset the simulation state to the state provided to this method

        Should be overwritten by child class if said child adds entities to the simulation
        """
        self.p = copy.deepcopy(state[:2])
        self.p_dot = copy.deepcopy(state[2:4])
        self.c = copy.deepcopy(state[4:6])
        self.grasped = copy.deepcopy(state[6])
# SimplifiedBiGraspingEnv end


# SimplifiedBiHandoverEnv
class SimplifiedBiHandoverEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "human-shadows"]}

    def __init__(self, render_mode='human'):
        super(SimplifiedBiHandoverEnv, self).__init__()
        self.p_0_init = np.array([-1.0, 1.0])
        self.p_0_dot_init = np.array([0.0, 0.0])
        self.p_1_init = np.array([1.0, 1.0])
        self.p_1_dot_init = np.array([0.0, 0.0])
        self.grasped_0_init = False
        self.grasped_1_init = False
        self.h = 1
        self.w = 1
        self.dt = 0.01

        self.c_0 = np.array([0.0, 0.0])
        self.cube_init_std = 0.1

        self.p_0 = copy.deepcopy(self.p_0_init)
        self.p_0_dot = copy.deepcopy(self.p_0_dot_init)
        self.p_1 = copy.deepcopy(self.p_1_init)
        self.p_1_dot = copy.deepcopy(self.p_1_dot_init)
        self.c = copy.deepcopy(self.c_0)
        self.grasped_0 = copy.deepcopy(self.grasped_0_init)
        self.grasped_1 = copy.deepcopy(self.grasped_1_init)

        self._max_episode_steps = 1000

        self.observation_space = spaces.Box(np.array([-1] * 12), np.array([1] * 12), dtype=np.float32)
        self.observation_dim = self.observation_space.shape[0]

        self.action_space = spaces.Box(np.array([-1] * 6), np.array([1] * 6), dtype=np.float32)
        self.action_dim = self.action_space.shape[0]
        self.max_action = 1

        self.window_size = (2 * 300, 300)  # The size of the PyGame window

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.t_step = 0

        self.load_simulation()
        self.reset()

    def load_simulation(self):
        if self.render_mode == 'human' or self.render_mode == 'human-shadows':
            # Initialize PyGame
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Simple Grasping Robot Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

    def _get_obs(self):
        obs = np.zeros((12,))
        obs[:2] = self.p_0
        obs[2:4] = self.p_0_dot
        obs[4:6] = self.p_1
        obs[6:8] = self.p_1_dot
        obs[8:10] = self.c
        obs[10] = int(self.grasped_0)
        obs[10] = int(self.grasped_1)
        return copy.deepcopy(obs)

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        return {'task_solved': False}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t_step = 0

        # Reset agent location
        self.p_0 = copy.deepcopy(self.p_0_init)
        self.p_0_dot = copy.deepcopy(self.p_0_dot_init)
        self.p_1 = copy.deepcopy(self.p_1_init)
        self.p_1_dot = copy.deepcopy(self.p_1_dot_init)
        self.c = copy.deepcopy(self.c_0)
        self.grasped_0 = copy.deepcopy(self.grasped_0_init)
        self.grasped_1 = copy.deepcopy(self.grasped_1_init)
        self.c = copy.deepcopy(self.c_0)
        self.c[0] += np.clip(0.5 + np.random.randn() * self.cube_init_std, 0, 1)

        if self.render_mode == 'human' or self.render_mode == 'human-shadows':
            self.screen.fill((255, 255, 255))  # Fill the screen with white

        return self._get_obs()

    def step(self, action):
        p_0_dot = np.clip(action[:2], -1, 1)
        p_1_dot = np.clip(action[2:4], -1, 1)
        p_0 = self.p_0
        p_1 = self.p_1
        c = self.c
        self.grasped_0 = action[4] >= 0 and np.linalg.norm(p_0 - c) < 0.05
        self.grasped_1 = action[5] >= 0 and np.linalg.norm(p_1 - c) < 0.05

        self.p_0 += p_0_dot * self.dt
        self.p_1 += p_1_dot * self.dt

        if self.grasped_1:  # cube is grasped
            self.c += p_1_dot * self.dt # cube and gripper move at the same velocity
        elif self.grasped_0:
            self.c += p_0_dot * self.dt  # cube and gripper move at the same velocity
        else:
            self.c[1] += -1 * self.dt

        self.p_0[0] = np.clip(self.p_0[0], -1, 0)
        self.p_0[1] = np.clip(self.p_0[1], 0, 1)
        self.p_1[0] = np.clip(self.p_1[0], 0, 1)
        self.p_1[1] = np.clip(self.p_1[1], 0, 1)
        self.c[0] = np.clip(self.c[0], -1, 1)
        self.c[1] = np.clip(self.c[1], 0, 1)

        info = self._get_info()

        self.t_step += 1
        if self.render_mode == 'human':
            self.render(mode=self.render_mode)  # Call the render method to update the display
        if self.render_mode == 'human-shadows' and self.t_step % 20 == 0:
            self.render(mode=self.render_mode)
        obs = self._get_obs()
        reward, rewards_dict = self.reward_fun(obs, action)

        return obs, reward, self.termination_condition() or info['task_solved'], rewards_dict, info

    def termination_condition(self):
        """
            Should be overwritten by subclass
        """
        return False

    def reward_fun(self, observation, action):
        info = self._get_info()

        rewards_dict = {}

        reward = 0

        return reward, rewards_dict

    def render(self, mode="human"):
        if mode == 'human':
            self.screen.fill((255, 255, 255))  # Fill the screen with white
        # print(self.p)
        # Draw the agent
        agent_0_pos = self._to_screen_coordinates(self.p_0)
        agent_1_pos = self._to_screen_coordinates(self.p_1)
        # print(agent_pos)
        if self.grasped_0:
            pygame.draw.circle(self.screen, (150, 150, 255), agent_0_pos, 10)
        else:
            pygame.draw.circle(self.screen, (150, 255, 150), agent_0_pos, 10)

        if self.grasped_1:
            pygame.draw.circle(self.screen, (150, 150, 255), agent_1_pos, 10)
        else:
            pygame.draw.circle(self.screen, (150, 255, 150), agent_1_pos, 10)


        # Draw the cube
        cube_pos = self._to_screen_coordinates(self.c)
        rect_pos = pygame.Rect(cube_pos[0] - self.window_size[0] / 20, cube_pos[1], self.window_size[0] / 20,
                               self.window_size[1] / 20)
        pygame.draw.rect(self.screen, (255, 0, 0), rect_pos)  # Draw cube

        pygame.image.save(self.screen, "screen.jpg")

        # Update the display
        pygame.display.flip()
        self.clock.tick(200)  # Limit the frame rate to 60 FPS

    def _to_screen_coordinates(self, pos):
        """
        Convert position in environment coordinates to screen coordinates.
        Environment coordinates range from 0 to 5 in x and -1 to 1 in y.
        Screen coordinates range from 0 to window_size.
        """
        return (
            int((pos[0] + 1) * (self.window_size[0] / 2)),
            # int((1 - pos[1]) * self.window_size[1]*0.95 - self.window_size[1]*0.05),
            # int(self.window_size[1]*0.95 - pos[1] *self.window_size[1]*0.95),
            int(self.window_size[1] * 0.95 * (1 - pos[1]))
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
# SimplifiedBiHandoverEnv end

# YumiEnv
class YumiEnv(gym.Env):
    """
        Base environment containing Yumi robot simulated in Pybullet.
        All environments that use the Yumi robot should inherit from this class.

        The environment implements the functions to train a Reinforcement Learning Agent with continuous state and action space
        The robot is operated by controlling the linear and angular velocity of the end effectors, as well as the opening/closing
        speed of the gripper.
        The state consists of the pose of the end effectors and the positions of the grippers' joint.

        Class inhering for this have the same state and action space, but they can be masked to control a subset of the action space
        or to observe a portion of the state.

        This class implements a reward function that encourages the agent to move the end effectors close.

    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, control_mode='joints', plane_height=0.0):
        """
        :param render_mode: None/human/rgb_array
        :param control_mode: joints/cartesian
        """

        self.simulation_rate = 1

        self.yumi_params = {
            'pos_noise': 0.00,
            'vel_noise': 0.00,
            'gripper_pos_noise': 0.00,
            'gripper_vel_noise': 0.00,
            'object_pos_noise': 0.00,
            'object_orient_noise': 0.00,
        }

        self.joints_limits = {'left': {'lower': [-2.94087978961, -2.50454747661, -2.9408797896, -2.15548162621,
                                                 -5.06145483078, -1.53588974176, -3.99680398707],
                                       'upper': [+2.94087978961, 0.759218224618, +2.9408797896, 1.3962634016,
                                                 5.06145483078, 2.40855436775, 3.99680398707]},
                              'right': {'lower': [-2.94087978961, -2.50454747661, -2.9408797896, -2.15548162621,
                                                  -5.06145483078, -1.53588974176, -3.99680398707],
                                        'upper': [+2.94087978961, 0.759218224618, +2.9408797896, 1.3962634016,
                                                  5.06145483078, 2.40855436775, 3.99680398707]}
                              }
        self.robot_joints_ul = ([0.0, 0.0] + self.joints_limits['left']['upper'] + [0.0, 0.0] +
                                self.joints_limits['right']['upper'] + [0.0, 0.0, 0.0])
        self.robot_joints_ll = ([0.0, 0.0] + self.joints_limits['left']['lower'] + [0.0, 0.0] +
                                self.joints_limits['right']['lower'] + [0.0, 0.0, 0.0])
        self.robot_joints_ranges = list(np.array(self.robot_joints_ul) - np.array(self.robot_joints_ll))
        self.robot_joints_rest = list((np.array(self.robot_joints_ul) + np.array(self.robot_joints_ll)) / 2)
        """
        robot_joints = [0.0, 0.0] + list(joints_vel[7:]) + [0.0] + list([left_gripper_vel]) + [0.0] + 
        list(joints_vel[:7]) + [0.0, 0.0] + list([right_gripper_vel])
        """
        self.control_mode = control_mode

        if self.control_mode == 'cartesian':
            # Observations are dictionaries with the agent's state.
            self.observation_space = spaces.Dict(
                {
                    # State is Pose and velocity (linear and angular) of both end effectors + gripper joint
                    "agent": spaces.Box(-1, 1, shape=((6 + 1) * 4,), dtype=float),
                }
            )
            self.observation_dim = self.observation_space['agent'].shape[0]

            # Action space is velocities (linear and angular) + gripper joint of arm
            self.action_space = spaces.Box(-1, 1, shape=((6 + 1) * 2,), dtype=float)
            self.action_dim = self.action_space.shape[0]
            self.max_action = 1
            self.measurement_noise = np.array(
                [yumi_params['pos_noise']]*6 + [yumi_params['vel_noise']]*6 +
                [yumi_params['gripper_pos_noise']] + [yumi_params['gripper_vel_noise']] +
                [yumi_params['pos_noise']] * 6 + [yumi_params['vel_noise']] * 6 +
                [yumi_params['gripper_pos_noise']] + [yumi_params['gripper_vel_noise']])

        elif self.control_mode == 'joints':
            # Observations are dictionaries with the agent's state.
            self.observation_space = spaces.Dict(
                {
                    # State is position and velocity of arm joints + gripper joint
                    "agent": spaces.Box(-1, 1, shape=((7 + 1) * 4,), dtype=float),
                }
            )
            self.observation_dim = self.observation_space['agent'].shape[0]

            # Action space is velocities of arm and gripper joint of each arm
            self.action_space = spaces.Box(-1, 1, shape=((7 + 1) * 2,), dtype=float)
            self.action_dim = self.action_space.shape[0]
            self.max_action = 1

            self.measurement_noise = np.array(
                [yumi_params['pos_noise']] * 7 + [yumi_params['gripper_pos_noise']] +
                [yumi_params['vel_noise']] * 7 + [yumi_params['gripper_vel_noise']] +
                [yumi_params['pos_noise']] * 7 + [yumi_params['gripper_pos_noise']] +
                [yumi_params['vel_noise']] * 7 + [yumi_params['gripper_vel_noise']])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.physicsClient = p.connect(p.GUI)
        elif self.render_mode == "rgb_array":
            raise NotImplementedError
        elif self.render_mode is None:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        self.dt = time_step / self.simulation_rate
        p.setTimeStep(self.dt)
        self._max_episode_steps = 1000

        p.setGravity(0.0, 0.0, gravity_constant)

        self.robot_id = p.loadURDF(yumi_urdf_path, [0, 0, 0],
                                   flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, -0.1], [0, 0, 0, 1])
        self.numJoints = p.getNumJoints(self.robot_id)
        sim_utils.setJointPosition(self.robot_id, [0.0] * self.numJoints)
        p.stepSimulation()

        self.end_effector_left_idx = 21
        self.end_effector_right_idx = 10

        self.max_gripper_width = 0.1
        self.max_gripper_speed = 0.1
        self.max_end_effector_speed = 0.1
        self.max_end_effector_angular_vel = 1.0

        self.max_episode_steps = 1e3

        self.q_0 = [0.0] * self.numJoints
        
        self.q_0[2] = 27.4418 * np.pi / 180
        self.q_0[3] = -126.984 * np.pi / 180
        self.q_0[4] = -114.068  * np.pi / 180
        self.q_0[4] = 35.8741  * np.pi / 180
        self.q_0[4] = 4.63424  * np.pi / 180
        self.q_0[4] = 58.1386  * np.pi / 180
        self.q_0[4] = -2.92198  * np.pi / 180

        self.q_0[12] = -np.pi
        self.q_0[14] = -np.pi
        self.q_0[17] = np.pi / 4
        self.dq_0 = [0.0] * self.numJoints

        self.plane_height = plane_height

        self.load_simulation()
        self.reset()

    def set_simulation_rate(self, simulation_rate):
        self.simulation_rate = simulation_rate

    def increase_task_difficulty(self):
        for k in self.yumi_params:
            self.yumi_params[k] += 0.001

    def load_simulation(self):
        """
            To be overridden by subclasses.
            This is where objects should be loaded
        """
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, self.plane_height])

    def getJacobian(self, link_idx):
        mpos, mvel, mtorq = sim_utils.getMotorJointStates(self.robot_id)

        result = p.getLinkState(self.robot_id,
                                link_idx,
                                computeLinkVelocity=1,
                                computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        zero_vec = [0.0] * len(mpos)
        jac_t, jac_r = p.calculateJacobian(self.robot_id, link_idx, com_trn, mpos, zero_vec, zero_vec)

        J_t = np.asarray(jac_t)
        J_r = np.asarray(jac_r)
        J = np.concatenate((J_t, J_r), axis=0)

        return J

    def getJacobians(self):
        J_right = self.getJacobian(self.end_effector_right_idx)
        J_left = self.getJacobian(self.end_effector_left_idx)
        return J_left, J_right

    def _get_obs(self):
        if self.control_mode == 'cartesian':
            left_end_effector_pose, left_end_effector_vel = (
                sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx, return_vel=True))
            right_end_effector_pose, right_end_effector_vel = (
                sim_utils.getLinkPose(self.robot_id, self.end_effector_right_idx, return_vel=True))

            pos, vel, _ = sim_utils.getJointStates(self.robot_id)

            left_gripper = pos[self.end_effector_left_idx] / self.max_gripper_width
            left_gripper_vel = vel[self.end_effector_left_idx]

            right_gripper = pos[self.end_effector_right_idx] / self.max_gripper_width
            right_gripper_vel = vel[self.end_effector_right_idx]

            obs = np.array(list(left_end_effector_pose) + list(left_end_effector_vel) + [left_gripper, left_gripper_vel]
                           + list(right_end_effector_pose) + list(right_end_effector_vel) + [right_gripper,
                                                                                             right_gripper_vel])
        elif self.control_mode == 'joints':
            pos, vel, _ = sim_utils.getJointStates(self.robot_id)
            obs = np.array(pos[2:2 + 7] + [pos[10]] + vel[2:2 + 7] + [vel[10]] +
                           pos[12:12 + 7] + [pos[21]] + vel[12:12 + 7] + [vel[21]])

        obs += np.random.randn(obs.shape[0]) * self.measurement_noise

        return obs

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        for idx in range(self.numJoints):
            p.resetJointState(self.robot_id, idx, self.q_0[idx], self.dq_0[idx])

        obs = self._get_obs()

        return obs


    def step(self, action):

        if self.control_mode == 'cartesian':
            left_end_effector_vel = list(action[:3] * self.max_end_effector_speed)
            left_end_effector_ang_vel = list(action[3:6] * self.max_end_effector_angular_vel)
            left_gripper_vel = action[6] * self.max_gripper_speed
            right_end_effector_vel = list(action[7:10] * self.max_end_effector_speed)
            right_end_effector_ang_vel = list(action[10:13] * self.max_end_effector_angular_vel)
            right_gripper_vel = action[13] * self.max_gripper_speed

            J_l, J_r = self.getJacobians()
            try:
                joints_vel = np.dot(np.linalg.pinv(J_l), np.array(left_end_effector_vel+left_end_effector_ang_vel))
                joints_vel += np.dot(np.linalg.pinv(J_r), np.array(right_end_effector_vel+right_end_effector_ang_vel))
            except np.linalg.LinAlgError as e:
                joints_vel = [0.0] * J_l.shape[1]
                print(e)

            robot_joints = [0.0, 0.0] + list(joints_vel[:7]) + [0.0] + [right_gripper_vel, right_gripper_vel] + list(
                joints_vel[9:16]) + [0.0, left_gripper_vel, left_gripper_vel]
        elif self.control_mode == 'joints':
            joints_vel = action[:14]
            left_gripper_vel = action[15]
            right_gripper_vel = action[14]
            robot_joints = [0.0, 0.0] + list(joints_vel[7:]) + [0.0] + [right_gripper_vel, right_gripper_vel] + list(
                joints_vel[:7]) + [0.0, left_gripper_vel, left_gripper_vel]


        p.setJointMotorControlArray(self.robot_id,
                                    list(range(self.numJoints)),
                                    p.VELOCITY_CONTROL,
                                    targetVelocities=robot_joints,
                                    forces=[max_force] * self.numJoints)
        for _ in range(self.simulation_rate):
            p.stepSimulation()

        observation = self._get_obs()
        info = self._get_info()
        reward, rewards_dict = self.reward_fun(observation, action)

        return observation, reward, self.termination_condition() or info['task_solved'], rewards_dict, info

    def reward_fun(self, observation, action):
        """
            Should be overwritten by subclass
            This version is just rewarding the agent if the end effectors are close
        """
        reward = 0.0
        rewards_dict = {}  # Defines the reward components

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict


    def termination_condition(self):
        """
            Should be overwritten by subclass
        """
        return False

    def _get_info(self):
        """
            Returns a dictionary with information about the environment
            Must contain the boolean field task_solved
        """
        
        return {'task_solved': False}

    def save_current_state(self):
        """
        :return: state representation of the simulation as a list of real numbers

        Should be overwritten by child class if said child adds entities to the simulation
        """
        pos, vel, _ = sim_utils.getJointStates(self.robot_id)

        return list(pos) + list(vel)

    def load_initial_state(self, state):
        """
        :param state: state in the form defined by save_current_state

        Reset the simulation to the given state, also overwrites the initial condition defined in the init method,
        therefore the reset method will reset the simulation state to the state provided to this method

        Should be overwritten by child class if said child adds entities to the simulation
        """
        self.q_0 = copy.deepcopy(state[:self.numJoints])
        self.dq_0 = copy.deepcopy(state[self.numJoints:])

    def check_collisions(self):
        """
            Checks if the robot is currently in contact with the ground.
            Child classes should extend this function if needed (e.g. more objects in the scene should be avoided).

            Return False if no collisions, True if collisions
            
        """
        contacts = p.getContactPoints(self.plane_id, self.robot_id) 
        if len(contacts) > 2:
            # print(contacts)
            return True
    def get_current_joint_state(self):
        pos, vel, _ = sim_utils.getJointStates(self.robot_id)
        return pos[2:2 + 7], vel[2:2 + 7], pos[12:12 + 7], vel[12:12 + 7]
# YumiEnv end

# YumiLeftEnv
class YumiLeftEnv(YumiEnv):
    """
        Environment to train a policy for a task with left arm of the Yumi robot

        The robot is initialized to an initial configuration.
        The observations are masked to the left arm
    """

    def __init__(self, render_mode, control_mode='cartesian', plane_height=0.0):
        super().__init__(render_mode=render_mode, control_mode=control_mode, plane_height=plane_height)
        if self.control_mode == 'cartesian':
            self.observation_space = spaces.Dict(
                {
                    # State is Pose and velocity (lin and ang) of both end effectors + gripper joint of left arm
                    "agent": spaces.Box(-1, 1, shape=((6 + 1) * 2,), dtype=float),
                }
            )
            self.observation_dim = self.observation_space['agent'].shape[0]

            # Action space is velocities (linear and angular) + gripper joint of left arm
            self.action_space = spaces.Box(-1, 1, shape=((6 + 1),), dtype=float)
            self.action_dim = self.action_space.shape[0]
        elif self.control_mode == 'joints':
            self.observation_space = spaces.Dict(
                {
                    # State is position and velocity of left arm + gripper joints
                    "agent": spaces.Box(-1, 1, shape=((7 + 1) * 2,), dtype=float),
                }
            )
            self.observation_dim = self.observation_space['agent'].shape[0]

            # Action space is velocities of arm + gripper joint of left arm
            self.action_space = spaces.Box(-1, 1, shape=((7 + 1),), dtype=float)
            self.action_dim = self.action_space.shape[0]

    def step(self, action):
        if self.control_mode == 'cartesian':
            real_action = np.zeros((14,))
            real_action[:7] = action
        elif self.control_mode == 'joints':
            real_action = np.zeros((16,))
            real_action[:7] = action[:7]
            real_action[14] = action[7]

        return super().step(real_action)

    def reward_fun(self, observation, action):
        """
            Should be overwritten by subclass
        """
        reward = 0.0
        rewards_dict = {}  # Defines the reward components

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def termination_condition(self):
        """
            Should be overwritten by subclass
        """
        cube_pose = sim_utils.getObjPose(self.cube_id)

        return cube_pose[2] > 0.2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs = self._get_obs()
        return obs

    def _get_info(self):
        cube_pose = sim_utils.getObjPose(self.cube_id)
        return {'task_solved': cube_pose[2] > 0.2}

    def _get_obs(self):
        if self.control_mode == 'cartesian':
            left_end_effector_pose, left_end_effector_vel = (
                sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx, return_vel=True))
            pos, vel, _ = sim_utils.getJointStates(self.robot_id)
            left_gripper = pos[self.end_effector_left_idx] / self.max_gripper_width
            left_gripper_vel = vel[self.end_effector_left_idx]
            obs = np.array(
                list(left_end_effector_pose) + list(left_end_effector_vel) + [left_gripper, left_gripper_vel])
        elif self.control_mode == 'joints':
            pos, vel, _ = sim_utils.getJointStates(self.robot_id)
            obs = np.array(pos[12:12 + 7] + [pos[21]] + vel[12:12 + 7] + [vel[21]])

        obs += np.random.randn(obs.shape[0]) * self.measurement_noise[obs.shape[0]:]

        return obs

    def get_current_joint_state(self):
        pos, vel, _ = sim_utils.getJointStates(self.robot_id)
        return pos[12:12 + 7], vel[12:12 + 7]

    def check_fingers_touching(self, object_id):
        finger_1 = False
        finger_2 = False
        contacts = p.getContactPoints(object_id, self.robot_id)  # encourage touching the cube
        total_contact_force = 0.0
        for contact in contacts:
            if (contact[1] == object_id and contact[2] == self.robot_id and contact[4] == 20) or (
                    contact[1] == self.robot_id and contact[2] == object_id and contact[3] in 20):
                finger_1 = True
                total_contact_force += contact[9]
            if (contact[1] == object_id and contact[2] == self.robot_id and contact[4] == 21) or (
                    contact[1] == self.robot_id and contact[2] == object_id and contact[3] in 21):
                finger_2 = True
                total_contact_force += contact[9]
        total_contact_force = np.abs(total_contact_force)

        return finger_1, finger_2, total_contact_force

    def reset_from_state(self, state):
        self.load_initial_state(state)
        for idx in range(self.numJoints):
            p.resetJointState(self.robot_id, idx, self.q_0[idx], self.dq_0[idx])
        obs = self._get_obs()

        return obs

    def set_pose_twist(self, pose_twist):
        end_effector_pos = pose_twist[:3]
        end_effector_orient = pose_twist[3:3+3]
        left_end_effector_vel = pose_twist[3+3:3+3+6]
        joints_pos = sim_utils.ik_end_effector(self.robot_id, self.end_effector_left_idx, end_effector_pos,
                                               end_effector_orient, euler=True) #, self.robot_joints_ll, self.robot_joints_ul,
                                               #self.robot_joints_ranges, self.robot_joints_rest)
        q_0 = copy.deepcopy(self.q_0)
        q_0[12:12 + 7] = joints_pos[9:9 + 7]

        J_l, J_r = self.getJacobians()
        joints_vel = np.dot(np.linalg.pinv(J_l), np.array(left_end_effector_vel))

        for idx1, idx2 in zip(range(12, 12 + 7), range(9, 9 + 7)):
            p.resetJointState(self.robot_id, idx1, joints_pos[idx2], joints_vel[idx2])

        obs = self._get_obs()

        return obs

    def reset_joints(self, positions, velocities, seed=None, options=None):
        # Manual reset, because we need to reset the robot to the original q_0 in order to do the IK
        for idx1, idx2 in zip(range(12, 12 + 7), range(7)):
            p.resetJointState(self.robot_id, idx1, positions[idx2], velocities[idx2])

        obs = self._get_obs()

        return obs
    
    def reset_from_pose(self, pose, seed=None, options=None):
        end_effector_pos = pose[:3]
        end_effector_orient = pose[3:]
        # super().reset(seed=seed, options=options)

        joints_pos = sim_utils.ik_end_effector(self.robot_id, self.end_effector_left_idx, end_effector_pos,
                                               end_effector_orient)  # , self.robot_joints_ll, self.robot_joints_ul,
        # self.robot_joints_ranges, self.robot_joints_rest)
        # q_0 = copy.deepcopy(self.q_0)
        # q_0[12:12 + 7] = joints_pos[9:9 + 7]
        # Manual reset, because we need to reset the robot to the original q_0 in order to do the IK
        for idx1, idx2 in zip(range(12, 12 + 7), range(9, 9 + 7)):
            p.resetJointState(self.robot_id, idx1, joints_pos[idx2], 0.0)

        obs = self._get_obs()

        return obs
# YumiLeftEnv end

# YumiGraspEnv
class YumiGraspEnv(YumiLeftEnv):
    """
        Task: Grasp the cube using the left gripper

        Specifications:
        -The cube can be placed in a zone in front of the robot
        -The cube pose is estimated and added to the observation
        -The end effector is initialized to be close to the cube (about 15cm)
    """

    def __init__(self, render_mode, control_mode='cartesian'):
        self.cube_load_pos = [0.5, 0, 0.0125]
        self.cube_load_lin_vel = [0.0] * 3
        self.cube_load_orient = [0, 0, 0, 1]
        self.cube_load_ang_vel = [0.0] * 3

        self.cube_initial_position = [0.55, 0.0]
        # self.cube_initial_std = [0.005, 0.005, 5 * np.pi/180]  # x, y, yaw
        self.cube_initial_std = [0.0] * 3  # x, y, yaw

        self.initial_end_effector_displacement = 0.01
        self.initial_end_effector_angle_displacement = 2 * np.pi / 180  # 5 degrees
        # self.initial_end_effector_displacement = 0.0
        # self.initial_end_effector_angle_displacement = 0.0
        self.initial_end_effector_height = 0.1
        self.picked = False

        super().__init__(render_mode=render_mode, control_mode=control_mode)
        if self.control_mode == 'cartesian':
            self.observation_space = spaces.Dict(
                {
                    # State is Pose and velocity (lin and ang) of both end effectors + gripper joint of left arm + cube pose
                    "agent": spaces.Box(-1, 1, shape=((6 + 1) * 2 + 6,), dtype=float),
                }
            )
            self.observation_dim = self.observation_space['agent'].shape[0]
        elif self.control_mode == 'joints':
            self.observation_space = spaces.Dict(
                {
                    # State is position and velocity of left arm + gripper joints + cube pose
                    "agent": spaces.Box(-1, 1, shape=((7 + 1) * 2 + 6,), dtype=float),
                }
            )
            self.observation_dim = self.observation_space['agent'].shape[0]


    def load_simulation(self):
        super().load_simulation()
        self.cube_id = p.loadURDF(cube_urdf_path, self.cube_load_pos)


    def _get_obs(self):
        """
        :return: Arm pose (from mother class) + arm pose in cube's reference frame
        """
        obs_arm = super()._get_obs()
        # left_end_effector_pose = sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx, return_vel=False)
        cube_pose = np.array(sim_utils.getObjPose(self.cube_id))

        cube_pose[:3] += np.random.randn(3) * self.yumi_params['object_pos_noise']
        cube_pose[3:] += np.random.randn(3) * self.yumi_params['object_orient_noise']

        obs = np.array(list(obs_arm) + list(cube_pose))

        return obs

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed, options=options)

        # Sample cube pos
        cube_pos = [self.cube_initial_position[0] + np.random.randn() * self.cube_initial_std[0],
                    self.cube_initial_position[1] + np.random.randn() * self.cube_initial_std[1],
                    self.cube_load_pos[2]]
        self.cube_load_pos = cube_pos

        self.cube_load_orient = sim_utils.euler_to_quat([0.0, 0.0, np.random.randn() * self.cube_initial_std[2]])

        p.resetBasePositionAndOrientation(self.cube_id, self.cube_load_pos, self.cube_load_orient)
        p.resetBaseVelocity(self.cube_id, self.cube_load_lin_vel, self.cube_load_ang_vel)

        # Reset robot pose accordingly
        # Compute an end effector pose
        end_effector_displacement = [0.0] * 2 + [self.initial_end_effector_height]

        end_effector_pos = np.array(cube_pos) + np.array(end_effector_displacement)
        end_effector_orient = np.array([np.pi, 0, 0])
        super().reset(seed=seed, options=options)

        joints_pos = sim_utils.ik_end_effector(self.robot_id, self.end_effector_left_idx, end_effector_pos,
                                               end_effector_orient) #, self.robot_joints_ll, self.robot_joints_ul,
                                               #self.robot_joints_ranges, self.robot_joints_rest)
        q_0 = copy.deepcopy(self.q_0)
        q_0[12:12 + 7] = joints_pos[9:9 + 7]
        # Manual reset, because we need to reset the robot to the original q_0 in order to do the IK
        for idx in range(self.numJoints):
            p.resetJointState(self.robot_id, idx, q_0[idx], self.dq_0[idx])

        self.picked = False

        obs = self._get_obs()

        return obs


    def reward_fun(self, observation, action):
        """
            Should be overwritten by subclass
        """
        reward = 0.0
        rewards_dict = {}  # Defines the reward components

        cube_pose = np.array(sim_utils.getObjPose(self.cube_id))
        # info = self._get_info()
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))

        distance = np.linalg.norm(np.array(cube_pose[:3]) - np.array(end_effector_left_pose[:3]))

        # finger_1 = False
        # finger_2 = False
        # contacts = p.getContactPoints(self.cube_id, self.robot_id)  # encourage touching the cube
        # total_contact_force = 0.0
        # for contact in contacts:
        #     if (contact[1] == self.cube_id and contact[2] == self.robot_id and contact[4] == 20) or (
        #             contact[1] == self.robot_id and contact[2] == self.cube_id and contact[3] in 20):
        #         finger_1 = True
        #         total_contact_force += contact[9]
        #     if (contact[1] == self.cube_id and contact[2] == self.robot_id and contact[4] == 21) or (
        #             contact[1] == self.robot_id and contact[2] == self.cube_id and contact[3] in 21):
        #         finger_2 = True
        #         total_contact_force += contact[9]

        finger_1, finger_2, total_contact_force = self.check_fingers_touching(self.cube_id)

        if cube_pose[2] > 0.02:
            self.picked = True # change the state from non-picked to picked

        rewards_dict = {
            "distance_to_cube_reward": -distance,  # encourage getting close to the target cube
            "finger_contacts_reward":  1 * finger_1 + 1 * finger_2,  # encourage interaction with the target cube
            "cube_height_reward":      (100 + 1000 * (cube_pose[2] - 0.02)) * self.picked,
            "dropping_cube_reward":    -10 * (cube_pose[2] < 0.02 and self.picked),
            "losing_focus_reward":     -10 * (np.linalg.norm(cube_pose[:3] - end_effector_left_pose[:3]) > 0.10 and self.picked)
        }

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        cube_pose = sim_utils.getObjPose(self.cube_id)
        return {'task_solved': cube_pose[2] > 0.1}

    def termination_condition(self):
        """
            Should be overwritten by subclass
        """
        cube_pose = np.array(sim_utils.getObjPose(self.cube_id))
        info = self._get_info()
        end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))

        return (cube_pose[2] < 0.02 or np.linalg.norm(cube_pose[:3] - end_effector_left_pose[:3]) > 0.08) and self.picked

    def reset_from_state(self, state):
        q = copy.deepcopy(state[:self.numJoints])
        dq = copy.deepcopy(state[self.numJoints:2 * self.numJoints])
        cube_state = copy.deepcopy(state[2 * self.numJoints:])

        for idx in range(self.numJoints):
            p.resetJointState(self.robot_id, idx, q[idx], dq[idx])

        p.resetBasePositionAndOrientation(self.cube_id, cube_state[:3], sim_utils.euler_to_quat(cube_state[3:6]))
        p.resetBaseVelocity(self.cube_id, cube_state[6:9], cube_state[9:12])

        obs = self._get_obs()

        return obs

    def load_initial_state(self, state):
        """
        :param state: state in the form defined by save_current_state

        Reset the simulation to the given state, also overwrites the initial condition defined in the init method,
        therefore the reset method will reset the simulation state to the state provided to this method

        Should be overwritten by child class if said child adds entities to the simulation
        """
        self.q_0 = copy.deepcopy(state[:self.numJoints])
        self.dq_0 = copy.deepcopy(state[self.numJoints:2 * self.numJoints])
        cube_state = copy.deepcopy(state[2 * self.numJoints:])
        print(cube_state)
        self.cube_load_pos = cube_state[:3]
        self.cube_load_orient = sim_utils.euler_to_quat(cube_state[3:6])
        self.cube_load_lin_vel = cube_state[6:9]
        self.cube_load_ang_vel = cube_state[9:12]

    def save_current_state(self):
        """
        :return: state representation of the simulation as a list of real numbers

        Should be overwritten by child class if said child adds entities to the simulation
        """
        robot_state = super().save_current_state()
        cube_pose = sim_utils.getObjPose(self.cube_id)
        cube_vel = sim_utils.getObjVel(self.cube_id)
        return robot_state + list(cube_pose) + list(cube_vel)
# YumiGraspEnv end

# YumiPlaceEnv
class YumiPlaceEnv(YumiGraspEnv):
    def __init__(self, render_mode, control_mode='cartesian'):
        self.cube_target_pos = [0.3, 0.5, 0.0]

        super().__init__(render_mode=render_mode, control_mode=control_mode)


    def reward_fun(self, observation, action):
        reward = 0
        cube_pose = sim_utils.getObjPose(self.cube_id)

        reward += -np.linalg.norm(np.array(cube_pose[:3]) - np.array(self.cube_target_pos))  #

        if cube_pose[2] > 0.02:
            reward += 1

        if cube_pose[2] < 0.02:
            reward += -10

        info = self._get_info()
        if info['task_solved']:
            reward += 1000

        return reward


    def _get_info(self):
        cube_pose = sim_utils.getObjPose(self.cube_id)
        return {'task_solved': np.linalg.norm(np.array(cube_pose[:3]) - np.array(self.cube_target_pos)) < 0.05}

    def termination_condition(self):
        """
            Should be overwritten by subclass
        """
        cube_pose = sim_utils.getObjPose(self.cube_id)
        info = self._get_info()

        return cube_pose[2] < 0.02 or info['task_solved']
# YumiPlaceEnv end

# YumiGraspEnv_randomized_initial_position
class YumiGraspEnv_randomized_initial_position(YumiGraspEnv):
    """
        Task: Grasp the cube using the left gripper

        Specifications:
        -The cube can be placed in a zone in front of the robot
        -The cube pose is estimated and added to the observation
        -The end effector is initialized to be close to the cube (about 15cm)
    """

    def __init__(self, render_mode, control_mode='cartesian'):
        super().__init__(render_mode=render_mode, control_mode=control_mode)

    def reset(self, seed=None, options=None):

        # Sample cube pos
        cube_pos = [self.cube_initial_position[0] + np.random.randn() * self.cube_initial_std[0],
                    self.cube_initial_position[1] + np.random.randn() * self.cube_initial_std[1],
                    self.cube_load_pos[2]]
        self.cube_load_pos = cube_pos

        self.cube_load_orient = sim_utils.euler_to_quat([0.0, 0.0, np.random.randn() * self.cube_initial_std[2]])

        # Reset robot pose accordingly
        # Compute an end effector pose
        end_effector_displacement = list(2 * (np.random.rand(2) - 0.5) * self.initial_end_effector_displacement) + [
            self.initial_end_effector_height]

        end_effector_pos = np.array(cube_pos) + np.array(end_effector_displacement)
        end_effector_orient = np.array([np.pi, 0, 0])
        super().reset(seed=seed, options=options)

        joints_pos = sim_utils.ik_end_effector(self.robot_id, self.end_effector_left_idx, end_effector_pos,
                                               end_effector_orient, self.robot_joints_ll, self.robot_joints_ul,
                                               self.robot_joints_ranges, self.robot_joints_rest)
        q_0 = copy.deepcopy(self.q_0)
        q_0[12:12 + 7] = joints_pos[9:9 + 7]
        # Manual reset, because we need to reset the robot to the original q_0 in order to do the IK
        for idx in range(self.numJoints):
            p.resetJointState(self.robot_id, idx, q_0[idx], self.dq_0[idx])


        # print(joints_pos)
        # print(self.q_0)
        # print(len(self.q_0))
        # print(self.numJoints)
        # print('---------------------------------')

        self.picked = False

        obs = self._get_obs()

        return obs
# YumiGraspEnv_randomized_initial_position end

# YumiGraspEnv_randomized_initial_pose
class YumiGraspEnv_randomized_initial_pose(YumiGraspEnv):
    """
        Task: Grasp the cube using the left gripper

        Specifications:
        -The cube can be placed in a zone in front of the robot
        -The cube pose is estimated and added to the observation
        -The end effector is initialized to be close to the cube (about 15cm)
    """

    def __init__(self, render_mode, control_mode='cartesian'):
        super().__init__(render_mode=render_mode, control_mode=control_mode)

    def reset(self, seed=None, options=None):

        # # Sample cube pos
        # cube_pos = [self.cube_initial_position[0] + np.random.randn() * self.cube_initial_std[0],
        #             self.cube_initial_position[1] + np.random.randn() * self.cube_initial_std[1],
        #             self.cube_load_pos[2]]
        # self.cube_load_pos = cube_pos
        #
        # self.cube_load_orient = sim_utils.euler_to_quat([0.0, 0.0, np.random.randn() * self.cube_initial_std[2]])

        # Reset robot pose accordingly
        # Compute an end effector pose

        super().reset(seed=seed, options=options)  # this resets the arm to the original q_0

        end_effector_displacement = list(2 * (np.random.rand(2) - 0.5) * self.initial_end_effector_displacement) + [
            self.initial_end_effector_height]

        end_effector_pos = np.array(self.cube_load_pos) + np.array(end_effector_displacement)
        end_effector_orient = np.array([np.pi + np.random.randn() * self.initial_end_effector_angle_displacement,
                                        np.pi * np.random.randn() * self.initial_end_effector_angle_displacement,
                                        np.pi * np.random.randn() * self.initial_end_effector_angle_displacement])

        joints_pos = sim_utils.ik_end_effector(self.robot_id, self.end_effector_left_idx, end_effector_pos,
                                               end_effector_orient)
        q_0 = copy.deepcopy(self.q_0)
        q_0[12:12 + 7] = joints_pos[9:9 + 7]
        # Manual reset, because we need to reset the robot to the original q_0 in order to do the IK
        for idx in range(self.numJoints):
            p.resetJointState(self.robot_id, idx, q_0[idx], self.dq_0[idx])


        self.picked = False

        obs = self._get_obs()

        return obs

    def increase_task_difficulty(self):
        super().increase_task_difficulty()
        self.cube_initial_std = [self.cube_initial_std[0] + 0.001, self.cube_initial_std[1] + 0.001,
                                 self.cube_initial_std[2] + np.pi / 180]
# YumiGraspEnv_randomized_initial_pose end

# YumiLeftEnv_objects_in_scene
class YumiLeftEnv_objects_in_scene(YumiLeftEnv):
    """
        Task: interact with objects in the scene using the left gripper
        The reward function is not implemented yet

        Specifications:
        -Provide a dictionary of objects to load into the scene, each entry must have the following fields:
         -key: name of the object
         -load_pos:
         -load_quat:
         -pos_std:
         -orient_std:
    """

    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        self.objects_id = None
        self.objects_dict = env_config_dict["objects"]
        if "robot_joints_pos" in env_config_dict.keys():
            positions = env_config_dict["robot_joints_pos"]
            if np.abs(positions[0]) > 2 * np.pi:
                positions = [np.pi * p / 180 for p in positions]
            self.robot_joints_pos = positions
        if "plane_height" in env_config_dict.keys():
            plane_height = env_config_dict["plane_height"]
        else:
            plane_height = 0.0

        super().__init__(render_mode=render_mode, control_mode=control_mode, plane_height=plane_height)
        if "gripper_state" in env_config_dict.keys() and env_config_dict["gripper_state"] >= 0:
            # by default it is closed
            self.q_0[-1] = env_config_dict["gripper_state"]
            self.q_0[-2] = env_config_dict["gripper_state"]

        self.number_of_objects = len(self.objects_dict.keys())

        if self.control_mode == 'cartesian':
            self.observation_space = spaces.Dict(
                {
                    # State is Pose and velocity (lin and ang) of left end effector + gripper joint of left arm + objects pose
                    "agent": spaces.Box(-1, 1, shape=((6 + 1) * 2 + 6*self.number_of_objects,), dtype=float),
                }
            )
            self.observation_dim = self.observation_space['agent'].shape[0]
        elif self.control_mode == 'joints':
            self.observation_space = spaces.Dict(
                {
                    # State is position and velocity of left arm + gripper joints + objects pose
                    "agent": spaces.Box(-1, 1, shape=((7 + 1) * 2 + 6*self.number_of_objects,), dtype=float),
                }
            )
            self.observation_dim = self.observation_space['agent'].shape[0]
        self.memory={}

    def load_simulation(self):
        super().load_simulation()
        self.objects_id = {}
        for k in self.objects_dict.keys():
            self.objects_id[k] = p.loadURDF(self.objects_dict[k]['urdf_path'],
                                            self.objects_dict[k]['load_pos'],
                                            self.objects_dict[k]['load_quat'])


    def _get_obs(self):
        """
        :return: Arm pose (from mother class) + arm pose in cube's reference frame
        """
        obs_arm = super()._get_obs()
        obs = list(copy.deepcopy(obs_arm))

        for k in self.objects_dict.keys():
            object_id = self.objects_id[k]
            obj_pose = np.array(sim_utils.getObjPose(object_id))
            obj_pose[:3] += np.random.randn(3) * self.yumi_params['object_pos_noise']
            obj_pose[3:] += np.random.randn(3) * self.yumi_params['object_orient_noise']
            obs += list(copy.deepcopy(obj_pose))

        return np.array(obs)

    def reset(self, seed=None, options=None):
        for k in self.objects_dict.keys():
            # Sample obj pos
            load_pos = self.objects_dict[k]['load_pos']
            load_pos[0] += np.random.randn() * self.objects_dict[k]['pos_std']
            load_pos[1] += np.random.randn() * self.objects_dict[k]['pos_std']

            load_quat = (np.array(sim_utils.quat_to_euler(self.objects_dict[k]['load_quat']))
                         + np.array([0.0, 0.0, np.random.randn() * self.objects_dict[k]['orient_std']]))
            load_quat = sim_utils.euler_to_quat(load_quat)

            p.resetBasePositionAndOrientation(self.objects_id[k], load_pos, load_quat)
            p.resetBaseVelocity(self.objects_id[k], [0.0]*3, [0.0]*3)

        q_0 = copy.deepcopy(self.q_0)
        if self.robot_joints_pos is not None:
            q_0[12:12 + 7] = self.robot_joints_pos

        # Manual reset, because we need to reset the robot to the original q_0 in order to do the IK
        for idx in range(self.numJoints):
            p.resetJointState(self.robot_id, idx, q_0[idx], self.dq_0[idx])

        self.memory={}
        obs = self._get_obs()

        return obs


    def reward_fun(self, observation, action):
        """
            Should be overwritten by subclass

            Useful utilities:
            -to get the pose of an object in the scene at the current time call
            obj_pose = np.array(sim_utils.getObjPose(obj_id))
            -To check if the robot's fingers are touching an object in the scene call:
            finger_1, finger_2, total_contact_force = self.check_fingers_touching(self.cube_id)
            which returns a boolean for each finger and the contact force between the fingers and the object
            -These utilities need the id of the object (assigned by the simulation engine), you can retrieve them by
            accessing self.objects_id['obj_name'], obj_name is name in the configuration dict.

            -The initial positions (and orientation) of the object in the scene are stored in self.objects_dict,
            you can access position by calling self.objects_dict['obj_name']['load_pos']

            -To get the pose of the left end effector call
            end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))

            Must return reward, rewards_dict
            rewards_dict must be defined explicitly contain all the reward components, if needed with entries equal to
            zero.
        """
        reward = 0.0
        rewards_dict = {}  # Defines the reward components

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        return {'task_solved': False}

    def termination_condition(self):
        return False

    def save_current_state(self):
        """
        :return: state representation of the simulation as a list of real numbers

        Should be overwritten by child class if said child adds entities to the simulation
        """
        robot_state = copy.deepcopy(super().save_current_state())

        for k in self.objects_dict.keys():
            object_id = self.objects_id[k]
            obj_pose = np.array(sim_utils.getObjPose(object_id))
            obj_vel = np.array(sim_utils.getObjVel(object_id))
            robot_state += list(copy.deepcopy(obj_pose)) + list(copy.deepcopy(obj_vel))

        return robot_state

    def reset_from_state(self, state):
        q = copy.deepcopy(state[:self.numJoints])
        dq = copy.deepcopy(state[self.numJoints:2 * self.numJoints])
        objs_state = copy.deepcopy(state[2 * self.numJoints:])

        for idx in range(self.numJoints):
            p.resetJointState(self.robot_id, idx, q[idx], dq[idx])

        for i, k in enumerate(self.objects_dict.keys()):
            object_id = self.objects_id[k]
            obj_state = objs_state[i*12:(i+1)*12]
            p.resetBasePositionAndOrientation(object_id, obj_state[:3], sim_utils.euler_to_quat(obj_state[3:6]))
            p.resetBaseVelocity(object_id, obj_state[6:9], obj_state[9:12])

        obs = self._get_obs()
        self.memory={}

        return obs
# YumiLeftEnv_objects_in_scene end


# YumiEnv_objects_in_scene
class YumiEnv_objects_in_scene(YumiEnv):
    """
        Task: interact with objects in the scene using the Yumi robot in bi-manual mode
        The reward function is not implemented

        Specifications:
        -Provide a dictionary of objects to load into the scene, each entry must have the following fields:
         -key: name of the object
         -load_pos:
         -load_quat:
         -pos_std:
         -orient_std:
    """

    def __init__(self, render_mode, env_config_dict, control_mode='cartesian'):
        self.objects_id = None
        self.objects_dict = env_config_dict["objects"]
        if "left_arm_joints_pos" in env_config_dict.keys():
            positions = env_config_dict["left_arm_joints_pos"]
            if np.abs(positions[0]) > 2 * np.pi:
                positions = [np.pi * p / 180 for p in positions]
            self.left_arm_joints_pos = positions
        if "right_arm_joints_pos" in env_config_dict.keys():
            positions = env_config_dict["right_arm_joints_pos"]
            if np.abs(positions[0]) > 2 * np.pi:
                positions = [np.pi * p / 180 for p in positions]
            self.right_arm_joints_pos = positions
        if "plane_height" in env_config_dict.keys():
            plane_height = env_config_dict["plane_height"]
        else:
            plane_height = 0.0

        super().__init__(render_mode=render_mode, control_mode=control_mode, plane_height=plane_height)
        if "left_gripper_state" in env_config_dict.keys() and env_config_dict["left_gripper_state"] >= 0:
            # by default it is closed
            self.q_0[-1] = env_config_dict["left_gripper_state"]
            self.q_0[-2] = env_config_dict["left_gripper_state"]
        if "right_gripper_state" in env_config_dict.keys() and env_config_dict["right_gripper_state"] >= 0:
            # by default it is closed
            self.q_0[10] = env_config_dict["right_gripper_state"]
            self.q_0[11] = env_config_dict["right_gripper_state"]

        self.number_of_objects = len(self.objects_dict.keys())

        if self.control_mode == 'cartesian':
            self.observation_space = spaces.Dict(
                {
                    # State is Pose and velocity (lin and ang) of end effectors + gripper joint of left arm + objects pose
                    "agent": spaces.Box(-1, 1, shape=((6 + 1) * 2 * 2 + 6*self.number_of_objects,), dtype=float),
                }
            )
            self.observation_dim = self.observation_space['agent'].shape[0]
        elif self.control_mode == 'joints':
            self.observation_space = spaces.Dict(
                {
                    # State is position and velocity of left and right arm + gripper joints + objects pose
                    "agent": spaces.Box(-1, 1, shape=((7 + 1) * 2 * 2 + 6*self.number_of_objects,), dtype=float),
                }
            )
            self.observation_dim = self.observation_space['agent'].shape[0]
        self.memory={}

    def load_simulation(self):
        super().load_simulation()
        self.objects_id = {}
        for k in self.objects_dict.keys():
            self.objects_id[k] = p.loadURDF(self.objects_dict[k]['urdf_path'],
                                            self.objects_dict[k]['load_pos'],
                                            self.objects_dict[k]['load_quat'])


    def _get_obs(self):
        """
        :return: Arm pose (from mother class) + arm pose in cube's reference frame
        """
        obs_arm = super()._get_obs()
        obs = list(copy.deepcopy(obs_arm))

        for k in self.objects_dict.keys():
            object_id = self.objects_id[k]
            obj_pose = np.array(sim_utils.getObjPose(object_id))
            obj_pose[:3] += np.random.randn(3) * self.yumi_params['object_pos_noise']
            obj_pose[3:] += np.random.randn(3) * self.yumi_params['object_orient_noise']
            obs += list(copy.deepcopy(obj_pose))

        return np.array(obs)

    def reset(self, seed=None, options=None):
        for k in self.objects_dict.keys():
            # Sample obj pos
            load_pos = self.objects_dict[k]['load_pos']
            load_pos[0] += np.random.randn() * self.objects_dict[k]['pos_std']
            load_pos[1] += np.random.randn() * self.objects_dict[k]['pos_std']

            load_quat = (np.array(sim_utils.quat_to_euler(self.objects_dict[k]['load_quat']))
                         + np.array([0.0, 0.0, np.random.randn() * self.objects_dict[k]['orient_std']]))
            load_quat = sim_utils.euler_to_quat(load_quat)

            p.resetBasePositionAndOrientation(self.objects_id[k], load_pos, load_quat)
            p.resetBaseVelocity(self.objects_id[k], [0.0]*3, [0.0]*3)

        q_0 = copy.deepcopy(self.q_0)
        if self.left_arm_joints_pos is not None:
            q_0[12:12 + 7] = self.left_arm_joints_pos
        if self.right_arm_joints_pos is not None:
            q_0[2:2 + 7] = self.right_arm_joints_pos

        # Manual reset
        for idx in range(self.numJoints):
            p.resetJointState(self.robot_id, idx, q_0[idx], self.dq_0[idx])

        self.memory={}
        obs = self._get_obs()

        return obs


    def check_fingers_touching(self, object_id):
        left_finger_1 = False
        left_finger_2 = False
        right_finger_1 = False
        right_finger_2 = False
        contacts = p.getContactPoints(object_id, self.robot_id)  # encourage touching the cube
        for contact in contacts:
            if (contact[1] == object_id and contact[2] == self.robot_id and contact[4] == 20) or (
                    contact[1] == self.robot_id and contact[2] == object_id and contact[3] in 20):
                left_finger_1 = True
            if (contact[1] == object_id and contact[2] == self.robot_id and contact[4] == 21) or (
                    contact[1] == self.robot_id and contact[2] == object_id and contact[3] in 21):
                left_finger_2 = True
            if (contact[1] == object_id and contact[2] == self.robot_id and contact[4] == 10) or (
                    contact[1] == self.robot_id and contact[2] == object_id and contact[3] in 10):
                right_finger_1 = True
            if (contact[1] == object_id and contact[2] == self.robot_id and contact[4] == 11) or (
                    contact[1] == self.robot_id and contact[2] == object_id and contact[3] in 11):
                right_finger_2 = True


        return left_finger_1, left_finger_2, right_finger_1, right_finger_2

    def reward_fun(self, observation, action):
        """
            Should be overwritten by subclass

            Useful utilities:
            -to get the pose of an object in the scene at the current time call
            obj_pose = np.array(sim_utils.getObjPose(obj_id))
            -To check if the robot's fingers are touching an object in the scene call:
            left_finger_1, left_finger_2, right_finger_1, right_finger_2 = self.check_fingers_touching(self.object_id)
            which returns a boolean for each finger and the contact force between the fingers and the object
            -These utilities need the id of the object (assigned by the simulation engine), you can retrieve them by
            accessing self.objects_id['obj_name'], obj_name is name in the configuration dict.

            -The initial positions (and orientation) of the object in the scene are stored in self.objects_dict,
            you can access position by calling self.objects_dict['obj_name']['load_pos']

            -To get the pose of the left end effector call
            end_effector_left_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_left_idx))

            -To get the pose of the right end effector call
            end_effector_right_pose = np.array(sim_utils.getLinkPose(self.robot_id, self.end_effector_right_idx))

            Must return reward, rewards_dict
            rewards_dict must be defined explicitly contain all the reward components, if needed with entries equal to
            zero.
        """
        reward = 0.0
        rewards_dict = {}  # Defines the reward components

        total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])
        total_bonuses = total_bonuses if total_bonuses > 0 else 1
        info = self._get_info()
        total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])
        task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']

        reward = total_shaping + task_solved_reward

        rewards_dict['task_solved_reward'] = task_solved_reward

        return reward, rewards_dict

    def _get_info(self):
        return {'task_solved': False}

    def termination_condition(self):
        return False

    def save_current_state(self):
        """
        :return: state representation of the simulation as a list of real numbers

        Should be overwritten by child class if said child adds entities to the simulation
        """
        robot_state = copy.deepcopy(super().save_current_state())

        for k in self.objects_dict.keys():
            object_id = self.objects_id[k]
            obj_pose = np.array(sim_utils.getObjPose(object_id))
            obj_vel = np.array(sim_utils.getObjVel(object_id))
            robot_state += list(copy.deepcopy(obj_pose)) + list(copy.deepcopy(obj_vel))

        return robot_state

    def reset_from_state(self, state):
        q = copy.deepcopy(state[:self.numJoints])
        dq = copy.deepcopy(state[self.numJoints:2 * self.numJoints])
        objs_state = copy.deepcopy(state[2 * self.numJoints:])

        for idx in range(self.numJoints):
            p.resetJointState(self.robot_id, idx, q[idx], dq[idx])

        for i, k in enumerate(self.objects_dict.keys()):
            object_id = self.objects_id[k]
            obj_state = objs_state[i*12:(i+1)*12]
            p.resetBasePositionAndOrientation(object_id, obj_state[:3], sim_utils.euler_to_quat(obj_state[3:6]))
            p.resetBaseVelocity(object_id, obj_state[6:9], obj_state[9:12])

        obs = self._get_obs()
        self.memory={}

        return obs
# YumiEnv_objects_in_scene end

# Example usage
if __name__ == "__main__":
    env = MobileRobotCorridorEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, info = env.step(action)
    env.close()
