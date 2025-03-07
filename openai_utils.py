import os
from openai import AzureOpenAI
import re

deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME")


def get_openai_client():
    client = AzureOpenAI(
        azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
        #azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=os.getenv("OPENAI_API_KEY"),
        # api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-02-01"
    )
    return client


def execute_prompt(client, prompt):
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    resp = response.choices[0].message.content
    return resp

def execute_code_prompt(client, code_prompt):
    response = execute_prompt(client, code_prompt)
    try:
        code = re.split('```[a-zA-Z0-9- ]*', response, maxsplit=10)[1]
        return code
    except Exception as e:
        print(e)
        print(response)
        return response

def class_code_prompt(client, prompt):
    response = execute_code_prompt(client, prompt)

    return response[response.find('class '):]


def create_prompt(task_definition, base_environment_code, new_class_id, terminal_reward=True):
    prompt = ("You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as " +
              "effective as possible. Your goal is to write a dense reward function for the environment that will help the agent " +
              "learn the task described in text. Your reward function should use useful variables from the environment\n"
              "Write a python class that satisfies the following requirements:\n" +
              "-The new class must inherit from the class defined in the code snipped included after the " +
              "requirements\n" +
              "-The class' name must be GeneratedEnv_"+str(new_class_id) + "\n" +
              "-The new class must only redefine the reward_fun and _get_info methods \n" +
              "-The new class must not redefine the constructor (the __init__ function) \n" +
              "-The new class must implement a reward function and a _get_info encoding " +
              "the following task for the environment:\n\n"
              + task_definition + "\n" +
              "-The reward_fun method must define a dictionary named reward_dict, reward_dict must contain all the "
              "values of the reward components, excluding the reward for solving the task. "
              "Then the method must conclude the following code: \n" )

    if terminal_reward:
        prompt +=("total_bonuses = sum([rewards_dict[k] if rewards_dict[k] > 0 else 0 for k in rewards_dict.keys()])\n" +
              "total_bonuses = total_bonuses if total_bonuses > 0 else 1\n" +
              "task_solved_reward = 10 * self._max_episode_steps * total_bonuses * info['task_solved']\n" +
              "total_shaping = sum([rewards_dict[k] for k in rewards_dict.keys()])\n" +
              "reward = total_shaping + task_solved_reward\n" +
              "rewards_dict['task_solved_reward'] = task_solved_reward\n" +
              "return reward, rewards_dict\n\n")
    else:
        prompt += ("reward = sum([rewards_dict[k] for k in rewards_dict.keys()])\n" +
                   "return reward, rewards_dict\n\n")

    prompt += ("The _get_info method must return a dict that contains a boolean field task_solved, that is computed "
              "according to the task description provided\n\n" +

              "If required by the prompt, the new class can implement a termination_condition method, that returns True if a failure condition is detected. "
              "Not all tasks require this failure condition to be implemented. \n\n"

              "\nHere is some advice on how to design reward functions:\n" +
              "-The reward function should implement a shaping that clearly guides the policy towards the goal\n" +
              "-You can give a rewarding bonus if the task is partially solved.\n" +
              "-If the environment has a termination condition that can halt an episode before reaching the goal area or "
              "configuration, then consider to use positive shaping terms in the reward function.\n" +
              "-The reward components should not be too disproportionate\n"
              "-To incentive to get close to an object you should reward the decrease of distance and the contacts with said "
              "objects, if you want to avoid to touch another object just give negative rewards if that is touched.\n"+
              "-If you want to grasp an object and/or not to drop it you should reward contacts witht the gripper fingers and the object\n" +
              "-To reward lifting of objects, you can assume that the 3rd dimension of position vectors is the vertical "
              "z axis, which is oriented towards the ceiling. You can use +height as reward shaping for lifting tasks.\n"+
              "-Use the negative distance to get close to objects.\n"+
              "-Penalties for unwanted actions should be very small in absolute value, compared to positive rewards.\n" +
              "-Give a positive reward when the end effector is touching an object to reward grasping of said object.\n" +
              "\nHere is some examples of reward functions for some tasks:\n" +
              open("examples.txt", "r").read()+
              "\n\n\n\n")
    print(prompt)

    return prompt + base_environment_code


def create_feedback_prompt(stats_feedback):
    return ("These are the statistics of the reward components and success rates evaluated during training:\n" + str(
        stats_feedback) + '\n\n' +
            "Please carefully analyze the policy feedback and provide a new, improved reward function "
            "that can better solve the task. Some helpful tips for analyzing the policy feedback: \n"
            "(1) If the success rates are always near zero, then you must reconsider the entire reward function \n\n"
            "(2) If the values for a certain reward component are near identical throughout, then this means \n\n"
            "RL is not able to optimize this component as it is written. You may consider \n"
            "(a) Changing its scale or the value of its temperature parameter \n"
            "(b) Re-writing the reward component \n"
            "(c) Discarding the reward component \n\n"
            "(3) If some reward components’ magnitude is significantly larger, then you must re-scale "
            "its value to a proper range Please analyze each existing reward component in the suggested manner "
            "above first, and then write the reward function code.")


def get_base_environment_code(env_name):
    with open('environments.py', 'r') as file:
        base_environments_code = file.read()

    start = base_environments_code.find(f'# {env_name}')
    end = base_environments_code.find(f'# {env_name} end')

    base_environments_code = base_environments_code[start:end]

    assert base_environments_code is not None
    return base_environments_code
