import numpy as np
import yaml

import openai_utils
import utils
from environments_experiments import prompts, dicts


# CodePolicy
class CodePolicy:
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] # <0 is for closing, > 0 for opening, must be in [-1, 1]
        right_end_effector_pose = state[14:20]
        right_end_effector_vel = state[20:26]
        right_gripper = state[26]
        right_gripper_vel = state[27]  # <0 is for closing, > 0 for opening, must be in [-1, 1]

        objects_state = state[27:] # If empty then there are no objects in sim

        # In this way the velocities are kept constant
        action = left_end_effector_vel + left_gripper_vel + right_end_effector_vel + right_gripper_vel

        # Instead, this keeps the robot still
        action = [0.0] * 14

        return np.array(action)
# CodePolicy end

# CodePolicyLeft
class CodePolicyLeft:
    def __init__(self):
        super().__init__()

    def select_action(self, state):
        left_end_effector_pose = state[:6]
        left_end_effector_vel = state[6:12]
        left_gripper = state[12]
        left_gripper_vel = state[13] # <0 is for closing, > 0 for opening, must be in [-1, 1]
        objects_state = state[14:] # If empty then there are no objects in sim

        # In this way the velocities are kept constant
        action = left_end_effector_vel + left_gripper_vel

        # Instead, this keeps the robot still
        action = [0.0] * 7

        return np.array(action)
# CodePolicyLeft end


def get_base_policy_code(name):
    with open('code_as_policy.py', 'r') as file:
        base_policy_code = file.read()

    start = base_policy_code.find('# {}'.format(name))
    end = base_policy_code.find('# {} end'.format(name))

    base_policy_code = base_policy_code[start:end]

    assert base_policy_code is not None
    return base_policy_code

def get_prompt(task, base_policy_name):
    prompt = ("Write a Python class that implements a manipulation policy for a bimanual robot.\n" +
              "The Python class must extend this class:\n" + get_base_policy_code(base_policy_name) + "\n\n" +
              "The policy must solve the following task:\n\n"+task+"\n\n"+
              "Position units are in meters, orientations are in euler angles (radiants).")
    return prompt

if __name__ == '__main__':
    parser = utils.get_argparse()
    parser.add_argument("--mode", type=str, default="left")  # options: [left, bi_manual]
    parser.add_argument("--num", type=int, default=10)
    args = parser.parse_args()


    # print(open("env_configs/" + args.env_config, "r").read())
    # exit()

    task_definition = prompts[args.env] + "\n\n"
    env_config = dicts[args.env]
    env_config = yaml.load(open("env_configs/" + env_config, "r"), Loader=yaml.FullLoader)
    task_definition += "The robot is in front the following objects\n"
    for k in env_config["objects"]:
        task_definition += k + ", position: " + str(env_config["objects"][k]["load_pos"]) + ", orientation: " + str(env_config["objects"][k]["load_quat"]) + "\n"

    task_definition += "Geometry of objects:\n\n"
    for k in env_config["objects"]:
        task_definition += k + ": \n\n"
        task_definition += open(env_config["objects"][k]["urdf_path"], "r").read()
        task_definition += "#########################################################################\n\n"
    task_definition += "#########################################################################"


    if args.mode == "left":
        base_policy_name = "CodePolicyLeft"
        task_definition += "\n\n Move only the left arm. keep right_end_effector_vel and right_gripper_vel to zero"
    else:
        base_policy_name = "CodePolicy"

    for n in range(args.num):
        prompt = get_prompt(task_definition, base_policy_name=base_policy_name)
        prompt += f"\n\n Call the policy class {args.env}_{n}"
        print(prompt)

        client = openai_utils.get_openai_client() # close connection to delete memory
        code_policy = openai_utils.execute_code_prompt(client, prompt)

        with open('code_policies.py', 'a') as code_outfile:
            code_outfile.write("\n\n")
            code_outfile.write(code_policy)
            code_outfile.close()












