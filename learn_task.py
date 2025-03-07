import copy
import yaml
import main
import openai_utils
import utils

"""
    Examples:
    
    Move the mobile robot towards the end of the corridor, while avoiding to hit walls and obstacles.
    
"""

tasks = [
    "The agent mush reach the cube position. Then grasp and lift the cube. Consider the task solved when the cube is at 0.5 height. There is no failure condition.",
    "The agent mush reach the cube position. Then grasp the cube. Finally, move the cube to the right (positive x direction). Consider the task solved when the cube x coordinate is > 0.99. There is no failure condition.",
    "The agent is holding the cube. Move the cube to (0, 0). Consider the task solved when the cube is at 0.05 distance from (0, 0) and is grasped. Consider the task failed when the agent doesn't grasp the cube.",
    "There are 2 agents. Both Agents mush reach the cube position. Then each must grasp and lift the cube. Consider the task solved when the cube is at 0.5 height. There is no failure condition.",
    "There are 2 agents. Agent 1 has to reach the cube, grasp it and hand it over to agent 0. Then agent 0 must bring the cube to (-0.5, 0.0). Consider the task solved when the cube is at a distance lower or equal than 0.1 meters from (-0.5, 0.0). There is no failure condition.",
]

parser = utils.get_argparse()
parser.add_argument("--env_config", default="")
parser.add_argument("--attempts", default=10, type=int)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--restart", action="store_true")
parser.add_argument("--shadow_render", action="store_true")
parser.add_argument("--no_termination_training", action="store_true")
parser.add_argument("--no_terminal_reward", action="store_true")

args = parser.parse_args()


if args.env_config == "":
    env_args = {"render_mode": "human" if args.render else None}
else:
    env_config = yaml.load(open("env_configs/" + args.env_config, "r"), Loader=yaml.FullLoader)
    env_args = {"render_mode": "human" if args.render else None, "env_config_dict": env_config}


print('\n\nDefine a task for one of the available base environments:')

test_environments = ["2DGraspEnv", "2DSlideEnv", "2DPlaceEnv", "2DBiGraspEnv", "2DBiHandoverEnv"]
base_environments = ["SimplifiedGraspingEnv", "SimplifiedGraspingEnv", "SimplifiedPlacingEnv", "SimplifiedBiGraspingEnv", "SimplifiedBiHandoverEnv"]
max_episodes = [301, 301, 101, 501, 1]

id = test_environments.index(args.name)
env_name = base_environments[id]

base_environment_code = openai_utils.get_base_environment_code(env_name)

task_definition = tasks[id]
args.max_episodes = max_episodes[id]

seed = 0
attempts = 10

trainining_name = copy.deepcopy(test_environments[id])

for attempt in range(args.start, args.attempts):
    seed = attempt

    # These are the only options considered
    if args.no_termination_training and args.no_terminal_reward:
        train_env_name = trainining_name + "_NoTermTrain_" + str(attempt)
    if args.no_terminal_reward:
        train_env_name = trainining_name + "_NoTermReward_" + str(attempt)
    else:
        train_env_name = trainining_name + "_" + str(attempt)

    if not (args.restart and attempt == args.start):
        client = openai_utils.get_openai_client()

        prompt = openai_utils.create_prompt(task_definition=task_definition,
                                            base_environment_code=base_environment_code,
                                            new_class_id=train_env_name,
                                            terminal_reward=not args.no_terminal_reward)

        response = openai_utils.class_code_prompt(client, prompt)

        response = '\nfrom environments import * \n' + response + '\n\n'

        with open('generated_environments.py', 'a') as code_outfile:
            code_outfile.write(response)


    args.policy = 'SAC'
    if args.no_terminal_reward:
        args.name = "logs/" + trainining_name + "/without_terminal_reward/" + train_env_name + "/" + args.policy + "/"
    else:
        args.name = "logs/" + trainining_name + "/with_terminal_reward/" + train_env_name + "/" + args.policy + "/"
    args.env = "GeneratedEnv_" + train_env_name
    args.seed = seed
    args.expl_noise = 0.0
    args.eval_episodes = 10
    args.eval_freq = 5000
    args.save_model = True
    args.max_episode_steps = -1
    args.depth = 3

    main.main(args, env_args)

