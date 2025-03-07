import numpy as np
import os
import tqdm

import time
import utils, sim_utils
import yaml
import pickle as pkl
from environments_experiments import prompts, dicts
parser = utils.get_argparse()
parser.add_argument("--mode", type=str, default="left") # left / bi_manual

args = parser.parse_args()

policy = utils.get_class_from_module('code_policies', args.policy)()

env_config = dicts[args.env]
env_config = yaml.load(open("env_configs/" + env_config, "r"), Loader=yaml.FullLoader)
env_args = {"render_mode": "human" if args.render else None, "env_config_dict": env_config,
            "control_mode": 'cartesian'}
env = utils.get_env(args.env, env_args)

if args.render and os.path.exists('pybullet_camera_placement.pkl'):
        placement_tuple = pkl.load(open('pybullet_camera_placement.pkl', 'rb'))
        sim_utils.set_current_camera_placement(*placement_tuple)

repetitions = 10
max_steps = 1000
success_rate = 0

logfile = "logs_code_policies/{}.np".format(args.env)
if os.path.exists(logfile):
    log = np.loadtxt(logfile).tolist()
    if type(log) is float:
        log = [log]
else:
    log = []

for i in tqdm.tqdm(range(repetitions)):
    state = env.reset()
    finished = False
    step = 0
    while (not finished and step < max_steps):
        # time.sleep(0.01)
        step += 1

        num_steps = 0
        action = policy.select_action(np.array(state))
        if args.mode == "left":
            action = action[:7]

        state, _, _, _, _ = env.step(action)
        num_steps += 1

        # if env.check_collisions():
        #     print("Detected collision!")

        if env._get_info()['task_solved']:
            # print("policy executed!")
            finished = True
            success_rate += 1

        if env.termination_condition():
            # print("Termination condition!")
            finished = True

print(f"{args.policy} - success rate:", success_rate/repetitions)

log.append(success_rate/repetitions)
np.savetxt(logfile, np.array(log))
