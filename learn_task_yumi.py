import copy
import os
import time
from ast import parse

import main
import openai_utils
import torch.multiprocessing as mp
import numpy as np
import torch
import yaml

import utils
import main_vectorized_agent
import pickle as pkl
from environments_experiments import dicts
"""
    Yumi in single arm mode (left arm).
    Examples:   

    -The robot gripper is close to the blue cube, touch it with the gripper fingers and push it close to the red cube. Consider the task solved if the distance between the cubes is less than 0.04 meters. Consider the task failed if the distance of the end effector from the blue cube is more than 0.1m.
    -The robot gripper is close to a blue cube and a red cube. Grasp the blue cube with both fingers, then move the blue cube upwards of 0.05 meters. Consider the task completed when the blue cube is lifted by 0.05m over its loading height. Consider the task failed if the distance of the end effector from the blue cube is more than 0.1m or the red cube is moved from its loading position of 0.005m or more.

    -The robot gripper is holding a blue cube, and a red cube is placed on a surface very close to the gripper. Do not drop the clue cube, keep the gripper fingers in contact with the blue cube. Place the blue cube on top of the red cube. Consider the task completed when the distance between the two cubes in the x-y plane is less than 0.005m, the absolute difference between the two cubes height is less or equal 0.025m, the red cube is within 0.005m from its loading position. Consider the task failed if the gripper looses contact with the blue cube or the red cube is moved from its loading position of 0.005m or more, or the two cubes are further than at loading time.
    -The robot gripper is holding a blue cube, which is stacked on top of a red cube. Let go of the blue cube, move the tower of cubes to the right (negative direction of the y axis) of 0.1 meters, by touching the red cube. Avoid touching the blue cube. Consider the task solved if both cubes are moved to the right of 0.1 meters from their loading position. Consider the task failed if the x-y distance between the cubes is > 0.01 meters, or the distance between end effector and the red cube is > 0.05 meters.

    -The robot is holding a vial with its gripper. Lower the vial and insert it into the vial carrier, which is just below the vial. Consider the task solved when the vial is still touched by both gripper fingers, and the vial is at 0.06m height or below. Consider the task failed if the fingers loose contact with the vial, or the vial's distance in the x-y plane from its loading position is more than 0.025m.
    -The robot has just inserted a vial in a vial carrier, the gripper is very close to the vial. Grasp the vial, and lift out of the vial carrier, avoid to touch the vial carrier itself. Consider the task completed when the vial is lifted of 0.1m above loading height. Consider the task failed if the fingers loose contact with the vial, or the vial's distance in the x-y plane from its loading position is more than 0.025m.
    
    -The robot is holding a vial with its gripper. Lower the vial and insert it into the lab centrifuge, which is just below the vial. Consider the task solved when the vial is still touched by both gripper fingers, and the vial is at 0.08m height or below. Consider the task failed if the fingers loose contact with the vial, or the vial's distance in the x-y plane from its loading position is more than 0.04m.
    
    
    Yumi in bi manual mode.
    Examples:   
    
        -[Failed] The robot is holding a vial with its left gripper. Keep the vial grasped by the left gripper, do not loose contact between the left gripper fingers and the vial. Move the right gripper close to the vial, grasp the vial with the right gripper, and only then release the vial with the left gripper. Consider the task solved when the vial is grasped by both gripper fingers of the right gripper, and the left gripper's fingers are not touching it. Consider the task failed if no finger is in contact with the vial, or the distance between the end effectors is more than 0.3 meters.
        -[Rewritten] The robot is holding a vial with its left gripper. Keep the vial grasped by the left gripper, do not loose contact between the left gripper fingers and the vial. Move the right gripper close to the vial, grasp the vial with the right gripper, and only then release the vial with the left gripper. Consider the task solved when the vial is grasped by both gripper fingers of the right gripper, and the left gripper's fingers have not been touching it for 50 time steps. Consider the task failed if no finger is in contact with the vial, or the distance between the end effectors is more than 0.3 meters.
        
        -The robot is holding a box with its left gripper. Keep the box grasped by the left gripper, do not loose contact between the left gripper fingers and the box. Move the right gripper close to the box, grasp the box with the right gripper, and only then release the box with the left gripper. Consider the task solved when the box is grasped by both gripper fingers of the right gripper, and the left gripper's fingers have not been touching it for 50 time steps. Consider the task failed if no finger is in contact with the box, or the distance between the end effectors is more than 0.4 meters.
        
        -The robot is holding a cube with its left gripper, and a cup with its right gripper. Keep the cup grasped by the right gripper, do not loose contact between the right gripper fingers and the cup. Place the cube into the cup. Consider the task solved when the cup is grasped by the gripper fingers of the right gripper, the cube is at a distance from the cup of 0.025 or less. Consider the task failed if no finger of the right gripper is in contact with the cup, or the distance between the left end effector and the cube is more than 0.2 meters.
        
        -The robot is holding a pen with its left gripper, and a tape with its right gripper. The tape has a hole in the middle. Keep the tape grasped by the right gripper, do not loose contact between the right gripper fingers and the cup. The same for the pen and the left gripper. Insert the pen into the hole of the tape. Consider the task solved when both objects are grasped, and the pen is at a distance from the tape of 0.005 or less. Consider the task failed if no finger of the right gripper is in contact with the tape, or no finger of the left gripper is in contact with the pen, or the distance between the effectors is more than 0.4 meters.

        -The robot has its left and right end effectors over a spaghetti box. Reach the spaghetti box with both grippers and grasp it using all fingers of both grippers. Consider the task partially solved when all four fingers are touching the box. Consider the task solved if the spaghetti box is lifted of 0.1m over its loading height, while all fingers are in contact with it. Consider the task failed when either end effector is further from the spaghetti box than 0.2 meters.
"""

log_dir = 'logs_vec_agent/'

def run_trainers(args, env_args=None):
    num_seeds_per_groups = int((args.seeds / args.num_cuda_devices) / args.num_independent_runs)

    if env_args is None:
        env_args = {'render_mode': args.render_mode}
    env = utils.get_env(args.env, env_args)

    processes = []

    for num_device in range(args.num_cuda_devices):
        for run in range(args.num_independent_runs):
            kwargs = {
                "state_dim": env.observation_dim,
                "action_dim": env.action_dim,
                "max_action": env.max_action,
                "discount": args.discount,
                "tau": args.tau,
                "ensemble_size": args.ensemble_size,
                "UTD": args.UTD,
                "depth": args.depth,
                "num_neurons": args.num_neurons,
                "device": torch.device("cuda:"+str(num_device)),
            }
            log_folder_name = log_dir + args.env + '/' + args.policy if args.name is None else args.name
            if not os.path.exists(log_folder_name):
                os.makedirs(log_folder_name)

            policy = utils.get_class_from_module(args.policy, "Algorithm")(**kwargs)

            if args.load_model != "":
                if args.load_model == "default":
                    policy_file = f"{args.env}_{args.seed}"
                    policy.load(log_folder_name + f"/models/{policy_file}", load_critic=False)
                else:
                    policy.load(args.load_model, load_critic=False)

            replay_buffer = utils.ReplayBuffer(env.observation_dim, env.action_dim,
                                            max_size=int(num_seeds_per_groups * 1e6),
                                            device=kwargs['device'])

            a = num_device * num_seeds_per_groups * args.num_independent_runs + run * num_seeds_per_groups
            b = num_device * num_seeds_per_groups * args.num_independent_runs + (run+1) * num_seeds_per_groups

            for seed in range(a, b):
                yaml.dump(kwargs, open(log_folder_name + f'/params_{seed}.yaml', 'w'))
                p = mp.Process(target=main_vectorized_agent.worker,
                            args=(policy, replay_buffer, copy.deepcopy(env_args), args, seed, log_folder_name))
                p.start()
                processes.append(p)

    del env # should close the env once it runs out of memory I guess.

    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = utils.get_argparse()
    parser.add_argument("--env_config", default="vial_grasp.yaml")
    parser.add_argument("--yumi_mode", default="left") # options: [left, bi_manual]
    parser.add_argument("--seeds", default=4, type=int)  # Number of seeds to train in parallel
    parser.add_argument("--max_attempts", default=1, type=int)
    parser.add_argument("--num_cuda_devices", default=1, type=int)
    parser.add_argument("--num_independent_runs", default=3, type=int)
    parser.add_argument("--restart", action="store_true") # if true uses the previously generated reward
    parser.add_argument("--no_termination_training", action="store_true")
    args = parser.parse_args()

    if args.env in dicts:
        args.env_config = dicts[args.env]
    env_config = yaml.load(open("env_configs/" + args.env_config, "r"), Loader=yaml.FullLoader)

    if args.yumi_mode == "left":
        env_name = "YumiLeftEnv_objects_in_scene"
    elif args.yumi_mode == "bi_manual":
        env_name = "YumiEnv_objects_in_scene"
    else:
        raise NotImplementedError(args.yumi_mode)
    base_environment_code = openai_utils.get_base_environment_code(env_name)

    env_args = {"render_mode": "human" if args.render else None, "env_config_dict": env_config,
                "control_mode": 'cartesian'}
    mp.set_start_method('spawn')

    new_class_id = 0

    if args.env == "":
        task_definition = "The robot is in front the following "
        task_definition += open("env_configs/" + args.env_config, "r").read()
        task_definition += "\n" + input('\n\n' + task_definition + '\nTask definition: ')
        prompt = openai_utils.create_prompt(task_definition=task_definition,
                                            base_environment_code=base_environment_code,
                                            new_class_id=new_class_id)

        args.env = f"GeneratedEnv_{new_class_id}"
        if not args.restart:
            client = openai_utils.get_openai_client()
            open('generated_environments.py', 'w').close() # Creates new file or if already exists removes contents

            response = openai_utils.class_code_prompt(client, prompt)
            response = '\nfrom environments import * \n' + response + '\n\n'
            with open('generated_environments.py', 'w') as code_outfile:
                code_outfile.write(response)
                code_outfile.close()
    run_trainers(args=args, env_args=env_args)
    log_folder_name = log_dir + args.env + '/' + args.policy if args.name is None else args.name

    attempt = 1
    while attempt < args.max_attempts:
        best_eval = -np.inf
        best_success_rate = -1
        best_seed = -1
        best_seed_evaluations = None
        best_seed_success_rates = None
        best_seed_eval_reward_components = None
        best_seed_terminated_early_count = None
        best_seed_terminated_max_time_count = None

        for seed in range(args.seeds):
            log_name = f"{args.policy}_{seed}"
            policy_succ_file = log_folder_name + f"/models/{log_name}_best_model_success_difficulty_0"
            if os.path.exists(policy_succ_file):
                print("The task has been solved, find policy in: " + policy_succ_file)
                exit(0)

            evaluations = np.load(log_folder_name+f"/{args.policy}_{seed}.npy")
            success_rates = np.load(log_folder_name+f"/{args.policy}_{seed}_rates.npy")
            evaluations_reward_components = pkl.load(open(log_folder_name + f"/{log_name}_reward_components.pkl", 'rb'))
            args.seed = seed
            args.test_model = True
            args_ = copy.deepcopy(args)
            args_.name = log_folder_name
            evaluation, _, success_rate, terminated_early, terminated_max_time, final_state\
                = main.main(args_, env_args=env_args)
            if success_rate >= best_success_rate and evaluation > best_eval:
                # its for sure < 100% (otherwise we would have found a successful model)
                best_eval = evaluation
                best_success_rate = success_rate
                best_seed = seed
                best_seed_evaluations = evaluations
                best_seed_success_rates = success_rates
                best_seed_eval_reward_components = evaluations_reward_components
                best_seed_terminated_early_count = terminated_early
                best_seed_terminated_max_time_count = terminated_max_time


        stats_feedback = {}
        for k in best_seed_eval_reward_components.keys():
            reward_progression = best_seed_eval_reward_components[k]
            stats_feedback[k] = [{'progression': [reward_progression[i] for i in sorted(np.random.randint(0, len(reward_progression), 10))] + [reward_progression[-1]],
                                   'min': np.min(reward_progression),
                                   'max': np.max(reward_progression),
                                   'mean': np.mean(reward_progression),
                                   'median': np.median(reward_progression)}]
        stats_feedback['total_reward'] = {'progression': [best_seed_evaluations[i] for i in sorted(np.random.randint(0, len(best_seed_evaluations), 10))] + [best_seed_evaluations[-1]],
                                         'min': np.min(best_seed_evaluations),
                                         'max': np.max(best_seed_evaluations),
                                         'mean': np.mean(best_seed_evaluations),
                                         'median': np.median(best_seed_evaluations)}

        stats_feedback['success_rates'] = {'progression': [best_seed_success_rates[i] for i in sorted(np.random.randint(0, len(best_seed_success_rates), 10))] + [best_seed_success_rates[-1]],
                                         'min': np.min(best_seed_success_rates),
                                         'max': np.max(best_seed_success_rates),
                                         'mean': np.mean(best_seed_success_rates),
                                         'median': np.median(best_seed_success_rates)}


        new_class_id += 1
        attempt += 1

        with open('generated_environments.py', 'r') as file:
            generated_environments_code = file.read()
        assert generated_environments_code is not None

        feedback = ''
        if best_seed_terminated_early_count > 0:
            feedback += f'{best_seed_terminated_early_count * 100:10.2f}% of the evaluations failed some constraints before the maximum time allowed, and did not solve the task.\n'
        if best_seed_terminated_max_time_count > 0:
            feedback += f'{best_seed_terminated_max_time_count * 100:10.2f}% of the evaluations terminated reaching the maximum time allowed, and did not solve the task.\n'
        if len(stats_feedback.keys()) > 1:
            feedback += openai_utils.create_feedback_prompt(stats_feedback=stats_feedback)

        # Retry with another reward
        prompt = (
                f"Your previous attempt at generating a reward yielded a policy with a success rate of {best_success_rate * 100:10.2f}%\n"
                + feedback +
                "Try again :\n" + openai_utils.create_prompt(task_definition=task_definition,
                                                             base_environment_code=base_environment_code,
                                                             new_class_id=new_class_id))
        prompt += "\n\nThis is the code you generated previously:\n\n\n" + generated_environments_code

        print('----------------------------------------\n\n')
        print(prompt)
        print('----------------------------------------\n\n')

        response = openai_utils.class_code_prompt(client, prompt) + '\n\n'
        with open('generated_environments.py', 'w') as code_outfile:
            code_outfile.write(generated_environments_code + '\n\n\n' + response)
            code_outfile.close()
        # Train
        args.env = f"GeneratedEnv_{new_class_id}"
        env_args = {"render_mode": "human" if args.render else None, "env_config_dict": env_config,
                    "control_mode": 'cartesian'}
        run_trainers(args=args, env_args=env_args)

