# Import the necessary libraries
import copy

import torch.multiprocessing as mp
import numpy as np
import torch
import yaml

import utils
import os

def worker(policy, replay_buffer, env_args, args, seed, log_folder_name):
    args.seed = seed

    render_mode = None
    if args.render:
        render_mode = 'human'
    env = utils.get_env(args.env, env_args)

    if args.no_termination_training:
        eval_env = utils.get_env(args.env, env_args)
        env._get_info = lambda: {'task_solved': False}
        # env.termination_condition = lambda: False
    else:
        eval_env = env

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    file_name = f"{args.env}_{args.seed}"
    log_name = f"{args.policy}_{args.seed}"
    log_args = {"file_name": file_name, "log_folder_name": log_folder_name, "log_name": log_name}

    utils.train_loop(env=env, eval_env=eval_env, replay_buffer=replay_buffer, args=args, log_args=log_args,
                     policy=policy, device=policy.device)


def main(args, env_args=None):
    mp.set_start_method('spawn')

    if env_args is None:
        env_args = {'render_mode': args.render_mode}
    env = utils.get_env(args.env, env_args)

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
        "device": torch.device(args.device),
    }
    log_folder_name = 'logs_vec_agent/' + args.env + '/' + args.policy if args.name is None else args.name
    if not os.path.exists(log_folder_name):
        os.makedirs(log_folder_name)

    policy = utils.get_class_from_module(args.policy, "Algorithm")(**kwargs)

    if args.load_model != "":
        if args.load_model == "default":
            policy_file = f"{args.env}_{args.seed}"
            policy.load(log_folder_name + f"/models/{policy_file}", load_critic=False)
        else:
            policy.load(args.load_model, load_critic=False)

    replay_buffer = utils.ReplayBuffer(env.observation_dim, env.action_dim, max_size=int(args.seeds * 1e6))
    del env
    processes = []

    for seed in range(args.start_seed, args.start_seed + args.seeds):
        yaml.dump(kwargs, open(log_folder_name + f'/params_{seed}.yaml', 'w'))
        p = mp.Process(target=worker, args=(policy, replay_buffer, copy.deepcopy(env_args), args, seed, log_folder_name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = utils.get_argparse()
    parser.add_argument("--seeds", default=3, type=int)  # Number of seeds to train in parallel
    parser.add_argument("--start_seed", default=0, type=int)  # Number of seeds to train in parallel
    parser.add_argument("--device", default='cuda:0', type=str)  # Number of seeds to train in parallel

    args = parser.parse_args()

    main(args=args)