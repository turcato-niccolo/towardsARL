import numpy as np
import torch
from setproctitle import setproctitle
import utils
import yaml
import pickle as pkl


def main(args, env_args=None):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    file_name = f"{args.env}_{args.seed}"
    log_folder_name = 'logs/' + args.env + '/' + args.policy if args.name is None else args.name
    log_name = f"{args.policy}_{args.seed}"
    log_args = {"file_name": file_name, "log_folder_name": log_folder_name, "log_name": log_name}

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    render_mode = None
    if args.render:
        render_mode = 'human'
    if args.shadow_render:
        render_mode = 'human-shadows'
    if env_args is None:
        env_args = {'render_mode': render_mode}
    env = utils.get_env(args.env, env_args)

    if args.test_model:
        if args.load_initial_state != "":
            initial_state_file = log_folder_name + f"/{log_name}_final_state.npy" \
                if args.load_initial_state == "default" else args.load_initial_state
            initial_state = np.load(initial_state_file)
            env.load_initial_state(list(initial_state))

        # Load parameters used during training
        with open(log_folder_name+f'/params_{args.seed}.yaml', 'r') as stream:
            try:
                kwargs = yaml.load(stream, Loader=yaml.Loader)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        # Create policy object and load saved model
        policy = utils.get_class_from_module(args.policy, "Algorithm")(**kwargs)
        setproctitle(f"RL Testing ||{args.env}||{args.policy}||{args.seed}")

        print('Evaluate successful policies')

        difficulty = 0
        while True:
            policy_file = log_folder_name + f"/models/{log_name}_best_model_success_difficulty_{difficulty}"
            try:
                policy.load(policy_file, load_critic=True)
            except Exception as e:
                print(e)
                break
            # if not os.path.exists(policy_file+'/'):
            #     os.makedirs(policy_file+'/')
            # policy.actor.save_to_csv_files(policy_file+'/')
            evaluation = utils.eval_policy(env, policy, args.seed, 0, args, return_feedback=True)

            increase_task_difficulty_fun = getattr(env, "increase_task_difficulty", None)
            if callable(increase_task_difficulty_fun):
                env.increase_task_difficulty()
                difficulty += 1
            else:
                break

        # Evaluate best policy according to reward function
        policy_file = log_folder_name + f"/models/{log_name}_best_model"
        policy.load(policy_file, load_critic=True)
        # if not os.path.exists(policy_file + '/'):
        #     os.makedirs(policy_file + '/')
        # policy.actor.save_to_csv_files(policy_file + '/')
        # Evaluate
        evaluation, eval_reward_components , solved_tasks,  terminated_early, terminated_max_time = (
            utils.eval_policy(env, policy, args.seed, 0, args, return_feedback=True))
        np.save(log_folder_name + f"/{log_name}_evaluation.npy", evaluation)
        pkl.dump(eval_reward_components, open(log_folder_name + f"/{log_name}_info_dict.pkl", "wb"))
        # Save the final state
        final_state = env.save_current_state()
        np.savetxt(log_folder_name + f"/{log_name}_final_state.txt", np.array(final_state))

        return evaluation, eval_reward_components, solved_tasks, terminated_early, terminated_max_time, final_state
    else:  # Train
        if args.no_termination_training:
            eval_env = utils.get_env(args.env, env_args)
            env._get_info = lambda : {'task_solved': False}
            env.termination_condition = lambda : False
        else:
            eval_env = env

        replay_buffer = utils.ReplayBuffer(env.observation_dim, env.action_dim, max_size=int(1e6))
        utils.train_loop(env=env, eval_env=eval_env, replay_buffer=replay_buffer, args=args, log_args=log_args)


if __name__ == "__main__":
    parser = utils.get_argparse()
    parser.add_argument("--env_config", default="")
    parser.add_argument("--shadow_render", action="store_true")
    parser.add_argument("--no_termination_training", action="store_true")
    args = parser.parse_args()
    if args.env_config == "":
        env_args = None
    else:
        env_config = yaml.load(open("env_configs/"+args.env_config, "r"), Loader=yaml.FullLoader)
        env_args = {"render_mode": "human" if args.render else None, "env_config_dict": env_config}

    main(args=args, env_args=env_args)