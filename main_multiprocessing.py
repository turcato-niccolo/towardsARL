# Import the necessary libraries
import torch.multiprocessing as mp
import numpy as np
import torch
import utils

def worker(shared_replay_buffer, args, seed):
    num_seeds_per_groups = int(args.seeds / args.num_cuda_devices)
    num_cuda_dev = int(seed // num_seeds_per_groups)
    args.seed = seed

    device = torch.device('cuda:'+str(num_cuda_dev))
    render_mode = None
    if args.render:
        render_mode = 'human'
    env = utils.get_env(args.env, {'render_mode': render_mode})

    replay_buffer = utils.HybridReplayBuffer(env.observation_dim, env.action_dim,
                                             shared_replay_buffer=shared_replay_buffer, max_size=int(2e6),
                                             device=device)

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    file_name = f"{args.env}_{args.seed}"
    log_folder_name = 'logs_mp/' + args.env + '/' + args.policy if args.name is None else args.name
    log_name = f"{args.policy}_{args.seed}"
    log_args = {"file_name": file_name, "log_folder_name": log_folder_name, "log_name": log_name}

    # Callback that updates the shared memory and local memory
    def cb_synch_shared_buffer():
        replay_buffer.synch_shared_memory(shared_batch_size=args.batch_size)

    utils.train_loop(env=env, replay_buffer=replay_buffer, args=args, log_args=log_args, device=device,
                     cb_episode_end=cb_synch_shared_buffer)


if __name__ == '__main__':
    parser = utils.get_argparse()
    parser.add_argument("--seeds", default=3, type=int)  # Number of seeds to train in parallel
    parser.add_argument("--num_cuda_devices", default=3, type=int)
    # Number of cuda devices available to train in parallel
    # We assume that the devices are numbered as [0, ..., num_cuda_devices-1]

    args = parser.parse_args()

    mp.set_start_method('spawn')

    shared_replay_buffer = utils.LockingReplayBuffer(capacity=int(1e6), device=torch.device('cpu'))

    processes = []

    for seed in range(args.seeds):
        p = mp.Process(target=worker, args=(shared_replay_buffer, args, seed))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
