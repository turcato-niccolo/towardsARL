import argparse
import copy
import importlib
import random
import yaml
from setproctitle import setproctitle
import numpy as np
import torch
import os
import pickle
import redis
import tqdm
from torch.multiprocessing import Lock
import bisect
from pathlib import Path
import pickle as pkl

min_Val = torch.tensor(1e-7).float()


def get_class_from_module(module_name, class_name):
    module = importlib.import_module(module_name)
    importlib.reload(module)
    return getattr(module, class_name)


def load_curves(path, algorithm_name, seeds):
    curves = []
    for i in seeds:
        name = f"{path}/{algorithm_name}_{i}.npy"
        if os.path.isfile(name):
            # print(f"{name} loaded", np.load(name).shape)
            curves.append(list(np.load(name)))
    max_size = max([len(curve) for curve in curves])
    # for i in range(len(curves)):
    #     if len(curves[i]) > max_size:
    #         max_size = len(curves[i])
    for i in range(len(curves)):
        while len(curves[i]) < max_size:
            curves[i].append(curves[i][-1])
        assert len(curves[i]) == max_size

    curves = np.array(curves)
    return curves


def smooth(data, length=10):
    data = data.copy()
    pad = data[:, -1, None].repeat(length - 1, axis=-1)
    pad_data = np.concatenate((data, pad), axis=-1)
    pad = data[:, 0, None].repeat(length - 1, axis=-1)
    pad_data = np.concatenate((pad, pad_data), axis=-1)
    for i in range(length, len(data[0])):
        assert i - length >= 0
        data[:, i] = np.mean(pad_data[:, i - length:i + length], axis=-1)
    return data


def eval_policy(env, policy, seed, step, args, return_feedback=False):
    avg_reward = 0.
    solved_tasks = 0
    terminated_early = 0
    terminated_max_time = 0

    avg_rewards = {}

    for k in tqdm.tqdm(range(args.eval_episodes)):
        state = env.reset(seed=seed + 100 * k)
        done = False
        num_steps = 0
        info = None
        while not done:
            # time.sleep(0.01)
            if 'SAC' in args.policy:
                action = policy.select_action(np.array(state), evaluate=True)
            else:
                action = policy.select_action(np.array(state))
            state, reward, done, rewards_dict, info = env.step(action)

            for k in rewards_dict.keys():
                if k in avg_rewards.keys():
                    avg_rewards[k] += rewards_dict[k]
                else:
                    avg_rewards[k] = rewards_dict[k]

            avg_reward += reward
            num_steps += 1
            if num_steps >= args.max_episode_steps:
                terminated_max_time += 1
            elif done and not info['task_solved']:
                terminated_early += 1

            done = done or num_steps >= args.max_episode_steps or info['task_solved']

        if info and info['task_solved']:
            solved_tasks += 1

    avg_reward /= args.eval_episodes
    for k in avg_rewards.keys():
        avg_rewards[k] /= args.eval_episodes

    print("---------------------------------------")
    print(f"{policy.device} - seed {seed} - {step} - Evaluation over {args.eval_episodes} episodes: {avg_reward:.3f} - Success rate: {solved_tasks / args.eval_episodes:.2f}")
    print("---------------------------------------")
    if return_feedback:
        return (avg_reward, avg_rewards, solved_tasks / args.eval_episodes, terminated_early / args.eval_episodes,
                terminated_max_time / args.eval_episodes)
    return avg_reward, avg_rewards, solved_tasks / args.eval_episodes


def train_loop(env, replay_buffer, args, log_args, eval_env=None, policy=None, device=None, cb_episode_end=None):
    setproctitle(f"RL Training ||{args.env}||{args.policy}||{args.seed}")
    if device is None:
        device = torch.device('cuda')
    if eval_env is None:
        eval_env = env
    file_name = log_args["file_name"]
    log_folder_name = log_args["log_folder_name"]
    log_name = log_args["log_name"]

    if args.load_initial_state != "":
        initial_state_file = log_folder_name + f"/{log_name}_final_state.npy" if args.load_initial_state == "default" else args.load_initial_state
        initial_state = np.load(initial_state_file)
        env.load_initial_state(list(initial_state))
        env.reset(seed=args.seed)

    if not os.path.exists(log_folder_name):
        os.makedirs(log_folder_name)
    if not os.path.exists(log_folder_name + "/models"):
        os.makedirs(log_folder_name + "/models")

    if args.max_episode_steps < 0:
        args.max_episode_steps = env._max_episode_steps

    if policy is None:
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
            "device": device,
        }
        policy = get_class_from_module(args.policy, "Algorithm")(**kwargs)
        yaml.dump(kwargs, open(log_folder_name + f'/params_{args.seed}.yaml', 'w'))

        if args.load_model != "":
            if args.load_model == "default":
                policy_file = file_name
                policy.load(log_folder_name + f"/models/{policy_file}", load_critic=True)
            else:
                policy.load(args.load_model, load_critic=True)

    eval_return, eval_reward_components, success_rate = eval_policy(eval_env, policy, args.seed, 0, args)
    evaluations = [eval_return]
    success_rates = [success_rate]
    evaluations_reward_components = {}
    for k in eval_reward_components.keys():
        evaluations_reward_components[k] = [eval_reward_components[k]]

    state = env.reset(seed=args.seed)
    # done, term = False, False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    best_eval_return = -np.inf
    best_success_return = -np.inf
    evaluate = False
    difficulty = 0

    for t in range(int(args.max_episodes * args.max_episode_steps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_steps:
            if args.initial_samples_from_policy:
                action = policy.select_action(np.array(state))
                # if replay_buffer.size >= args.batch_size:
                #     policy.optimize_critic(replay_buffer, args.batch_size)  # gather data and build Q estimate
            else:
                action = env.action_space.sample()

        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, policy.max_action * args.expl_noise, size=env.action_dim)
            ).clip(-policy.max_action, policy.max_action)

        # Perform action
        next_state, reward, done, _, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < args.max_episode_steps else 0

        # Store data in replay buffer
        with torch.no_grad():
            replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t > args.start_steps:
            policy.train(replay_buffer, args.batch_size)

        if done or episode_timesteps >= args.max_episode_steps:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"{policy.device} - seed {args.seed} - Total T: {t + 1}, Episode Num: {episode_num + 1} "
                  f"Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state = env.reset(seed=args.seed)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            if cb_episode_end:
                cb_episode_end()
            # policy.reset()

        if not evaluate and t % args.eval_freq == 0 and t > 0:
            evaluate = True

        # Evaluate episode
        if evaluate:
            Path(log_folder_name).mkdir(parents=True, exist_ok=True)
            # We need to do a copy here and evaluate the copy beacause if this loop is executed by multiple threads
            # sharing the same policy, another thread might update the parameters while the evaluation is running.
            eval_return, eval_reward_components, success_rate = (
                eval_policy(eval_env, policy, args.seed, (t + 1) // args.eval_freq, args))
            env.reset(seed=args.seed)
            evaluations.append(eval_return)
            success_rates.append(success_rate)
            for k in eval_reward_components.keys():
                evaluations_reward_components[k].append(eval_reward_components[k])
            np.save(log_folder_name + f"/{log_name}", evaluations)
            np.save(log_folder_name + f"/{log_name}_rates", success_rates)
            pkl.dump(evaluations_reward_components, open(log_folder_name + f"/{log_name}_reward_components.pkl", 'wb'))

            if args.save_model and eval_return >= best_eval_return:
                best_eval_return = eval_return
                print('New best policy found: save to', log_folder_name + '/models')
                policy.save(log_folder_name + f"/models/{log_name}_best_model")
                # pkl.dump(replay_buffer, open(log_folder_name + f"/models/{args.policy}_{args.seed}_best_model_replay_buffer.pkl", 'wb'))
            if success_rate == 1.0 and eval_return >= best_success_return:
                best_success_return = eval_return
                policy.save(log_folder_name + f"/models/{log_name}_best_model_success_difficulty_{difficulty}")

                increase_task_difficulty_fun = getattr(env, "increase_task_difficulty", None)
                if callable(increase_task_difficulty_fun):
                    env.increase_task_difficulty()
                    print('Increased task difficulty')
                    difficulty += 1
            evaluate = False

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="")  # OpenAI gym environment name
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--ensemble_size", default=None, type=int)  # Number of Q functions
    parser.add_argument("--UTD", default=None, type=int)  # Number of Q functions
    parser.add_argument("--eval_episodes", default=10, type=int)  # Number of Q functions
    parser.add_argument("--num_neurons", default=512, type=int)  # Number of neurons in networks layers
    parser.add_argument("--depth", default=3, type=int)  # Number of networks layers
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_steps", default=5000, type=int)  # episodes initial random policy is used
    parser.add_argument("--max_episode_steps", default=1e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_episodes", default=1e4, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.2, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--save_model", action="store_true", default=True)  # Save model and optimizer parameters
    parser.add_argument("--test_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--initial_samples_from_policy", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--name", default=None)  # Name for logging
    parser.add_argument("--load_initial_state", default="")
    # initial state load file name, "" doesn't load, "default" uses {log_name}_final_state
    # otherwise specify the path and the filename to load, for example to load the final state of a previous task
    return parser

class LockingReplayBuffer:
    def __init__(self, capacity, device, top_percentage=0.1):
        self.capacity = capacity
        self.buffer = []
        self.lock = Lock()
        self.device = device
        self.top_percentage = top_percentage

    def add(self, state, action, next_state, reward, done):
        sasrd = (state, action, next_state, reward, done)
        with self.lock:
            bisect.insort_left(self.buffer, sasrd, key=lambda r: -r[3])  # sort by reward (highest to lowest)
            self.buffer = self.buffer[:self.capacity]

    def add_batch(self, batch):
        with self.lock:
            for k in range(batch[0].shape[0]):
                sasrd = (batch[0][k, :].detach().cpu().numpy(), batch[1][k, :].detach().cpu().numpy(),
                         batch[2][k, :].detach().cpu().numpy(), batch[3][k].detach().cpu().numpy().item(),
                         batch[4][k].detach().cpu().numpy().item())
                bisect.insort_left(self.buffer, sasrd, key=lambda r: -r[3])  # sort by reward (highest to lowest)
            self.buffer = self.buffer[:self.capacity]

    def sample(self, batch_size):
        #with self.lock:
        # Random selection among the top percentage in the buffer

        # print(self.buffer[0][0].shape)
        # print(len(self.buffer))

        sampling_dim = int(len(self.buffer) * self.top_percentage)

        indices = np.random.choice(sampling_dim, min(batch_size, sampling_dim), replace=False)
        # states, actions, next_states, rewards, dones = [], [], [], [], []
        # for idx in indices:
        #     states.append(self.buffer[idx][0])
        #     actions.append(self.buffer[idx][1])
        #     next_states.append(self.buffer[idx][2])
        #     rewards.append(self.buffer[idx][3])
        #     dones.append(self.buffer[idx][4])
        # # print(len(states), states[0].shape)
        # return (torch.stack(states, dim=0),
        #         torch.stack(actions, dim=0),
        #         torch.stack(next_states, dim=0),
        #         torch.stack(rewards, dim=0),
        #         torch.stack(dones, dim=0))
        states, actions, next_states, rewards, dones = zip(*[self.buffer[idx] for idx in indices])
        # print(rewards)
        # print('-------------------')
        return (torch.tensor(states, dtype=torch.float32, device=self.device),
                torch.tensor(actions, dtype=torch.float32, device=self.device),
                torch.tensor(next_states, dtype=torch.float32, device=self.device),
                torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device),
                torch.tensor(np.array(dones), dtype=torch.float32, device=self.device))

    def __len__(self):
        with self.lock:
            return len(self.buffer)

    def make_copy(self):
        copy_obj = LockingReplayBuffer(capacity=self.capacity, device=self.device, top_percentage=self.top_percentage)
        copy_obj.buffer = copy.deepcopy(self.buffer)
        return copy_obj

class HybridReplayBuffer:
    """
        Class that implements an interface for a local replay buffer and a shared replay buffer

        The local buffer only contains experience added from this instance, while the shared buffer contains
        experience shared between multiple agents.

        The two buffers are the same dimension, sampling happens by sampling half of the batch from the local one
        and half from the shared one
    """
    def __init__(self, state_dim, action_dim, shared_replay_buffer, max_size=1e6, device=None):
        self.local_replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(max_size), device=device)
        self.shared_replay_buffer = shared_replay_buffer
        self.local_copy_of_shared_replay_buffer = shared_replay_buffer.make_copy()

    def add(self, state, action, next_state, reward, done):
        self.local_replay_buffer.add(state, action, next_state, reward, done)
        self.local_copy_of_shared_replay_buffer.add(state, action, next_state, reward, done)

    def sample(self, batch_size):
        local_states, local_actions, local_next_states, local_rewards, local_dones = (
            self.local_replay_buffer.sample(batch_size=int(batch_size / 2)))
        shared_states, shared_actions, shared_next_states, shared_rewards, shared_dones = (
            self.local_copy_of_shared_replay_buffer.sample(batch_size=int(batch_size / 2)))

        states = torch.cat([local_states, shared_states.to(local_states.device)], dim=0)
        actions = torch.cat([local_actions, shared_actions.to(local_states.device)], dim=0)
        next_states = torch.cat([local_next_states, shared_next_states.to(local_states.device)], dim=0)
        rewards = torch.cat([local_rewards[:, 0], shared_rewards[:].to(local_states.device)], dim=0).unsqueeze(-1)
        dones = torch.cat([local_dones[:, 0], shared_dones[:].to(local_states.device)], dim=0).unsqueeze(-1)


        return states, actions, next_states, rewards, dones

    def synch_shared_memory(self, shared_batch_size):
        # Add a batch to shared buffer
        local_batch = self.local_replay_buffer.sample(batch_size=shared_batch_size)
        self.shared_replay_buffer.add_batch(local_batch)
        # Copy the current shared replay buffer in local variable
        self.local_copy_of_shared_replay_buffer = self.shared_replay_buffer.make_copy()

    def size(self):
        return self.local_replay_buffer.size, self.shared_replay_buffer.size()


# class SharedReplayBuffer:
#     def __init__(self, db_name="replay_buffer.db", max_size=1e6, device=None):
#         initialize_database(db_name=db_name)
#         self.db_name = db_name
#         self.max_size = max_size
#
#         if device is None:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = device
#
#
#     def _execute(self, query, params=None):
#         conn = sqlite3.connect(self.db_name)
#         c = conn.cursor()
#         if params:
#             c.execute(query, params)
#         else:
#             c.execute(query)
#         conn.commit()
#         conn.close()
#
#     def add(self, state, action, next_state, reward, done):
#         state = pickle.dumps(state)
#         next_state = pickle.dumps(next_state)
#         action = pickle.dumps(action)
#         query = '''INSERT INTO replay_buffer (state, action, next_state, reward, done)
#                       VALUES (?, ?, ?, ?, ?)'''
#         self._execute(query, (state, action, next_state, reward, done))
#
#         # Maintain the FIFO size limit
#         self._maintain_size()
#
#     def _maintain_size(self):
#         conn = sqlite3.connect(self.db_name)
#         c = conn.cursor()
#         c.execute('''SELECT COUNT(*) FROM replay_buffer''')
#         count = c.fetchone()[0]
#         if count > self.max_size:
#             # Delete the oldest entries to maintain the buffer size
#             c.execute('''DELETE FROM replay_buffer
#                             WHERE id IN (SELECT id FROM replay_buffer
#                                          ORDER BY id ASC
#                                          LIMIT ?)''', (count - self.max_size,))
#         conn.commit()
#         conn.close()
#
#     def sample(self, batch_size):
#         conn = sqlite3.connect(self.db_name)
#         c = conn.cursor()
#         c.execute('''SELECT * FROM replay_buffer ORDER BY RANDOM() LIMIT ?''', (batch_size,))
#         rows = c.fetchall()
#         conn.close()
#
#         # batch = [(pickle.loads(row[1]), pickle.loads(row[2]), pickle.loads(row[3]), row[4], row[5]) for row in rows]
#
#         states = torch.FloatTensor([pickle.loads(row[1]) for row in rows]).to(self.device)
#         actions = torch.FloatTensor([pickle.loads(row[2]) for row in rows]).to(self.device)
#         next_states = torch.FloatTensor([pickle.loads(row[3]) for row in rows]).to(self.device)
#         rewards = torch.FloatTensor([row[4] for row in rows]).to(self.device)
#         dones = torch.FloatTensor([row[5] for row in rows]).to(self.device)
#
#         return states, actions, next_states, rewards, dones
#
#     def size(self):
#         conn = sqlite3.connect(self.db_name)
#         c = conn.cursor()
#         c.execute('''SELECT COUNT(*) FROM replay_buffer''')
#         count = c.fetchone()[0]
#         conn.close()
#         return count
class SharedReplayBuffer:
    def __init__(self, db_name="replay_buffer", max_size=1e6, device=None, host='localhost', port=6379):
        self.client = redis.StrictRedis(host=host, port=port, db=0)
        self.key = db_name
        self.max_size = max_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device


    def add(self, state, action, next_state, reward, done):
        exp = pickle.dumps((state, action, next_state, reward, done))
        self.client.rpush(self.key, exp)

        # Maintain the FIFO size limit
        self._maintain_size()

    def _maintain_size(self):
        self.client.ltrim(self.key, -self.max_size, -1)

    def sample_best(self, batch_size):
        # Fetch all entries
        buffer_size = self.client.llen(self.key)
        all_entries = [pickle.loads(self.client.lindex(self.key, i)) for i in range(buffer_size)]

        # Sort entries by reward in descending order and take top_k
        samples = sorted(all_entries, key=lambda x: x[3], reverse=True)[:batch_size]

        states = torch.FloatTensor([sample[0] for sample in samples]).to(self.device)
        actions = torch.FloatTensor([sample[1] for sample in samples]).to(self.device)
        next_states = torch.FloatTensor([sample[2] for sample in samples]).to(self.device)
        rewards = torch.FloatTensor([sample[3] for sample in samples]).to(self.device)
        dones = torch.FloatTensor([sample[4] for sample in samples]).to(self.device)

        return states, actions, next_states, rewards, dones

    def sample(self, batch_size):
        buffer_size = self.client.llen(self.key)

        idxs = random.sample(range(buffer_size), batch_size)
        samples = [pickle.loads(self.client.lindex(self.key, i)) for i in idxs]

        # batch = [(pickle.loads(row[1]), pickle.loads(row[2]), pickle.loads(row[3]), row[4], row[5]) for row in rows]

        states = torch.FloatTensor([sample[0] for sample in samples]).to(self.device)
        actions = torch.FloatTensor([sample[1] for sample in samples]).to(self.device)
        next_states = torch.FloatTensor([sample[2] for sample in samples]).to(self.device)
        rewards = torch.FloatTensor([sample[3] for sample in samples]).to(self.device)
        dones = torch.FloatTensor([sample[4] for sample in samples]).to(self.device)

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.client.llen(self.key)

    def view_entries(self):
        # Get the length of the list
        buffer_size = self.client.llen(self.key)

        # Fetch all entries
        entries = [pickle.loads(self.client.lindex(self.key, i)) for i in range(buffer_size)]
        return entries


    def clear(self):
        self.client.delete(self.key)

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )



class FunctionBuffer():

    def __init__(self, input_dim, output_dim, max_size=int(1e2)):
        self.max_size = max_size
        self.memory = np.zeros((max_size, input_dim + output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.position = 0
        self.is_full = False

    def push(self, input_, output_):
        self.memory[self.position, :self.input_dim] = input_
        self.memory[self.position, self.input_dim:] = output_

        if not self.is_full and self.position == self.max_size - 1:
            self.is_full = True

        self.position = (self.position + 1) % self.max_size

    def get_batch(self):
        if self.is_full:
            return self.memory
        else:
            return self.memory[:self.position, :]

    def get_batch_X_Y(self):
        if self.is_full:
            return self.memory[:, :self.input_dim], self.memory[:, self.input_dim:]
        else:
            return self.memory[:self.position, :self.input_dim], self.memory[:self.position, self.input_dim:]



def get_env(envname, args):
    with open('environments.py', 'r') as file:
        base_environments = file.read().replace('\n', '')
        if base_environments is not None and envname in base_environments:
            return get_class_from_module('environments', envname)(**args)

    with open('generated_environments.py', 'r') as file:
        generated_environments = file.read().replace('\n', '')
        if generated_environments is not None and envname in generated_environments:
            return get_class_from_module('generated_environments', envname)(**args)

    with open('environments_experiments.py', 'r') as file:
        generated_environments = file.read().replace('\n', '')
        if generated_environments is not None and envname in generated_environments:
            return get_class_from_module('environments_experiments', envname)(**args)

    with open('environments_2d_experiments.py', 'r') as file:
        generated_environments = file.read().replace('\n', '')
        if generated_environments is not None and envname in generated_environments:
            return get_class_from_module('environments_2d_experiments', envname)(**args)

    return None

# import torch.multiprocessing as mp
#
# def worker(policy, replay_buffer, env_args, args, seed, log_folder_name):
#     args.seed = seed
#
#     render_mode = None
#     if args.render:
#         render_mode = 'human'
#     env = get_env(args.env, env_args)
#
#     np.random.seed(seed)
#     torch.random.manual_seed(seed)
#
#     file_name = f"{args.env}_{args.seed}"
#     log_name = f"{args.policy}_{args.seed}"
#     log_args = {"file_name": file_name, "log_folder_name": log_folder_name, "log_name": log_name}
#
#     train_loop(env=env, replay_buffer=replay_buffer, args=args, log_args=log_args, policy=policy)
#
#
# def main(args, env_args=None):
#     mp.set_start_method('spawn')
#
#     if env_args is None:
#         env_args = {'render_mode': args.render_mode}
#
#     print(args.env, env_args)
#     env =  (args.env, env_args)
#     print(env)
#
#     kwargs = {
#         "state_dim": env.observation_dim,
#         "action_dim": env.action_dim,
#         "max_action": env.max_action,
#         "discount": args.discount,
#         "tau": args.tau,
#         "ensemble_size": args.ensemble_size,
#         "UTD": args.UTD,
#         "depth": args.depth,
#         "num_neurons": args.num_neurons,
#         "device": torch.device(args.device),
#     }
#     log_folder_name = 'logs_vec_agent/' + args.env + '/' + args.policy if args.name is None else args.name
#     if not os.path.exists(log_folder_name):
#         os.makedirs(log_folder_name)
#
#     policy = get_class_from_module(args.policy, "Algorithm")(**kwargs)
#
#     if args.load_model != "":
#         if args.load_model == "default":
#             policy_file = f"{args.env}_{args.seed}"
#             policy.load(log_folder_name + f"/models/{policy_file}", load_critic=False)
#         else:
#             policy.load(args.load_model, load_critic=False)
#
#     replay_buffer = ReplayBuffer(env.observation_dim, env.action_dim, max_size=int(args.seeds * 1e6))
#
#     processes = []
#
#     for seed in range(args.start_seed, args.start_seed + args.seeds):
#         yaml.dump(kwargs, open(log_folder_name + f'/params_{seed}.yaml', 'w'))
#         p = mp.Process(target=worker, args=(policy, replay_buffer, env_args, args, seed, log_folder_name))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
#
#
# if __name__ == '__main__':
#     buffer = SharedReplayBuffer(db_name='logs/YumiFetchEnv/TD3/replay_buffer',
#                                 max_size=1e6 / 2, )
#
#     print('buffer.size()', buffer.size())
#     print(buffer.view_entries())
#
#     print("Final Buffer size:", buffer.size())
#     print("Sampled batch:", buffer.sample(6))