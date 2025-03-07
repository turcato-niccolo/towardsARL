import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.nn.init import calculate_gain

class DeterministicActor(nn.Module):
    def __init__(self, state_dim, action_dim, device, max_action, depth=2, num_neurons=512):
        super(DeterministicActor, self).__init__()

        self.device = device
        self.layers = []

        self.layers.append(nn.Linear(state_dim, num_neurons, device=device, dtype=torch.float32))
        for _ in range(depth-1):
            self.layers.append(nn.Linear(num_neurons, num_neurons, device=device, dtype=torch.float32))
        self.layers.append(nn.Linear(num_neurons, action_dim, device=device, dtype=torch.float32))

        self.layers = nn.ParameterList(self.layers)
        self.depth = depth
        self.num_neurons = num_neurons
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.layers[0](state))
        for k in range(self.depth-1):
            a = F.relu(self.layers[k+1](a))

        return self.max_action * torch.tanh(self.layers[-1](a))

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def save_to_csv_files(self, path):
        np.savetxt(path+'l0_w.csv', self.layers[0].weight.detach().cpu().numpy(), delimiter=',')
        np.savetxt(path+'l0_b.csv', self.layers[0].bias.detach().cpu().numpy(), delimiter=',')
        for k in range(self.depth - 1):
            np.savetxt(path + f'l{k+1}_w.csv', self.layers[k+1].weight.detach().cpu().numpy(), delimiter=',')
            np.savetxt(path + f'l{k+1}_b.csv', self.layers[k+1].bias.detach().cpu().numpy(), delimiter=',')
        np.savetxt(path + f'l{self.depth}_w.csv', self.layers[self.depth].weight.detach().cpu().numpy(), delimiter=',')
        np.savetxt(path + f'l{self.depth}_b.csv', self.layers[self.depth].bias.detach().cpu().numpy(), delimiter=',')


class Actor_with_uncertainty(nn.Module):
    def __init__(self, state_dim, action_dim, device, max_action, depth=2, num_neurons=512):
        super(Actor_with_uncertainty, self).__init__()
        self.device = device
        self.layers = []

        self.layers.append(nn.Linear(state_dim+1, num_neurons, device=device, dtype=torch.float32))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(num_neurons, num_neurons, device=device, dtype=torch.float32))
        self.layers.append(nn.Linear(num_neurons, action_dim, device=device, dtype=torch.float32))

        self.layers = nn.ParameterList(self.layers)
        self.depth = depth
        self.num_neurons = num_neurons
        self.max_action = max_action

    def forward(self, state, uncertain):
        su = torch.cat([state, uncertain], 1)  # .unsqueeze(0).expand(state.size(dim=0), -1, -1)
        a = F.relu(self.layers[0](su))
        for k in range(self.depth-1):
            a = F.relu(self.layers[k+1](a))
        return self.max_action * torch.tanh(self.layers[-1](a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device, depth=2, num_neurons=512):
        super(Critic, self).__init__()

        # Q architecture
        self.layers = []

        self.layers.append(nn.Linear(state_dim+action_dim, num_neurons, device=device))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(num_neurons, num_neurons, device=device))
        self.layers.append(nn.Linear(num_neurons, 1, device=device))

        self.layers = nn.ParameterList(self.layers)
        self.depth = depth
        self.num_neurons = num_neurons


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        sa = F.relu(self.layers[0](sa))
        for k in range(self.depth-1):
            sa = F.relu(self.layers[k+1](sa))

        return self.layers[-1](sa)

class Critic_ensemble(nn.Module):

    def __init__(self, state_dim, action_dim, device, depth=2, num_neurons=512, ensemble_size=10):
        super(Critic_ensemble, self).__init__()
        self.depth = depth
        self.num_neurons = num_neurons

        # Q architecture
        self.layers = []

        self.layers.append(EnsembleFC(state_dim + action_dim + 1, num_neurons, ensemble_size=ensemble_size, device=device))
        for _ in range(depth - 1):
            self.layers.append(EnsembleFC(num_neurons, num_neurons, ensemble_size=ensemble_size, device=device))
        self.layers.append(EnsembleFC(num_neurons, 1, ensemble_size=ensemble_size, device=device))

        self.layers = nn.ParameterList(self.layers)

        self.ensemble_size = ensemble_size
        self.device = device

    def forward(self, state, action, uncertain):
        q = self.get_Q_estimates(state, action, uncertain)

        mean = q.mean(0)
        std = q.std(0)

        return mean, std

    def get_Q_estimates(self, state, action, uncertain):
        # print('state.size():', state.size())
        # print('action.size():', action.size())
        # print('uncertain.size():', uncertain.size())

        sau = torch.cat([state, action, uncertain], -1).unsqueeze(0).expand(self.ensemble_size, -1, -1)
        q = F.relu(self.layers[0](sau))
        for k in range(self.depth-1):
            q = F.relu(self.layers[k+1](q))

        return q

    def get_Q_estimate(self, state, action, uncertain, k):
        Qs = self.get_Q_estimates(state, action, uncertain)
        return Qs[k, :, :]

class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, device: torch.device, bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.device = device
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features).to(self.device))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features).to(self.device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

def kaiming_uniform_(tensor, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.size(-2)
    num_output_fmaps = tensor.size(-1)
    receptive_field_size = 1
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out