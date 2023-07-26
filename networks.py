import torch
from torch.nn import Linear, ReLU

class V_network(torch.nn.Module):
    def __init__(self, num_states, hidden_units_1, hidden_units_2):
        super(V_network, self).__init__()
        self.num_states = num_states
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.activation = ReLU()
        self.fc1 = Linear(in_features=self.num_states, out_features=self.hidden_units_1)
        self.fc2 = Linear(in_features=self.hidden_units_1, out_features=self.hidden_units_2)
        self.output_layer = Linear(self.hidden_units_2, 1)

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device='cuda')
        x = self.fc1(state)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x


class Q_network(torch.nn.Module):
    def __init__(self, num_states, num_actions, hidden_units_1, hidden_units_2):
        super(Q_network, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.activation = ReLU()
        self.fc1 = Linear(in_features=self.num_states + self.num_actions, out_features=self.hidden_units_1)
        self.fc2 = Linear(in_features=self.hidden_units_1, out_features=self.hidden_units_2)
        self.output_layer = Linear(in_features=self.hidden_units_2, out_features=1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class Actor_Network(torch.nn.Module):
    def __init__(self, num_states, num_actions, hidden_units_1, hidden_units_2):
        super(Actor_Network, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.activation = ReLU()
        self.fc1 = Linear(in_features=self.num_states, out_features=self.hidden_units_1)
        self.fc2 = Linear(in_features=self.hidden_units_1, out_features=self.hidden_units_2)
        self.mu = Linear(in_features=self.hidden_units_2, out_features=self.num_actions)
        self.sigma = Linear(in_features=self.hidden_units_2, out_features=self.num_actions)

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device='cuda')
        x = self.fc1(state)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        mean = self.mu(x)
        var = self.sigma(x)
        return mean, var

    def sample_actions(self, state):
        mean, var = self.forward(state)
        probability_distribution = torch.distributions.Normal(mean, var)
        actions = probability_distribution.sample()
        log_probs = probability_distribution.log_prob(actions)
        return torch.clamp(actions, -1, 1) * 0.0015, log_probs


