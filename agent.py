import torch
from networks import V_network, Q_network, Actor_Network
from torch.optim import Adam
from replay_buffer import ReplayBuffer

# class Agent:
#     def __init__(self, num_states, num_actions, hidden_units_1=256, hidden_units_2=256,
#                  learning_rate=0.001, discount_factor=0.99, tau=0.001, alpha=1, memory_size=100000, batch_size=128):
#         #Hyperparameters
#         self.num_states = num_states
#         self.num_actions = num_actions
#         self.hidden_units_1 = hidden_units_1
#         self.hidden_units_2 = hidden_units_2
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.tau = tau
#         self.alpha = alpha
#         self.memory_size = memory_size
#         self.batch_size = batch_size
#         #Network initialization
#         self.actor_net = Actor_Network(num_states=self.num_states, num_actions=self.num_actions,
#                                hidden_units_1=self.hidden_units_1, hidden_units_2=self.hidden_units_2).to(device='cuda')
#         self.q_net1 = Q_network(num_states=self.num_states, num_actions=self.num_actions,
#                                hidden_units_1=self.hidden_units_1, hidden_units_2=self.hidden_units_2).to(device='cuda')
#         # self.q_net2 = Q_network(num_states=self.num_states, num_actions=self.num_actions,
#         #                        hidden_units_1=self.hidden_units_1, hidden_units_2=self.hidden_units_2).to(device='cuda')
#         self.v_net = V_network(num_states=self.num_states, hidden_units_1=self.hidden_units_1,
#                                hidden_units_2=self.hidden_units_2).to(device='cuda')
#         self.target_v_net = V_network(num_states=self.num_states, hidden_units_1=self.hidden_units_1,
#                                hidden_units_2=self.hidden_units_2).to(device='cuda')
#         #Optimizers
#         self.actor_optim = Adam(params=self.actor_net.parameters(), lr=self.learning_rate)
#         self.q1_optim = Adam(params=self.q_net1.parameters(), lr=self.learning_rate)
#         # self.q2_optim = Adam(params=self.q_net2.parameters(), lr=self.learning_rate)
#         self.v_net_optim = Adam(params=self.v_net.parameters(), lr=self.learning_rate)
#         #Memory initialization
#         self.memory = ReplayBuffer(mem_size=self.memory_size, num_states=self.num_states,
#                                    num_actions=self.num_actions, batch_size=self.batch_size)
#         #At this point do nothing
#         self.polyak_update(target_network=self.target_v_net, network=self.v_net, tau=1)
#
#     def polyak_update(self, target_network, network, tau):
#         #Slowly updates the target networks
#         for target_param, param in zip(target_network.parameters(), network.parameters()):
#             target_param.data.copy_(tau * param.data + target_param.data * (1.0 - tau))
# #
#     def train_step(self):
# #         #With 2 Q networks
# #         # states, actions, next_states, rewards, terms, truncs = self.memory.sample_memories()
# #         # #Compute loss of V
# #         # v_output = self.v_net.forward(states) #V_ψ(st)
# #         # a_t, log_probs = self.actor_net.sample_actions(states)
# #         # q1 = self.q_net1.forward(states, a_t)
# #         # q2 = self.q_net2.forward(states, a_t)
# #         # q = torch.min(q1, q2) #Q_θ(s,a)
# #         # first_mean = torch.mean(q - log_probs)
# #         # inside = torch.pow(v_output - first_mean, 2)
# #         # v_loss = torch.mean(inside)
# #         #
# #         # #Compute loss of Q1 and Q2
# #         # v_hat_output = self.target_v_net.forward(next_states)
# #         # q_hat = rewards + self.discount_factor * torch.mean(v_output)
# #         # inside_q1_loss = torch.pow(q1 - q_hat, 2)
# #         # inside_q2_loss = torch.pow(q2 - q_hat, 2)
# #         # q1_loss = torch.mean(inside_q1_loss)
# #         # q2_loss = torch.mean(inside_q2_loss)
# #         #
# #         # #Compute loss of actor
# #         # actor_loss = torch.mean(log_probs - q) #Προς το παρον οχι reparametrization
# #         #
# #         # # print("BEFORE STEP: \n")
# #         # # print(f"V LOSS: {v_loss.item()} \n")
# #         # # print(f"Q1 LOSS: {q1_loss.item()} \n")
# #         # # print(f"Q2 LOSS: {q2_loss.item()} \n")
# #         # # print(f"ACTOR LOSS: {actor_loss.item()} \n")
# #         #
# #         #
# #         # #Backpropagation
# #         # self.v_net_optim.zero_grad()
# #         # self.q1_optim.zero_grad()
# #         # self.q2_optim.zero_grad()
# #         # self.actor_optim.zero_grad()
# #         # v_loss.backward(retain_graph=True)
# #         # q1_loss.backward(retain_graph=True)
# #         # q2_loss.backward(retain_graph=True)
# #         # actor_loss.backward(retain_graph=True)
# #         # self.v_net_optim.step()
# #         # self.q1_optim.step()
# #         # self.q2_optim.step()
# #         # self.actor_optim.step()
# #         #
# #         # # print("AFTER STEP: \n")
# #         # # print(f"V LOSS: {v_loss.item()} \n")
# #         # # print(f"Q1 LOSS: {q1_loss.item()} \n")
# #         # # print(f"Q2 LOSS: {q2_loss.item()} \n")
# #         # # print(f"ACTOR LOSS: {actor_loss.item()} \n")
# #         #
# #         # #Update target network
# #         # self.polyak_update(target_network=self.target_v_net,
# #         #                    network=self.v_net, tau=self.tau)
# #
# #         #With 1 Q network
#         states, actions, next_states, rewards, terms, truncs = self.memory.sample_memories()
#         #Compute loss of V
#         self.v_net_optim.zero_grad()
#         v_output = self.v_net.forward(states) #V_ψ(st)
#         a_t, log_probs = self.actor_net.sample_actions(states)
#         q = self.q_net1.forward(states, a_t)
#         first_mean = torch.mean(q - self.alpha * log_probs, 1, keepdim=True)
#         v_loss = 0.5 * torch.nn.functional.mse_loss(v_output, first_mean)
#         v_loss.backward(retain_graph=True)
#         self.v_net_optim.step()
#         #
#         # #Compute loss of Q1 and Q2
#         self.q1_optim.zero_grad()
#         v_hat_output = self.target_v_net.forward(next_states)
#         q_hat = rewards + self.discount_factor * torch.mean(v_hat_output) \
#                 * (1 - torch.logical_or(terms, truncs).long())
#         # inside_q1_loss = torch.pow(q - q_hat, 2)
#         q_ = self.q_net1.forward(states, actions)
#         q1_loss = 0.5 * torch.nn.functional.mse_loss(q_, q_hat)
#         q1_loss.backward(retain_graph=True)
#         self.q1_optim.step()
#         #
#         #
#         # #Compute loss of actor
#         self.actor_optim.zero_grad()
#         actor_loss = torch.mean(log_probs - q_) #Προς το παρον οχι reparametrization
#         actor_loss.backward()
#         self.actor_optim.step()
#
#         #Update target network
#         self.polyak_update(target_network=self.target_v_net, network=self.v_net, tau=self.tau)

class Agent:
    def __init__(self, num_states, num_actions, hidden_units_1=256, hidden_units_2=256,
                 learning_rate=0.001, discount_factor=0.99, tau=0.005, alpha=0.25, memory_size=500000,
                 batch_size=128):
        # Hyperparameters
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.alpha = alpha
        self.memory_size = memory_size
        self.batch_size = batch_size
        # Network initialization
        self.actor_net = Actor_Network(num_states=self.num_states, num_actions=self.num_actions,
                                       hidden_units_1=self.hidden_units_1,
                                       hidden_units_2=self.hidden_units_2).to(device='cuda')
        self.q_net1 = Q_network(num_states=self.num_states, num_actions=self.num_actions,
                                hidden_units_1=self.hidden_units_1, hidden_units_2=self.hidden_units_2).to(
            device='cuda')
        self.q_net2 = Q_network(num_states=self.num_states, num_actions=self.num_actions,
                               hidden_units_1=self.hidden_units_1, hidden_units_2=self.hidden_units_2).to(device='cuda')
        self.target_q1 = Q_network(num_states=self.num_states, num_actions=self.num_actions,
                               hidden_units_1=self.hidden_units_1, hidden_units_2=self.hidden_units_2).to(device='cuda')
        self.target_q2 = Q_network(num_states=self.num_states, num_actions=self.num_actions,
                               hidden_units_1=self.hidden_units_1, hidden_units_2=self.hidden_units_2).to(device='cuda')
        # Optimizers
        self.actor_optim = Adam(params=self.actor_net.parameters(), lr=self.learning_rate)
        self.q1_optim = Adam(params=self.q_net1.parameters(), lr=self.learning_rate)
        self.q2_optim = Adam(params=self.q_net2.parameters(), lr=self.learning_rate)
        # Memory initialization
        self.memory = ReplayBuffer(mem_size=self.memory_size, num_states=self.num_states,
                                   num_actions=self.num_actions, batch_size=self.batch_size)
        # At this point do nothing
        self.polyak_update(target_network=self.target_q1, network=self.q_net1, tau=1)
        self.polyak_update(target_network=self.target_q2, network=self.q_net2, tau=1)

    def polyak_update(self, target_network, network, tau):
        # Slowly updates the target networks
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau * param.data + target_param.data * (1.0 - tau))

        #From the OPEN AI SITE
    def train_step(self):
        states, actions, next_states, rewards, terms, truncs = self.memory.sample_memories()


        a_prime, log_probs = self.actor_net.sample_actions(next_states)
        a_reparam = self.actor_net.sample_reparam_actions(states)
        y = rewards + self.discount_factor * (1 - torch.logical_or(terms, truncs).long()) *\
            (torch.min(self.target_q1.forward(next_states, a_prime), self.target_q2.forward(next_states, a_prime))
             - self.alpha * log_probs)
        self.q1_optim.zero_grad()
        self.q2_optim.zero_grad()
        q1_loss = torch.nn.functional.mse_loss(self.q_net1.forward(states, actions), y)
        q2_loss = torch.nn.functional.mse_loss(self.q_net2.forward(states, actions), y)
        q1_loss.backward(retain_graph=True)
        self.q1_optim.step()
        q2_loss.backward(retain_graph=True)
        self.q2_optim.step()

        self.actor_optim.zero_grad()
        actor_loss = torch.nn.functional.mse_loss(torch.min(self.q_net1.forward(states, a_reparam),
                                                            self.q_net2.forward(states, a_reparam)), self.alpha * log_probs)
        actor_loss.backward()
        self.actor_optim.step()

        self.polyak_update(self.target_q1, self.q_net1, self.tau)
        self.polyak_update(self.target_q2, self.q_net2, self.tau)








