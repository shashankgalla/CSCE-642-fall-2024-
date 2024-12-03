import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
from gymnasium import Env, spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import os

# Experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, float(done)))  # Convert done to float
        
    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([exp.state for exp in experiences])
        actions = torch.FloatTensor([exp.action for exp in experiences]).unsqueeze(-1)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).unsqueeze(-1)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences])
        dones = torch.FloatTensor([exp.done for exp in experiences]).unsqueeze(-1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        
        self.hidden = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        self.action_scale = 24.0  # Maximum voltage
        self.action_bias = 0.0
        
    def forward(self, state):
        x = self.hidden(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(x)
        return q1

class SACController:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, alpha=0.2, automatic_entropy_tuning=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training parameters
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 128
        self.warmup_steps = 1000
        self.max_action = 24.0
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # Entropy temperature
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.tensor(np.log(self.alpha)).to(self.device)
            self.log_alpha.requires_grad = True
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Statistics
        self.training_step = 0
        self.episode_reward = 0
        self.rewards_history = []
        self.critic_losses = []
        self.alpha_losses = []
        self.actor_losses = []

        # For state representation
        self.prev_error = 0.0  # Initialize previous error

    def get_state_representation(self, sensor_position, attack_estimate, 
                                 reference, attack_detected, prev_action=None):
        if attack_detected:
            # Compensate sensor reading with attack estimate
            compensated_position = sensor_position - attack_estimate
        else:
            compensated_position = sensor_position

        # Normalize position error
        position_error = (compensated_position - reference) / np.pi

        # Change in error
        delta_error = position_error - getattr(self, 'prev_error', 0.0)
        self.prev_error = position_error

        # State representation
        state = np.array([
            position_error,
            delta_error,
            float(attack_detected),
            (prev_action if prev_action is not None else 0.0) / self.max_action
        ])

        return state

    def select_action(self, state, evaluate=False):
        """Select action using the current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            _, _, action = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]
        else:
            action, _, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]

    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return

        self.training_step += 1

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Critic loss
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            next_q1_target, next_q2_target = self.critic_target(next_states, next_actions)
            min_next_q_target = torch.min(next_q1_target, next_q2_target) - self.alpha * next_log_pi
            target_q = rewards + (1 - dones) * self.gamma * min_next_q_target

        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        pi, log_pi, _ = self.actor.sample(states)
        q1_pi = self.critic.q1(states, pi)
        actor_loss = (self.alpha * log_pi - q1_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Temperature (alpha) loss
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            self.alpha_losses.append(alpha_loss.item())
        else:
            alpha_loss = 0

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Logging for monitoring
        if self.training_step % 1000 == 0:
            print(f"Training Step: {self.training_step}, Critic Loss: {critic_loss.item():.4f}, "
                  f"Actor Loss: {actor_loss.item():.4f}, Alpha: {self.alpha:.4f}")

        # Store losses for analysis
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())

    def save(self, path):
        """Save controller state"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None
        }, path)

    def load(self, path):
        """Load controller state"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha = checkpoint['alpha']
        self.log_alpha = checkpoint['log_alpha']
        if self.automatic_entropy_tuning and 'alpha_optimizer_state_dict' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])