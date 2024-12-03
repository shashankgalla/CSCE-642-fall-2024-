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
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.max_action = 24.0  # Maximum voltage
        
    def forward(self, state):
        return self.network(state) * self.max_action

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class DDPGController:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr_actor=1e-5, lr_critic=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training parameters
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 128
        self.warmup_steps = 1000
        self.max_action = 24.0
        
        # Exploration noise parameters
        self.noise_std = 2.0
        self.noise_decay = 0.9995
        self.noise_min = 0.1
        
        # Statistics
        self.training_step = 0
        self.episode_reward = 0
        self.rewards_history = []
        self.critic_losses = []  # List to store critic losses
        
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
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().detach().numpy()
        
        if not evaluate:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise
            # Ensure action stays within valid range
            action = np.clip(action, -self.max_action, self.max_action)
            # Decay the exploration noise over time
            self.noise_std = max(self.noise_std * self.noise_decay, self.noise_min)
        else:
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
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
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_actions = next_actions.clamp(-self.max_action, self.max_action)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q
            
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        actor_actions = self.actor(states)
        actor_actions = actor_actions.clamp(-self.max_action, self.max_action)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Logging for monitoring
        if self.training_step % 1000 == 0:
            print(f"Training Step: {self.training_step}, Critic Loss: {critic_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}")
        
        # Store critic loss for analysis
        self.critic_losses.append(critic_loss.item())
    
    def save(self, path):
        """Save controller state"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load controller state"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])