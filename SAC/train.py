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
from agent import SACController



def train_rl_controller(env, episodes=1000, steps_per_episode=1000, save_interval=100):
    """Train the RL controller with multiple reference trajectory types"""
    
    # Initialize controller
    state_dim = 4  # Updated state representation dimension
    action_dim = 1  # voltage
    controller = SACController(state_dim, action_dim)
    
    # Training metrics
    episode_rewards = []
    attack_performance = []
    normal_performance = []
    
    def generate_step_trajectory(step, params):
        """Generate step trajectory with variable parameters"""
        if step < params['step_time']:
            return 0
        return params['amplitude']
    
    def generate_sine_trajectory(step, params):
        """Generate sinusoidal trajectory"""
        return params['amplitude'] * np.sin(2 * np.pi * params['frequency'] * step / steps_per_episode)
    
    def generate_square_trajectory(step, params):
        """Generate square wave trajectory"""
        period = params['period']
        return params['amplitude'] if (step % period) < (period/2) else -params['amplitude']
    
    def generate_ramp_trajectory(step, params):
        """Generate ramp trajectory"""
        return min(params['slope'] * step, params['max_val'])
    
    def generate_constant_trajectory(step, params):
        """Generate constant trajectory"""
        return params['value']
    
    def generate_multi_step_trajectory(step, params):
        """Generate multi-step trajectory with varying amplitudes"""
        step_sequence = params['step_sequence']
        current_segment = step // params['segment_length']
        if current_segment >= len(step_sequence):
            return step_sequence[-1]
        return step_sequence[current_segment]

    def get_random_trajectory_config():
        """Select random trajectory type and parameters"""
        trajectory_types = ['step', 'sine', 'square', 'ramp', 'constant', 'multi_step']
        selected_type = random.choice(trajectory_types)
        
        if selected_type == 'step':
            return {
                'type': 'step',
                'params': {
                    'step_time': random.randint(100, 500),
                    'amplitude': random.uniform(np.pi/6, np.pi/2)
                }
            }
        elif selected_type == 'sine':
            return {
                'type': 'sine',
                'params': {
                    'amplitude': random.uniform(np.pi/6, np.pi/3),
                    'frequency': random.uniform(0.5, 2.0)
                }
            }
        elif selected_type == 'square':
            return {
                'type': 'square',
                'params': {
                    'period': random.randint(100, 300),
                    'amplitude': random.uniform(np.pi/6, np.pi/3)
                }
            }
        elif selected_type == 'ramp':
            return {
                'type': 'ramp',
                'params': {
                    'slope': random.uniform(0.001, 0.003),
                    'max_val': random.uniform(np.pi/4, np.pi/2)
                }
            }
        elif selected_type == 'multi_step':
            num_steps = random.randint(3, 5)
            return {
                'type': 'multi_step',
                'params': {
                    'step_sequence': [random.uniform(-np.pi/3, np.pi/3) for _ in range(num_steps)],
                    'segment_length': random.randint(150, 250)
                }
            }
        else:  # constant
            return {
                'type': 'constant',
                'params': {
                    'value': random.uniform(-np.pi/3, np.pi/3)
                }
            }
    
    print("Starting RL controller training...")
    for episode in range(episodes):
        state, _ = env.reset()
        controller.prev_error = 0.0  # Reset previous error at the start of each episode
        episode_reward = 0
        attack_active = False
        prev_action = 0.0
        
        # Generate random trajectory type for this episode
        trajectory_config = get_random_trajectory_config()
        print(f"Episode {episode+1} using {trajectory_config['type']} trajectory")
        
        # Randomly decide attack parameters for this episode
        will_attack = random.random() < 0.5
        if will_attack:
            attack_start = random.randint(200, 600)
            attack_duration = random.randint(100, 300)
            attack_magnitude = random.uniform(-np.pi/3, np.pi/3)
        
        # Run episode
        for step in range(steps_per_episode):
            # Generate reference based on trajectory type
            if trajectory_config['type'] == 'step':
                reference = generate_step_trajectory(step, trajectory_config['params'])
            elif trajectory_config['type'] == 'sine':
                reference = generate_sine_trajectory(step, trajectory_config['params'])
            elif trajectory_config['type'] == 'square':
                reference = generate_square_trajectory(step, trajectory_config['params'])
            elif trajectory_config['type'] == 'ramp':
                reference = generate_ramp_trajectory(step, trajectory_config['params'])
            elif trajectory_config['type'] == 'multi_step':
                reference = generate_multi_step_trajectory(step, trajectory_config['params'])
            else:  # constant
                reference = generate_constant_trajectory(step, trajectory_config['params'])
            
            env.target_position = reference
            
            # Handle attack timing
            if will_attack and step == attack_start:
                env.start_attack("bias", attack_magnitude)
                attack_active = True
            elif will_attack and step == (attack_start + attack_duration):
                env.stop_attack()
                attack_active = False
            
            # Get state representation
            state_rep = controller.get_state_representation(
                sensor_position=state[1],
                attack_estimate=state[4],
                reference=reference,
                attack_detected=bool(state[3]),
                prev_action=prev_action
            )
            
            # Select action
            if len(controller.replay_buffer) < controller.warmup_steps:
                action = np.random.uniform(-controller.max_action, controller.max_action, size=(1,))
            else:
                action = controller.select_action(state_rep)
            prev_action = float(action[0])
            
            # Step environment
            next_state, reward, done, _, info = env.step(action)
            
            # Store experience
            next_state_rep = controller.get_state_representation(
                sensor_position=next_state[1],
                attack_estimate=next_state[4],
                reference=reference,
                attack_detected=bool(next_state[3]),
                prev_action=prev_action
            )
            
            controller.replay_buffer.push(
                state_rep, action[0], reward, next_state_rep, done
            )
            
            # Train controller
            if len(controller.replay_buffer) >= controller.batch_size:
                controller.train_step()
            
            episode_reward += reward
            state = next_state
            
            # Prevent early termination during training
            if done:
                print(f"Episode {episode+1} ended early at step {step+1}")
                break
        
        # Store metrics
        episode_rewards.append(episode_reward)
        if will_attack:
            attack_performance.append(episode_reward)
        else:
            normal_performance.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}, Average Reward (last 10): {avg_reward:.2f}")
            
            # Print trajectory statistics
            if hasattr(controller, 'critic_losses') and len(controller.critic_losses) > 0:
                print(f"Recent Critic Loss: {np.mean(controller.critic_losses[-100:]):.4f}")
            if hasattr(controller, 'actor_losses') and len(controller.actor_losses) > 0:
                print(f"Recent Actor Loss: {np.mean(controller.actor_losses[-100:]):.4f}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            controller.save(f'sac_controller_checkpoint_{episode+1}.pth')
    
    # Save final model
    controller.save('sac_controller_final.pth')
    
    return controller, {
        'episode_rewards': episode_rewards,
        'attack_performance': attack_performance,
        'normal_performance': normal_performance,
        'critic_losses': controller.critic_losses,
        'actor_losses': controller.actor_losses,
        'alpha_losses': controller.alpha_losses
    }

def plot_training_results(metrics):
    """Plot training metrics"""
    plt.figure(figsize=(15, 15))
    
    # Plot episode rewards
    plt.subplot(3, 1, 1)
    plt.plot(metrics['episode_rewards'], label='Overall Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Training Performance')
    plt.legend()
    plt.grid(True)
    
    # Plot moving averages
    plt.subplot(3, 1, 2)
    window = 100
    if len(metrics['episode_rewards']) >= window:
        moving_avg = np.convolve(metrics['episode_rewards'], 
                                 np.ones(window)/window, mode='valid')
        plt.plot(moving_avg,
                 label=f'Overall ({window}-ep moving avg)')
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Reward')
    plt.title(f'Moving Average Performance ({window} episodes)')
    plt.legend()
    plt.grid(True)
    
    # Plot losses
    plt.subplot(3, 1, 3)
    plt.plot(metrics['critic_losses'], label='Critic Loss')
    plt.plot(metrics['actor_losses'], label='Actor Loss')
    if 'alpha_losses' in metrics and metrics['alpha_losses']:
        plt.plot(metrics['alpha_losses'], label='Alpha Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Losses During Training')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sac_controller_training.pdf', bbox_inches='tight', dpi=300)
    plt.show()
