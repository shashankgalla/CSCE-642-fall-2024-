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
from agent import DDPGController 



def train_rl_controller(env, episodes=1000, steps_per_episode=1000, save_interval=100):
    """Train the RL controller with the pre-trained observer using diverse reference trajectories"""
    
    # Initialize controller
    state_dim = 4  # Updated state representation dimension
    action_dim = 1  # voltage
    controller = DDPGController(state_dim, action_dim)
    
    # Training metrics
    episode_rewards = []
    attack_performance = []
    normal_performance = []
    
    def generate_step_trajectory(step, step_params):
        """Generate step function trajectory with varying parameters"""
        step_time = step_params['step_time']
        step_amplitude = step_params['amplitude']
        if step < step_time:
            return 0
        return step_amplitude
    
    def generate_sine_trajectory(step, sine_params):
        """Generate sinusoidal trajectory"""
        amplitude = sine_params['amplitude']
        frequency = sine_params['frequency']
        return amplitude * np.sin(2 * np.pi * frequency * step / steps_per_episode)
    
    def generate_square_trajectory(step, square_params):
        """Generate square wave trajectory"""
        period = square_params['period']
        amplitude = square_params['amplitude']
        return amplitude if (step % period) < (period/2) else -amplitude
    
    def generate_ramp_trajectory(step, ramp_params):
        """Generate ramp trajectory"""
        slope = ramp_params['slope']
        max_val = ramp_params['max_val']
        return min(slope * step, max_val)
    
    def generate_constant_trajectory(step, constant_params):
        """Generate constant trajectory"""
        return constant_params['value']
    
    def get_random_trajectory_type():
        """Randomly select trajectory type and parameters"""
        trajectory_types = ['step', 'sine', 'square', 'ramp', 'constant']
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
        trajectory_config = get_random_trajectory_type()
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
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            controller.save(f'rl_controller_checkpoint_{episode+1}.pth')
    
    # Save final model
    controller.save('rl_controller_final.pth')
    
    return controller, {
        'episode_rewards': episode_rewards,
        'attack_performance': attack_performance,
        'normal_performance': normal_performance,
        'critic_losses': controller.critic_losses
    }

