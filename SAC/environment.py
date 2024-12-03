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

# Define the environment
class StepperMotorEnv(Env):
    def __init__(self, adaptive_observer=None, terminate_on_out_of_bounds=True):
        super().__init__()
        
        # Motor Parameters
        self.J = 0.01      # moment of inertia
        self.b = 0.5       # damping coefficient
        self.K = 0.5       # torque constant
        self.R = 1.0       # motor resistance
        self.L = 0.5       # motor inductance
        self.dt = 0.01     # time step
        self.terminate_on_out_of_bounds = terminate_on_out_of_bounds
        # State Space Limits
        self.max_position = np.pi
        self.max_velocity = 10.0
        self.max_voltage = 24.0
        self.uncertainty_bound = 0.1
        
        # System matrices for linear model
        self.A = np.array([[0]])
        self.B = np.array([[self.K/(self.J*self.R)]])
        self.C = np.array([[1]])
        self.D = np.array([[0]])
        
        # History window for reconstruction
        self.window_size = 50
        self.state_history = deque(maxlen=self.window_size)
        self.control_history = deque(maxlen=self.window_size)
        
        # Action space
        self.action_space = spaces.Box(
            low=-self.max_voltage,
            high=self.max_voltage,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([-self.max_position, -self.max_position, -self.max_position, 0, -np.pi]),
            high=np.array([self.max_position, self.max_position, self.max_position, 1, np.pi]),
            dtype=np.float32
        )
        
        # Attack parameters
        self.under_attack = False
        self.attack_type = None
        self.attack_magnitude = 0.0

        # Safety margins
        self.safe_margin = np.pi / 2
        self.target_margin = np.pi / 12
        self.target_position = 0.0

        # Noise parameters
        self.sensor_noise_std = 0.01
        self.process_noise_std = 0.005

        # Reconstruction parameters
        self.last_trustworthy_state = None
        self.last_trustworthy_time = 0

        # Initialize the RL-based adaptive observer
        if adaptive_observer is not None:
            self.adaptive_observer = adaptive_observer
        else:
            self.adaptive_observer = None  # No observer during training

        # Initial reset to set up state
        self.state = None
        self.prev_velocity = None
        self.prev_error = None
        self.time_step = 0
        state, _ = self.reset()

    def reset(self, seed=None):
        """Reset the environment to an initial state."""
        # First handle the superclass reset (for proper random seed handling)
        super().reset(seed=seed)
        
        # Perform the actual reset
        initial_state = self._initial_reset()
        
        # Return state and empty info dict as per gymnasium interface
        return initial_state, {}

    def _initial_reset(self):
        """Initial environment reset to set up state."""
        # Generate initial position within safe limits
        initial_position = float(np.random.uniform(-np.pi/6, np.pi/6))
        
        # Clear history buffers
        self.state_history.clear()
        self.control_history.clear()
        
        # Reset dynamic states
        self.prev_velocity = 0.0
        self.prev_error = 0.0
        self.time_step = 0
        
        # Reset attack states
        self.under_attack = False
        self.attack_type = None
        self.attack_magnitude = 0.0

        # Reset the observer
        if self.adaptive_observer is not None:
            self.adaptive_observer.reset()
        
        # Generate initial sensor reading with noise
        initial_sensor_noise = np.random.normal(0, self.sensor_noise_std)
        initial_state = np.array([
            initial_position,                    # Physical position
            initial_position + initial_sensor_noise,  # Sensor reading
            initial_position,                    # Estimated position
            0,                                  # Attack detection status
            0                                   # Estimated attack magnitude
        ])
        
        # Set reconstruction parameters
        self.last_trustworthy_state = float(initial_position)
        self.last_trustworthy_time = 0
        
        # Update environment state
        self.state = initial_state
        
        return initial_state

    def simulate_motor_dynamics(self, voltage, current_position, current_velocity):
        """Simulate physical motor dynamics with process noise"""
        # Calculate acceleration based on motor physics
        acceleration = (self.K * voltage / self.R - self.b * current_velocity) / self.J
        
        # Add process noise to acceleration
        process_noise = np.random.normal(0, self.process_noise_std)
        acceleration += process_noise
        
        # Update velocity
        new_velocity = current_velocity + acceleration * self.dt
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)
        
        # Update position
        new_position = current_position + new_velocity * self.dt
        new_position = np.clip(new_position, -self.max_position, self.max_position)
        
        return new_position, new_velocity

    def reconstruct_state(self, current_time):
        """State reconstructor using control history"""
        if self.last_trustworthy_state is None:
            return (float(self.state[0]), 0.0)
                
        Na = current_time - self.last_trustworthy_time
        if Na <= 0:
            return (float(self.last_trustworthy_state), 0.0)
                
        reconstructed_state = float(self.last_trustworthy_state)
        
        for i in range(min(Na, len(self.control_history))):
            control_term = float(self.B * self.control_history[-i-1])
            reconstructed_state += control_term * self.dt
                
        uncertainty_sum = float(Na * self.uncertainty_bound)
            
        return (reconstructed_state, uncertainty_sum)

    def step(self, action):
        """Perform one time step within the environment."""
        self.time_step += 1
        current_time = self.time_step
        
        # Get current physical state
        current_physical_position = float(self.state[0])
        current_velocity = self.prev_velocity
        
        # Use the action (voltage) provided by the agent
        voltage = float(np.clip(action[0], -self.max_voltage, self.max_voltage))
        
        # Simulate true physical dynamics
        new_physical_position, new_velocity = self.simulate_motor_dynamics(
            voltage, current_physical_position, current_velocity
        )
        self.prev_velocity = new_velocity
        
        # Generate sensor reading from physical position
        sensor_noise = np.random.normal(0, self.sensor_noise_std)
        sensor_position = new_physical_position + sensor_noise
        
        # Apply attack if active
        if self.under_attack:
            sensor_position += self.attack_magnitude
        
        # Use RL observer to estimate state and detect attacks
        if self.adaptive_observer is not None:
            x_hat, a_hat, attack_detected = self.adaptive_observer.update(
                sensor_position,    # Use sensor reading as measurement
                voltage,
                new_physical_position,   # true_state
                self.under_attack,       # attack_flag
                self.attack_magnitude    # true_attack_magnitude
            )
        else:
            x_hat = 0.0
            a_hat = 0.0
            attack_detected = False
        
        # Store history
        self.control_history.append(voltage)
        self.state_history.append(sensor_position)
        
        # Update trustworthy state if not under attack
        if not self.under_attack:
            self.last_trustworthy_state = sensor_position
            self.last_trustworthy_time = current_time
        
        # Get reconstructed state
        reconstructed_state, uncertainty = self.reconstruct_state(current_time)
        
        # Update state vector
        self.state = np.array([
            new_physical_position,  # True physical position
            sensor_position,        # Sensor reading (with noise/attack)
            x_hat,                  # Estimated position from RL observer
            int(attack_detected),   # Attack detection status
            a_hat                   # Estimated attack magnitude
        ])
        
        # Calculate position and velocity errors
        position_error = new_physical_position - self.target_position
        velocity_error = new_velocity
        
        # Define thresholds
        position_threshold = 0.05  # radians
        
        # Base reward: negative absolute position error
        reward = -abs(position_error)
        
        # Positive reward when close to the target
        if abs(position_error) < position_threshold:
            reward += 10.0
        
        # Penalty for large actions to discourage excessive control inputs
        reward -= 0.01 * (voltage)**2
        
        # Check if within safe set
        out_of_bounds = not self.check_safe_set(new_physical_position)
        done = False
        if self.terminate_on_out_of_bounds:
            done = out_of_bounds
        
        info = {
            'physical_position': new_physical_position,
            'sensor_position': sensor_position,
            'estimated_position': x_hat,
            'estimated_attack': a_hat,
            'attack_detected': attack_detected,
            'in_safe_set': self.check_safe_set(new_physical_position),
            'in_target_set': self.check_target_set(new_physical_position),
            'control_input': voltage
        }
        
        return self.state, reward, done, False, info

    def start_attack(self, attack_type="bias", magnitude=0.5):
        """Start sensor attack"""
        self.under_attack = True
        self.attack_type = attack_type
        self.attack_magnitude = magnitude

    def stop_attack(self):
        """Stop sensor attack"""
        self.under_attack = False
        self.attack_type = None
        self.attack_magnitude = 0.0

    def check_safe_set(self, position):
        """Check if position is within safe set"""
        safe_lower = self.target_position - self.safe_margin
        safe_upper = self.target_position + self.safe_margin
        return safe_lower <= position <= safe_upper

    def check_target_set(self, position):
        """Check if position is within target set"""
        target_lower = self.target_position - self.target_margin
        target_upper = self.target_position + self.target_margin
        return target_lower <= position <= target_upper
