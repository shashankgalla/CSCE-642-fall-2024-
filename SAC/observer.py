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



# Define the Observer Network
class ObserverNetwork(nn.Module):
    def __init__(self, measurement_dim, control_dim, residual_dim, state_dim, window_size=10):
        super(ObserverNetwork, self).__init__()
        
        # Measurement encoder
        self.measurement_encoder = nn.Sequential(
            nn.Linear(measurement_dim * window_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Control encoder
        self.control_encoder = nn.Sequential(
            nn.Linear(control_dim * window_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Residual encoder
        self.residual_encoder = nn.Sequential(
            nn.Linear(residual_dim * window_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # State estimation
        self.state_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
        
        # Attack estimation
        self.attack_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, measurement_dim)
        )
        
        # Attack detection
        self.attack_detector = nn.Sequential(
            nn.Linear(192, 32),  # 128 from combined features + 64 from residuals
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, measurement_history, control_history, residual_history):
        # Flatten the histories
        flat_measurements = measurement_history.view(1, -1)
        flat_controls = control_history.view(1, -1)
        flat_residuals = residual_history.view(1, -1)
        
        # Extract features
        measurement_features = self.measurement_encoder(flat_measurements)
        control_features = self.control_encoder(flat_controls)
        residual_features = self.residual_encoder(flat_residuals)
        
        combined = torch.cat([measurement_features, control_features], dim=1)
        
        # Generate state and attack magnitude estimates
        state_estimate = self.state_estimator(combined)
        attack_magnitude = self.attack_estimator(combined)
        
        # For attack detection, include residual features
        combined_for_detection = torch.cat([combined, residual_features], dim=1)
        attack_probability = self.attack_detector(combined_for_detection).squeeze(-1)
        
        return state_estimate, attack_magnitude, attack_probability

class AdaptiveObserver:
    def __init__(self, A, B, C, D, dt, state_dim=1, measurement_dim=1, control_dim=1):
        """
        Initialize the RL-based adaptive observer.
        
        Args:
            A, B, C, D: System matrices for the linear model
            dt: Time step
            state_dim: Dimension of the state vector
            measurement_dim: Dimension of the measurement vector
            control_dim: Dimension of the control input vector
        """
        # System matrices
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.dt = dt
        
        # Dimensions
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.control_dim = control_dim
        
        # Set up device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = 100
        
        # Initialize neural network observer
        self.observer_net = ObserverNetwork(
            measurement_dim=measurement_dim,
            control_dim=control_dim,
            residual_dim=measurement_dim,  # Assuming residual has same dimension as measurement
            state_dim=state_dim,
            window_size=self.window_size
        ).to(self.device)
        
        # Training parameters
        self.learning_rate = 1e-4
        self.optimizer = optim.Adam(self.observer_net.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
        
        # History buffers for temporal analysis
        self.measurement_history = torch.zeros(self.window_size, measurement_dim, device=self.device)
        self.control_history = torch.zeros(self.window_size, control_dim, device=self.device)
        self.residual_history = torch.zeros(self.window_size, measurement_dim, device=self.device)
        self.history_index = 0
        
        # State estimation variables
        self.x_hat = np.zeros((state_dim, 1))
        self.attack_magnitude = np.zeros((measurement_dim, 1))
        self.attack_probability = 0.0
        self.prev_error = 0.0
        
        # Statistical analysis buffers
        self.clean_measurement_history = deque(maxlen=50)
        self.measurement_variance = 0.0

    def update(self, y, u, true_state, attack_flag, true_attack_magnitude):
        """
        Update the observer state with new measurements and control inputs.
        While we accept true_state for loss computation, the state estimation
        is based only on sensor measurements.
        
        Args:
            y: Current measurement
            u: Current control input
            true_state: Actual system state (used only for loss computation)
            attack_flag: Boolean indicating if system is under attack
            true_attack_magnitude: Actual attack magnitude
        """
        # Detect attack state transitions for history management
        attack_transition = False
        if hasattr(self, 'prev_attack_flag') and self.prev_attack_flag != attack_flag:
            attack_transition = True

            if not attack_flag:  # Attack just ended
                # Reset history buffers partially to remove attacked measurements
                self.measurement_history *= 0.5
                self.control_history *= 0.5
                self.residual_history *= 0.5
        self.prev_attack_flag = attack_flag

        # Convert inputs to tensors with error handling
        try:
            y_tensor = torch.tensor([y], dtype=torch.float32, device=self.device)
            u_tensor = torch.tensor([u], dtype=torch.float32, device=self.device)
        except Exception as e:
            print(f"Error converting inputs to tensors: {e}")
            return self.x_hat.flatten()[0], self.attack_magnitude.flatten()[0], False

        # Compute residual between measurement and estimated measurement
        estimated_measurement = float(self.C @ self.x_hat)
        residual = y - estimated_measurement

        # Update measurement, control, and residual histories with confidence weighting
        idx = self.history_index % self.window_size
        if attack_flag:
            # Reduce the influence of measurements during attacks
            confidence_weight = 0.3
        else:
            confidence_weight = 1.0

        # Update histories with confidence weighting
        self.measurement_history[idx] = y_tensor * confidence_weight
        self.control_history[idx] = u_tensor
        self.residual_history[idx] = torch.tensor([residual], dtype=torch.float32, device=self.device)
        self.history_index += 1

        # Early return with smooth initialization
        if self.history_index < self.window_size:
            # Use median of recent measurements during initialization
            if self.history_index > 2:
                recent_measurements = self.measurement_history[:self.history_index]
                self.x_hat = torch.median(recent_measurements).cpu().numpy().reshape(-1, 1)
            return self.x_hat.flatten()[0], self.attack_magnitude.flatten()[0], False

        # Prepare history tensors with proper ordering
        measurement_tensor = torch.roll(self.measurement_history, -idx-1, dims=0)
        control_tensor = torch.roll(self.control_history, -idx-1, dims=0)
        residual_tensor = torch.roll(self.residual_history, -idx-1, dims=0)

        # Prepare training tensors for loss computation
        true_state_tensor = torch.tensor([true_state], dtype=torch.float32, device=self.device)
        attack_flag_tensor = torch.tensor([float(attack_flag)], dtype=torch.float32, device=self.device)
        true_attack_magnitude_tensor = torch.tensor([true_attack_magnitude], dtype=torch.float32, device=self.device)

        # Forward pass through neural network
        try:
            state_estimate, attack_mag, attack_prob = self.observer_net(
                measurement_tensor, control_tensor, residual_tensor
            )
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return self.x_hat.flatten()[0], self.attack_magnitude.flatten()[0], False

        # Ensure outputs and targets have matching shapes
        state_estimate = state_estimate.view(-1)
        true_state_tensor = true_state_tensor.view(-1)
        attack_mag = attack_mag.view(-1)
        true_attack_magnitude_tensor = true_attack_magnitude_tensor.view(-1)
        attack_prob = attack_prob.view(-1)
        attack_flag_tensor = attack_flag_tensor.view(-1)

        # Compute losses with adaptive weighting
        state_loss = self.loss_function(state_estimate, true_state_tensor)
        attack_detection_loss = nn.BCELoss()(attack_prob, attack_flag_tensor)
        attack_magnitude_loss = self.loss_function(attack_mag, true_attack_magnitude_tensor)

        # Adjust loss weights based on attack status
        if attack_flag:
            # Increase importance of attack detection during attacks
            detection_weight = 0.5
            magnitude_weight = 0.2
        else:
            detection_weight = 0.2
            magnitude_weight = 0.1

        # Combined loss with adaptive weights
        total_loss = state_loss + detection_weight * attack_detection_loss + magnitude_weight * attack_magnitude_loss

        # Optimization step with error handling
        try:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.observer_net.parameters(), 1.0)
            self.optimizer.step()
        except Exception as e:
            print(f"Error in optimization step: {e}")

        # Update state estimates with robust fusion
        nn_state_estimate = state_estimate.detach().cpu().numpy().reshape(-1, 1)
        attack_detected = attack_prob.item() > 0.5

        # Adaptive state fusion based on attack status and transitions
        if attack_detected:
            # Trust neural network more during attacks
            nn_weight = 0.95
            measurement_weight = 0.05
        elif attack_transition and not attack_flag:
            # Smooth transition after attack ends
            nn_weight = 0.7
            measurement_weight = 0.3
        else:
            # Normal operation - rely more on measurements
            nn_weight = 0.7
            measurement_weight = 0.3

        # Update state estimate using measurement-based fusion
        self.x_hat = (nn_weight * nn_state_estimate +
                      measurement_weight * y.reshape(-1, 1))

        # Store attack-related estimates
        self.attack_magnitude = attack_mag.detach().cpu().numpy().reshape(-1, 1)
        self.attack_probability = attack_prob.item()

        # Implement exponential moving average for attack probability
        if hasattr(self, 'prev_attack_prob'):
            alpha = 0.7  # Smoothing factor
            self.attack_probability = (alpha * self.attack_probability +
                                       (1 - alpha) * self.prev_attack_prob)
        self.prev_attack_prob = self.attack_probability

        # More stable attack detection with hysteresis
        if hasattr(self, 'prev_attack_detected'):
            if self.prev_attack_detected:
                # Higher threshold to turn off attack detection
                attack_detected = self.attack_probability > 0.4
            else:
                # Lower threshold to turn on attack detection
                attack_detected = self.attack_probability > 0.6
        self.prev_attack_detected = attack_detected

        return self.x_hat.flatten()[0], self.attack_magnitude.flatten()[0], attack_detected

    def reset(self):
        """Reset the observer state and all history buffers."""
        # Reset state estimates
        self.x_hat = np.zeros((self.state_dim, 1))
        self.attack_magnitude = np.zeros((self.measurement_dim, 1))
        self.attack_probability = 0.0
        
        # Reset history buffers
        self.measurement_history.zero_()
        self.control_history.zero_()
        self.residual_history.zero_()
        self.history_index = 0
        self.prev_error = 0.0
        
        # Reset statistical buffers
        self.clean_measurement_history.clear()
        self.measurement_variance = 0.0

def train_observer(observer, env, epochs=10, steps_per_epoch=1000):
    """Train the observer on data from the environment."""
    print("Training observer...")
    for epoch in range(epochs):
        state, _ = env.reset()
        for step in range(steps_per_epoch):
            action = np.random.uniform(-env.max_voltage, env.max_voltage, size=(1,))
            next_state, _, done, _, info = env.step(action)
            # Update the observer with the true state
            observer.update(
                y=next_state[1],  # Sensor reading
                u=action[0],
                true_state=next_state[0],  # True physical position
                attack_flag=env.under_attack,
                true_attack_magnitude=env.attack_magnitude
            )
            if done:
                state, _ = env.reset()
            else:
                state = next_state
        print(f"Epoch {epoch+1}/{epochs} completed.")
    # Save the observer
    torch.save(observer.observer_net.state_dict(), 'observer_net.pth')