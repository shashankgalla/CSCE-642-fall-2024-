import os
from environment import StepperMotorEnv
from observer import AdaptiveObserver, train_observer
from agent import SACController
from train import train_rl_controller, plot_training_results
from utils import test_controller

import torch
import numpy as np

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # 1. Initialize observer
    env_for_loading = StepperMotorEnv(adaptive_observer=None)
    loaded_observer = AdaptiveObserver(
        A=env_for_loading.A, 
        B=env_for_loading.B, 
        C=env_for_loading.C, 
        D=env_for_loading.D, 
        dt=env_for_loading.dt
    )
    
    # Check if the observer network exists
    if os.path.exists('Observer_model_1000_episodes__.pth'):
        # Load the pre-trained observer
        loaded_observer.observer_net.load_state_dict(torch.load('Observer_model_1000_episodes__.pth'))
        print("Loaded pre-trained observer.")
    else:
        # Train the observer
        train_observer(loaded_observer, env_for_loading, epochs=5)
    
    # 2. Create environment with trained observer and prevent early termination during training
    env = StepperMotorEnv(adaptive_observer=loaded_observer, terminate_on_out_of_bounds=False)
    
    # 3. Train SAC controller
    print("Starting SAC controller training...")
    controller, training_metrics = train_rl_controller(env, episodes=5, steps_per_episode=1000)
    
    # 4. Plot training results
    plot_training_results(training_metrics)
    
    # 5. Test the trained controller
    print("\nTesting trained controller...")
    test_controller(env, controller)
    
    
if __name__ == "__main__":
    main() 
