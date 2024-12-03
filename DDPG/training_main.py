import os
from environment import StepperMotorEnv
from observer import AdaptiveObserver, train_observer
from agent import DDPGController
from train import train_rl_controller
from utils import test_controller
from test_multiple_trajectories import test_controller as test_controller_multiple_trajectories
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
    if os.path.exists('observer_net.pth'):
        # Load the pre-trained observer
        loaded_observer.observer_net.load_state_dict(torch.load('observer_net.pth'))
        print("Loaded pre-trained observer.")
    else:
        # Train the observer
        train_observer(loaded_observer, env_for_loading, epochs=1000)
        # Save the observer
        torch.save(loaded_observer.observer_net.state_dict(), 'observer_net.pth')
        print("Trained observer saved.")
    
    # 2. Create environment with trained observer and prevent early termination during training
    env = StepperMotorEnv(adaptive_observer=loaded_observer, terminate_on_out_of_bounds=False)

    controller, training_metrics = train_rl_controller(env, episodes=1000, steps_per_episode=1000)
    # Save the trained RL controller
    
    print("Trained RL controller saved.")
    
    
if __name__ == "__main__":
    main() 
