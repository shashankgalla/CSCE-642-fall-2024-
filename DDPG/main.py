import os
from environment import StepperMotorEnv
from observer import AdaptiveObserver, train_observer
from agent import DDPGController
from train import train_rl_controller
from utils import test_controller
from test_multiple_trajectories import test_controller as test_controller_multiple_trajectories
import torch

def main():
    # Initialize the stepper motor environment
    env = StepperMotorEnv()

    # Initialize the observer with the correct system parameters from the environment
    observer = AdaptiveObserver(A=env.A, B=env.B, C=env.C, D=env.D, dt=env.dt)

    # Set the observer in the environment
    env.adaptive_observer = observer  # This line is crucial

    # Check for a pre-existing observer model
    observer_model_path = 'observer_net.pth'
    if os.path.exists(observer_model_path):
        observer.observer_net.load_state_dict(torch.load(observer_model_path))
        print("Loaded pre-trained observer.")
    else:
        # Train the observer if no pre-trained model is available
        train_observer(observer, env, epochs=100)
        torch.save(observer.observer_net.state_dict(), observer_model_path)
        print("Observer trained and saved.")

    # Initialize the DDPG controller
    controller = DDPGController(state_dim=4, action_dim=1)

    # Train the RL controller, if needed
    # train_rl_controller(env, controller, episodes=1000, steps_per_episode=1000)

    # Test the trained controller
    test_controller(env, controller)

    test_controller_multiple_trajectories(env, controller)


if __name__ == "__main__":
    main()
