# Deep Reinforcement Learning Based Attack Mitigation and Recovery controller (Reinforcement Learning Course Project)
CSCE-642-fall-2024-TANU


## Project Overview
This project implements a secure control system for a stepper motor using reinforcement learning (RL) and attack detection mechanisms. The system is designed to maintain stable motor operation even under sensor attacks while ensuring safety constraints are met.

## Key Features
- Neural network adaptive observer for state estimation and attack detection
- Deep Deterministic Policy Gradient (DDPG)  and Soft Actor-Critic (SAC) controller for robust motor control
- Real-time attack detection and recovery mechanisms


## Directory Structure
```
project_root/
├── DDPG/
│   ├── observer.py        # RL-based adaptive observer implementation
│   ├── training_main.py   # Main training script for DDPG
│   ├── main.py           # Main testing script for DDPG implementation
│   ├── test_multiple_trajectories.py # Testing utilities for different trajectories
│   ├── train.py          # Training utilities for DDPG
│   ├── environment.py    # Stepper motor environment simulation
│   ├── agent.py         # DDPG agent implementation
│   └── utils.py         # Helper functions and utilities
│
├── SAC/
│   ├── observer.py      # Observer implementation for SAC
│   ├── training_main.py # Main training script for SAC
│   ├── train.py        # Training utilities for SAC
│   ├── environment.py  # Environment configuration for SAC
│   ├── agent.py       # SAC agent implementation
│   └── utils.py       # Helper functions and utilities
```

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Gymnasium


## Usage

### Using SAC Implementation
1. Navigate to SAC directory:
```bash
cd SAC
```
2. Train and Test the system:
```bash
python training_main.py
```


### Using DDPG Implementation
1. Navigate to DDPG directory:
```bash
cd DDPG
```
2. Train the system:
```bash
python training_main.py
```
3. Testing the system:
```bash
python main.py
```


Note: Each implementation (SAC and DDPG) has its own configurations and may require different hyperparameters for optimal performance.


## Key Components

### Observer Network
- Neural network-based state observer
- Real-time attack detection capabilities
- Residual-based monitoring
- Adaptive state estimation

### RL-Based Controller 
- DDPG and SAC implementation
- Automatic entropy tuning
- Experience replay buffer
- State representation handling

### Environment
- Simulated stepper motor dynamics
- Configurable attack scenarios
- Safety constraint monitoring
- Multiple reference trajectory types

## Performance Metrics
The system tracks several key performance indicators:
- Attack detection accuracy
- Recovery time
- Control performance



## References
Environment development and other parts of code are adapted from:

[1] Zhang, L., Liu, M., & Kong, F. (2023). Demo: Simulation and Security Toolbox for Cyber-Physical Systems. In Proceedings - 29th IEEE Real-Time and Embedded Technology and Applications Symposium, RTAS 2023 (pp. 357-358). (Proceedings of the IEEE Real-Time and Embedded Technology and Applications Symposium, RTAS; Vol. 2023-May). Institute of Electrical and Electronics Engineers Inc.. https://doi.org/10.1109/RTAS58335.2023.00040

GitHub Link: https://github.com/lion-zhang/CPSim

## Acknowledgment

We would like to acknowledge the use of ChatGPT for its support in debugging the code and generating several utility functions for this project. 



