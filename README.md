# CSCE-642-fall-2024-
Reinforcement Learning Course Project

# Deep Reinforcement Learning Based Attack Mitigation and Recovery controller

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

### SAC Controller
- Soft Actor-Critic implementation
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
- Safety constraint satisfaction
- False positive/negative rates

## Visualization
The project includes comprehensive visualization tools for:
- Training progress
- Control performance
- Attack detection results
- System state trajectories



