# CSCE-642-fall-2024-
Reinforcement Learning Course Project

# Deep Reinforcement Learning Based Attack Mitigation and Recovery controller



## Project Overview
This project implements a secure control system for a stepper motor using reinforcement learning (RL) and attack detection mechanisms. The system is designed to maintain stable motor operation even under sensor attacks while ensuring safety constraints are met.

## Key Features
- Neural network adaptive observer for state estimation and attack detection
- Soft Actor-Critic (SAC) controller for robust motor control
- Real-time attack detection and recovery mechanisms


## File Structure
- `observer.py`: Implementation of the RL-based adaptive observer
- `training_main.py`: Main training and testing script for the system (main file)
- `agent.py`: SAC agent implementation for motor control
- `train.py`: Training utilities and functions
- `utils.py`: Helper functions and utilities
- `environment.py`: Stepper motor environment simulation

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Gymnasium


## Usage
1. Run the system:
```bash
python training_main.py
```


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



