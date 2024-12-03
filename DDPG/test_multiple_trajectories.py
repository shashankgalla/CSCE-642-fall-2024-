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

def calculate_detection_metrics(states, attack_start, attack_duration):
    """Calculate detection and recovery metrics"""
    detection_flags = states[:, 3]  # Attack detection flags
    position_errors = np.abs(states[:, 0] - states[:, 1])  # Position error
    
    # Initialize metrics
    detection_time = None
    recovery_start_time = None
    recovery_time = None
    false_positives = 0
    false_negatives = 0
    
    # Calculate detection time
    for t in range(attack_start, attack_start + attack_duration):
        if detection_flags[t]:
            detection_time = t
            break
    
    # Calculate recovery metrics
    if detection_time is not None:
        # Find recovery start (when position error starts decreasing consistently)
        for t in range(detection_time, attack_start + attack_duration):
            if t > 0 and position_errors[t] < position_errors[t-1]:
                recovery_start_time = t
                break
        
        # Calculate recovery time (when error becomes small enough)
        if recovery_start_time is not None:
            error_threshold = 0.1  # Define acceptable error threshold
            for t in range(recovery_start_time, len(position_errors)):
                if position_errors[t] < error_threshold:
                    recovery_time = t - recovery_start_time
                    break
    
    # Calculate false positives (before attack and after recovery)
    false_positives = np.sum(detection_flags[:attack_start]) + \
                     np.sum(detection_flags[attack_start + attack_duration:])
    
    # Calculate false negatives (missed detections during attack)
    if detection_time is not None:
        false_negatives = np.sum(1 - detection_flags[attack_start:detection_time])
    
    return {
        'detection_time': detection_time,
        'recovery_start_time': recovery_start_time,
        'recovery_time': recovery_time,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'detection_accuracy': 1 - (false_positives + false_negatives) / len(states)
    }

def plot_results(time, scenarios, attack_start, attack_duration):
    """Plot all results with scenario-specific safe and target sets"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'legend.fontsize': 16,
        'figure.figsize': (18, 22),
        'lines.linewidth': 2
    })
    
    total_scenarios = len(scenarios)
    scenario_names = list(scenarios.keys())
    num_plots = total_scenarios * 2  # Each scenario has a position and control plot
    num_rows = total_scenarios
    num_cols = 2  # Position and control plots side by side
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 3))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    
    labels = ['Physical Position', 'Sensor Reading', 'Adaptive Observer Estimate']
    colors = ['red', 'forestgreen', 'blue']
    
    dt = time[1] - time[0]
    attack_start_time = attack_start * dt
    attack_duration_time = attack_duration * dt
    
    trajectory_legend_handles = []
    region_legend_handles = []
    
    for idx, scenario_name in enumerate(scenario_names):
        row = idx
        scenario = scenarios[scenario_name]
        
        # Determine if the scenario is an attack scenario
        is_attack_scenario = 'attack' in scenario_name
        
        # Get trajectory type for title
        trajectory_type = scenario_name.split('_')[0].capitalize()
        attack_status = 'With Attack' if is_attack_scenario else 'No Attack'
        title = f"{trajectory_type} Reference - {attack_status}"
        
        # Position plot
        ax_pos = axes[row, 0]
        scenario_time = scenario['time']
        
        # Plot safe set
        safe_bounds = scenario['safe_bounds']
        ax_pos.fill_between(
            scenario_time, safe_bounds[:, 0], safe_bounds[:, 1], 
            color='yellow', alpha=0.1, label='Safe Set'
        )
        
        # Plot target set
        target_bounds = scenario['target_bounds']
        ax_pos.fill_between(
            scenario_time, target_bounds[:, 0], target_bounds[:, 1], 
            color='lightblue', alpha=0.1, label='Target Set'
        )
        
        if is_attack_scenario:
            # Adjust attack period to scenario time
            attack_end_time = attack_start_time + attack_duration_time
            ax_pos.axvspan(
                attack_start_time, attack_end_time, 
                facecolor='lightcoral', alpha=0.15, zorder=3, label='Attack Period'
            )
            
            # Add detection and recovery markers if available
            if 'detection_time' in scenario and scenario['detection_time'] is not None:
                detect_time = scenario['detection_time'] * dt
                ax_pos.axvline(x=detect_time, color='purple', linestyle='--', 
                             label='Attack Detection', alpha=0.8)
            
            if 'recovery_start_time' in scenario and scenario['recovery_start_time'] is not None:
                recovery_time = scenario['recovery_start_time'] * dt
                ax_pos.axvline(x=recovery_time, color='green', linestyle='--', 
                             label='Recovery Start', alpha=0.8)
        
        # Plot positions
        lines = []
        ref_line, = ax_pos.plot(
            scenario_time, scenario['references'], 'k--', 
            label='Reference', linewidth=1.5
        )
        lines.append(ref_line)
        
        for i in range(3):
            line, = ax_pos.plot(
                scenario_time, scenario['states'][:, i], 
                color=colors[i], label=labels[i], linewidth=1.5
            )
            lines.append(line)
        
        if idx == 0:
            trajectory_legend_handles = lines
        
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Position (rad)')
        ax_pos.set_title(title)
        ax_pos.grid(True, linestyle='--', alpha=0.7)
        ax_pos.set_ylim([-np.pi, np.pi])
        ax_pos.set_xlim([0, scenario_time[-1]])
        
        # Control plot
        ax_ctrl = axes[row, 1]
        plot_control(ax_ctrl, scenario_time, scenario['controls'])
        
        if is_attack_scenario:
            ax_ctrl.axvspan(
                attack_start_time, attack_end_time, 
                facecolor='lightcoral', alpha=0.15, label='Attack Period'
            )
            
            # Add detection and recovery markers to control plot
            if 'detection_time' in scenario and scenario['detection_time'] is not None:
                detect_time = scenario['detection_time'] * dt
                ax_ctrl.axvline(x=detect_time, color='purple', linestyle='--', alpha=0.8)
            
            if 'recovery_start_time' in scenario and scenario['recovery_start_time'] is not None:
                recovery_time = scenario['recovery_start_time'] * dt
                ax_ctrl.axvline(x=recovery_time, color='green', linestyle='--', alpha=0.8)
        
        ax_ctrl.set_xlim([0, scenario_time[-1]])
    
    # Create region legend handles
    region_legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor='yellow', alpha=0.2, label='Safe Set'),
        Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.2, label='Target Set'),
        Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.2, label='Attack Period'),
        Line2D([0], [0], color='purple', linestyle='--', label='Attack Detection'),
        Line2D([0], [0], color='green', linestyle='--', label='Recovery Start')
    ]
    
    # Add legends
    handles_labels = ax_pos.get_legend_handles_labels()
    fig.legend(handles=trajectory_legend_handles + region_legend_handles, 
               loc='upper center', ncol=6, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Stepper Motor Control Scenarios', fontsize=20)
    plt.savefig('stepper_motor_results_all_scenarios.pdf', bbox_inches='tight', dpi=300)
    plt.show()


def plot_control(ax, scenario_time, controls):
    """Plot control inputs"""
    ax.plot(scenario_time, controls, 'k', label='Control Action', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Input (V)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim([-25, 25])

def test_controller(env, controller, simulation_steps=500):
    """Test the trained RL controller with various scenarios"""
    dt = env.dt
    time = np.arange(0, simulation_steps * dt, dt)
    
    # Attack parameters
    attack_start = 150
    attack_duration = 100
    attack_magnitude = np.pi/4
    
    # Initialize data storage
    scenario_names = ['step_no', 'step_attack',
                      'sine_no', 'sine_attack',
                      'square_no', 'square_attack',
                      'ramp_no', 'ramp_attack',
                      'constant_no', 'constant_attack']
    scenarios = {}
    for name in scenario_names:
        scenarios[name] = {'states': np.zeros((simulation_steps, 3)), 
                          'controls': np.zeros(simulation_steps),
                          'references': np.zeros(simulation_steps),
                          'safe_bounds': np.zeros((simulation_steps, 2)),
                          'target_bounds': np.zeros((simulation_steps, 2)),
                          'time': time.copy()}

    def generate_reference(t, reference_type='step'):
        """Generate reference trajectory based on type."""
        if reference_type == 'step':
            # Existing step reference code
            step_duration = 100
            if t < step_duration:
                return 0.0
            elif t < 2 * step_duration:
                return np.pi/6  # Reduced amplitude for stability
            elif t < 3 * step_duration:
                return -np.pi/6
            elif t < 4 * step_duration:
                return np.pi/8
            else:
                return 0.0
        elif reference_type == 'sine':
            amplitude = np.pi / 6  # Adjust amplitude as needed
            frequency = 0.05  # Adjust frequency as needed
            return amplitude * np.sin(2 * np.pi * frequency * t * dt)
        elif reference_type == 'square':
            period = 100  # Adjust period as needed
            return (np.pi / 6) * (1 if (t // period) % 2 == 0 else -1)
        elif reference_type == 'ramp':
            slope = (np.pi / 2) / simulation_steps
            return slope * t
        else:  # default to zero reference
            return 0.0

    class SafeEnvWrapper:
        """Wrapper to prevent early termination during testing"""
        def __init__(self, env):
            self.__dict__.update(env.__dict__)
            self.env = env
        
        def reset(self):
            # Reset with controlled initial state
            self.env.state = np.zeros(5)  # Start at zero
            self.env.prev_velocity = 0.0
            self.env.prev_error = 0.0
            self.env.time_step = 0
            self.env.under_attack = False
            self.env.attack_type = None
            self.env.attack_magnitude = 0.0
            
            if hasattr(self.env, 'adaptive_observer'):
                self.env.adaptive_observer.reset()
            
            return self.env.state, {}
        
        def step(self, action):
            next_state, reward, done, trunc, info = self.env.step(action)
            # Ignore done signal during testing
            return next_state, reward, False, trunc, info
                
        def __getattr__(self, name):
            return getattr(self.env, name)
    
    # Wrap the environment
    safe_env = SafeEnvWrapper(env)
    
    def run_scenario(scenario_name, reference_type='step', attack=False):
        print(f"\nRunning scenario: {scenario_name}")
        
        # Reset environment with safe initial conditions
        state, _ = safe_env.reset()
        controller.prev_error = 0.0  # Reset previous error
        prev_action = 0
        scenario = scenarios[scenario_name]
        
        # Store full state history including detection flag
        full_states = np.zeros((simulation_steps, 5))  # Include detection flag and attack estimate

        # Warmup period
        for t_warmup in range(50):
            ref = 0.0
            safe_env.target_position = ref
            state_rep = controller.get_state_representation(
                sensor_position=state[1],
                attack_estimate=state[4],
                reference=ref,
                attack_detected=bool(state[3]),
                prev_action=prev_action
            )
            with torch.no_grad():
                action = controller.select_action(state_rep, evaluate=True)
            action = np.clip(action, -safe_env.max_voltage, safe_env.max_voltage)
            state, _, _, _, _ = safe_env.step(action)
            prev_action = float(action[0])
        
        for t in range(simulation_steps):
            current_time = t * dt
            
            # Generate reference
            ref = generate_reference(t, reference_type)
            safe_env.target_position = ref
            
            # Handle attack
            if attack and t == attack_start:
                print(f"Starting attack at t={current_time:.2f}s")
                attack_sign = 1 if ref <= 0 else -1
                safe_env.start_attack("bias", attack_sign * attack_magnitude)
            elif attack and t == (attack_start + attack_duration):
                print(f"Stopping attack at t={current_time:.2f}s")
                safe_env.stop_attack()
            
            # Get control action
            state_rep = controller.get_state_representation(
                sensor_position=state[1],
                attack_estimate=state[4],
                reference=ref,
                attack_detected=bool(state[3]),
                prev_action=prev_action
            )
            
            with torch.no_grad():
                action = controller.select_action(state_rep, evaluate=True)
            action = np.clip(action, -safe_env.max_voltage, safe_env.max_voltage)
            prev_action = float(action[0])
            
            # Step environment
            next_state, _, _, _, info = safe_env.step(action)
            
            # Store results
            scenario['states'][t] = [
                next_state[0],  # Physical position
                next_state[1],  # Sensor reading
                next_state[2]   # Observer estimate
            ]
            full_states[t] = next_state[:5]  # Store full state including detection flag and attack estimate
            scenario['controls'][t] = prev_action
            scenario['references'][t] = ref
            scenario['safe_bounds'][t] = [
                ref - safe_env.safe_margin,
                ref + safe_env.safe_margin
            ]
            scenario['target_bounds'][t] = [
                ref - safe_env.target_margin,
                ref + safe_env.target_margin
            ]
            
            # Debug prints
            if t % 100 == 0:
                print(f"t={current_time:.2f}s, ref={ref:.2f}, pos={next_state[0]:.2f}, "
                      f"control={prev_action:.2f}, attack_detected={bool(next_state[3])}")
            
            state = next_state
        
        # Calculate detection metrics for attack scenarios
        if attack:
            metrics = calculate_detection_metrics(full_states, attack_start, attack_duration)
            scenario.update({
                'detection_time': metrics['detection_time'],
                'recovery_start_time': metrics['recovery_start_time'],
                'recovery_time': metrics['recovery_time'],
                'false_positives': metrics['false_positives'],
                'false_negatives': metrics['false_negatives'],
                'detection_accuracy': metrics['detection_accuracy']
            })
            
            # Print detection metrics with None checks
            print("\nDetection Metrics:")
            if metrics['detection_time'] is not None:
                print(f"Detection Time: {metrics['detection_time'] * dt:.2f}s")
            else:
                print("Detection Time: None (Attack not detected)")
                
            if metrics['recovery_start_time'] is not None:
                print(f"Recovery Start Time: {metrics['recovery_start_time'] * dt:.2f}s")
            else:
                print("Recovery Start Time: None (Recovery not started)")
                
            if metrics['recovery_time'] is not None:
                print(f"Recovery Duration: {metrics['recovery_time'] * dt:.2f}s")
            else:
                print("Recovery Duration: None (Recovery not completed)")
                
            print(f"False Positives: {metrics['false_positives']}")
            print(f"False Negatives: {metrics['false_negatives']}")
            print(f"Detection Accuracy: {metrics['detection_accuracy']:.2%}")
    
    # Run scenarios
    print("Starting scenario testing...")
    torch.manual_seed(0)
    np.random.seed(0)
    
    run_scenario('step_no', 'step', False)
    run_scenario('step_attack', 'step', True)
    run_scenario('sine_no', 'sine', False)
    run_scenario('sine_attack', 'sine', True)
    run_scenario('square_no', 'square', False)
    run_scenario('square_attack', 'square', True)
    run_scenario('ramp_no', 'ramp', False)
    run_scenario('ramp_attack', 'ramp', True)
    run_scenario('constant_no', 'constant', False)
    run_scenario('constant_attack', 'constant', True)
    
    print("\nPlotting results...")
    plot_results(time, scenarios, attack_start, attack_duration)

