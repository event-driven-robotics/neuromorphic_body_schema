import math
import time
import mujoco
from mujoco import viewer
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving images
import matplotlib.pyplot as plt
import argparse
# Constants
MODEL_PATH = "/home/fferrari-iit.local/Code/neuromorphic_body_schema/neuromorphic_body_schema/models/icub_v2_full_body.xml"
MUJOCO_TIMESTEP = 0.005  # MuJoCo internal timestep
DATA_DIR = "/home/fferrari-iit.local/Code/neuromorphic_body_schema/Data/neck/data_sinusoidal"
PLOTS_DIR = "/home/fferrari-iit.local/Code/neuromorphic_body_schema/Data/neck/plots_sinusoidal"

def set_stable_pose(model, data):
    """Set the robot to a stable standing pose - ONLY set velocities to zero, don't change positions."""
    data.qvel[:] = 0.0

def create_trajectory_plot(data_points, target_velocity_deg_s):
    """Create and save trajectory plots for position, velocity, and acceleration."""
    
    # Extract data
    times = [dp[0] for dp in data_points]
    positions = [dp[1] for dp in data_points]
    velocities = [dp[2] for dp in data_points]
    accelerations = [dp[3] for dp in data_points]
    
    # Setup plots with optimizations
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(3, 1, layout='constrained', figsize=(10, 8))
    
    # Plot data
    ax[0].plot(times, positions, 'r-.', linewidth=2, label='Position')
    ax[1].plot(times, velocities, 'b-.', linewidth=2, label='Velocity')
    ax[2].plot(times, accelerations, 'g-.', linewidth=2, label='Acceleration')
    
    # Set titles and labels
    ax[0].set_title(f'Right Elbow Joint Position (Target: {target_velocity_deg_s:.1f}°/s)')
    ax[1].set_title('Right Elbow Joint Velocity')
    ax[2].set_title('Right Elbow Joint Acceleration')
    ax[0].set_xlabel('Time (s)')
    ax[1].set_xlabel('Time (s)')
    ax[2].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position (deg)')
    ax[1].set_ylabel('Velocity (deg/s)')
    ax[2].set_ylabel('Acceleration (deg/s²)')
    
    # Add grids
    for a in ax:
        a.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_filename = f"trajectory_{target_velocity_deg_s:.1f}.png"
    plot_filepath = os.path.join(PLOTS_DIR, plot_filename)
    
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    return plot_filepath

def run_simulation(target_velocity_deg_s, num_cycles=5, motion_type='linear'):
    """Run simulation for a specific target velocity and save data."""
    
    print(f"Running simulation with target velocity: {target_velocity_deg_s}°/s")
    
    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # Set MuJoCo timestep
    model.opt.timestep = MUJOCO_TIMESTEP
    
    # Find the right elbow joint
    try:
        neck_id = model.joint("neck_yaw").id
    except:
        print("Error: Could not find 'neck' joint in model")
        return
    
    # Check if there's an actuator for the right elbow
    neck_actuator_id = None
    for i in range(model.nu):
        actuator_name = model.actuator(i).name if hasattr(model.actuator(i), 'name') else f"actuator_{i}"
        if "neck_yaw" in actuator_name.lower():
            neck_actuator_id = i
            print(f"Found right elbow actuator: {actuator_name}")
            break

    # Get joint limits
    joint_range = model.jnt_range[neck_id]
    min_angle = joint_range[0] + np.deg2rad(1)  # Minimum angle in radians with safety margin
    max_angle = joint_range[1] - np.deg2rad(1)  # Maximum angle in radians with safety margin
    
    print(f"  Joint limits: {math.degrees(min_angle):.2f}° to {math.degrees(max_angle):.2f}°")
    
    # Initialize simulation state
    current_angle = min_angle
    angle_range = max_angle - min_angle
    target_velocity_rad_s = math.radians(target_velocity_deg_s)
    direction = 1  # 1 for increasing, -1 for decreasing
    step_count = 0
    cycle_count = 0
    T = (angle_range * math.pi) / (2 * target_velocity_rad_s)
    t = 0.0
    
    # Initialize to default neutral pose first
    mujoco.mj_resetData(model, data)
    
    # Set initial neck position to 5 degrees
    data.qpos[neck_id] = current_angle
    
    # Initialize MuJoCo data properly
    mujoco.mj_forward(model, data)
    
    # Data storage
    data_points = []
    
    # Initial settling phase
    print("  Settling robot in initial pose...")
    for i in range(500):
        set_stable_pose(model, data)
        data.qpos[neck_id] = current_angle
        mujoco.mj_step(model, data)
    if motion_type == 'sinusoidal':
        phase = 0.0  # Phase for sinusoidal motion
        prev_phase = 0.0
        
    print(f"  Starting motion for {num_cycles} cycles...")
    
    # Main simulation loop
    while cycle_count < num_cycles:
        if motion_type == 'sinusoidal':
            prev_phase = phase
            phase += math.pi * MUJOCO_TIMESTEP / T
            if phase > 2 * math.pi:
                phase -= 2 * math.pi
            next_angle= min_angle + angle_range * (1 - math.cos(phase)) / 2
            if prev_phase > phase:
                cycle_count += 1
                print(f"    Completed cycle {cycle_count}/{num_cycles}")

            current_angle = next_angle
            
        else:
            # Calculate motion
            angle_increment = direction * target_velocity_rad_s * MUJOCO_TIMESTEP
            next_angle = current_angle + angle_increment
            # Check boundaries and count cycles
            if next_angle >= max_angle:
                next_angle = max_angle
                if direction == 1:  # Was going up, now reverse
                    direction = -1
                    print(f"    Cycle {cycle_count + 1}: Reached max angle {math.degrees(max_angle):.2f}°")
            elif next_angle <= min_angle:
                next_angle = min_angle
                if direction == -1:  # Was going down, now reverse and count cycle
                    direction = 1
                    cycle_count += 1
                    print(f"    Completed cycle {cycle_count}/{num_cycles}")
            
            current_angle = next_angle
        
        # Control the joint
        if neck_actuator_id is not None:
            data.ctrl[neck_actuator_id] = current_angle
        else:
            data.qpos[neck_id] = current_angle
        
        # Step the simulation
        mujoco.mj_step(model, data)
        step_count += 1
        
        # Record data every 10 steps (50ms intervals)
        if step_count % 10 == 0:
            actual_angle = data.qpos[neck_id]
            actual_velocity = data.qvel[neck_id]
            actual_acceleration = data.qacc[neck_id]
            simulation_time = step_count * MUJOCO_TIMESTEP
            
            data_points.append([
                simulation_time,
                math.degrees(actual_angle),
                math.degrees(actual_velocity),
                math.degrees(actual_acceleration)
            ])
    
    # Save data to file
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"run_{target_velocity_deg_s:.1f}.txt"
    filepath = os.path.join(DATA_DIR, filename)
    
    with open(filepath, 'w') as f:
        f.write("# Time(s) Position(deg) Velocity(deg/s) Acceleration(deg/s²)\n")
        
        for data_point in data_points:
            f.write(f"{data_point[0]:.6f} {data_point[1]:.6f} {data_point[2]:.6f} {data_point[3]:.6f}\n")
    
    # Create and save trajectory plot
    plot_filepath = create_trajectory_plot(data_points, target_velocity_deg_s)
    
    print(f"  Data saved to: {filepath}")
    print(f"  Plot saved to: {plot_filepath}")
    print(f"  Total data points: {len(data_points)}")
    print(f"  Simulation time: {data_points[-1][0]:.2f}s")
    
    return True

def main():
    """Main function to run simulations with different velocities."""
    parser = argparse.ArgumentParser(description="iCub neck joint motion simulation")
    parser.add_argument('--motion', choices=['sinusoidal', 'linear'], default='linear', help="Type of motion: sinusoidal or linear")
    args = parser.parse_args()
    print("Starting elbow motion data generation...")
    print(f"Data will be saved to: {DATA_DIR}")
    
    # Generate 20 velocities on logarithmic scale from 1 to 60 deg/s
    min_velocity = 1
    max_velocity = 60
    num_simulations = 20
    
    # Create logarithmic scale
    log_min = np.log10(min_velocity)
    log_max = np.log10(max_velocity)
    log_velocities = np.linspace(log_min, log_max, num_simulations)
    velocities = 10**log_velocities
    
    print(f"\nVelocities to test ({num_simulations} simulations):")
    for i, vel in enumerate(velocities):
        print(f"  {i+1:2d}: {vel:.2f} deg/s")
    
    print(f"\nStarting simulations...")
    
    successful_runs = 0
    failed_runs = 0
    
    for i, velocity in enumerate(velocities):
        print(f"\n--- Simulation {i+1}/{num_simulations} ---")
        try:
            success = run_simulation(velocity, num_cycles=5, motion_type=args.motion)
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
        except Exception as e:
            print(f"  ERROR: Simulation failed with error: {e}")
            failed_runs += 1
    
    print(f"\n=== Summary ===")
    print(f"Total simulations: {num_simulations}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Data saved to: {DATA_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    
    if successful_runs > 0:
        print(f"\nFiles created:")
        for velocity in velocities[:successful_runs]:
            data_filename = f"run_{velocity:.1f}.txt"
            plot_filename = f"trajectory_{velocity:.1f}.png"
            print(f"  {data_filename} + {plot_filename}")

if __name__ == "__main__":
    main()
