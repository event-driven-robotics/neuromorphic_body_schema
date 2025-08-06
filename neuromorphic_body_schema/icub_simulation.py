import math
import time
import mujoco
from mujoco import viewer
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import matplotlib.pyplot as plt
from collections import deque

# Constants
MODEL_PATH = "/home/fferrari-iit.local/Code/neuromorphic_body_schema/neuromorphic_body_schema/models/icub_v2_full_body.xml"
TARGET_VELOCITY_DEG_S = 5.0  # Target velocity in degrees per second
RENDER_INTERVAL = 0.05  # 50ms between renders
MUJOCO_TIMESTEP = 0.005  # MuJoCo internal timestep
STEPS_PER_RENDER = int(RENDER_INTERVAL / MUJOCO_TIMESTEP)  # 10 steps per render

def set_stable_pose(model, data):
    """Set the robot to a stable standing pose - ONLY set velocities to zero, don't change positions."""
    data.qvel[:] = 0.0

class EfficientPlotter:
    """Efficient plotting class that minimizes matplotlib overhead."""
    
    def __init__(self, max_points=500):
        self.max_points = max_points
        
        # Use deques for efficient data management
        self.times = deque(maxlen=max_points)
        self.positions = deque(maxlen=max_points)
        self.velocities = deque(maxlen=max_points)
        self.accelerations = deque(maxlen=max_points)
        
        # Setup plots with optimizations
        plt.ion()
        plt.style.use('fivethirtyeight')
        self.fig, self.ax = plt.subplots(3, 1, layout='constrained', figsize=(10, 8))
        
        # Initialize empty line objects
        self.pos_line, = self.ax[0].plot([], [], 'r-.', linewidth=2)
        self.vel_line, = self.ax[1].plot([], [], 'b-.', linewidth=2)
        self.acc_line, = self.ax[2].plot([], [], 'g-.', linewidth=2)
        
        # Set titles and labels
        self.ax[0].set_title('Right Elbow Joint Position')
        self.ax[1].set_title('Right Elbow Joint Velocity')
        self.ax[2].set_title('Right Elbow Joint Acceleration')
        self.ax[0].set_xlabel('Time (s)')
        self.ax[1].set_xlabel('Time (s)')
        self.ax[2].set_xlabel('Time (s)')
        self.ax[0].set_ylabel('Position (deg)')
        self.ax[1].set_ylabel('Velocity (deg/s)')
        self.ax[2].set_ylabel('Acceleration (deg/s²)')
        
        # Enable blitting for faster updates
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        
        plt.show(block=False)
        
        # Track update timing
        self.last_update = time.time()
        self.update_interval = 0.4  #
        
    def add_data(self, t, pos, vel, acc):
        """Add new data point."""
        self.times.append(t)
        self.positions.append(pos)
        self.velocities.append(vel)
        self.accelerations.append(acc)
        
    def update_if_needed(self):
        """Update plots only if enough time has passed."""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval and len(self.times) > 0:
            self.update_plots()
            self.last_update = current_time
            return True
        return False
    
    def update_plots(self):
        """Efficiently update the plot lines."""
        if len(self.times) == 0:
            return
            
        try:
            # Convert deques to arrays for plotting
            times_array = np.array(self.times)
            pos_array = np.array(self.positions)
            vel_array = np.array(self.velocities)
            acc_array = np.array(self.accelerations)
            
            # Restore background
            self.fig.canvas.restore_region(self.background)
            
            # Update line data
            self.pos_line.set_data(times_array, pos_array)
            self.vel_line.set_data(times_array, vel_array)
            self.acc_line.set_data(times_array, acc_array)
            
            # Auto-scale axes only when needed
            for ax in self.ax:
                ax.relim()
                ax.autoscale_view()

            
            plt.draw()
            plt.pause(0.000000001)  # Use a very small pause to allow GUI updates
            
        except Exception as e:
            print(f"Plot update error: {e}")
            # Fallback to normal drawing
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except:
                pass

def main():
    """Main simulation loop."""
    
    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # Set MuJoCo timestep
    model.opt.timestep = MUJOCO_TIMESTEP
    
    # Find the right elbow joint
    try:
        r_elbow_id = model.joint("r_elbow").id
    except:
        print("Error: Could not find 'r_elbow' joint in model")
        return
    
    # Check if there's an actuator for the right elbow
    r_elbow_actuator_id = None
    for i in range(model.nu):
        actuator_name = model.actuator(i).name if hasattr(model.actuator(i), 'name') else f"actuator_{i}"
        if "r_elbow" in actuator_name.lower():
            r_elbow_actuator_id = i
            print(f"Found right elbow actuator: {actuator_name}")
            break
    
    if r_elbow_actuator_id is None:
        print("No right elbow actuator found, will use direct position control with smoothing")
    
    # Get joint limits
    joint_range = model.jnt_range[r_elbow_id]
    min_angle = joint_range[0] + np.deg2rad(1)  # Minimum angle in radians
    max_angle = joint_range[1] - np.deg2rad(1)  # Maximum angle in radians
    
    print(f"Right elbow joint limits:")
    print(f"  Min: {math.degrees(min_angle):.2f}°")
    print(f"  Max: {math.degrees(max_angle):.2f}°")
    print(f"  Range: {math.degrees(max_angle - min_angle):.2f}°")
    print(f"MuJoCo timestep: {MUJOCO_TIMESTEP}s")
    print(f"Render interval: {RENDER_INTERVAL}s ({STEPS_PER_RENDER} MuJoCo steps)")
    print(f"Target velocity: {TARGET_VELOCITY_DEG_S}°/s")
    
    # Initialize simulation state
    current_angle = min_angle
    target_velocity_rad_s = math.radians(TARGET_VELOCITY_DEG_S)
    direction = 1  # 1 for increasing, -1 for decreasing
    step_count = 0
    render_count = 0
    
    # Initialize to default neutral pose first
    mujoco.mj_resetData(model, data)
    
    # Set initial elbow position
    data.qpos[r_elbow_id] = current_angle
    
    # Initialize MuJoCo data properly
    mujoco.mj_forward(model, data)  # Forward kinematics to ensure consistency
    
    # Setup efficient plotting
    plotter = EfficientPlotter(max_points=500)
    

    # Create viewer
    with viewer.launch_passive(model, data) as viewer_instance:
        
        for i in range(500):  # More steps for better settling
            set_stable_pose(model, data)
            mujoco.mj_step(model, data)
            if i % 50 == 0:  # Update viewer every 50 steps
                viewer_instance.sync()
                time.sleep(0.01)
        
        # Reset counters after initialization
        step_count = 0
        render_count = 0
        
        while viewer_instance.is_running():
            
            angle_increment = direction * target_velocity_rad_s * MUJOCO_TIMESTEP
            next_angle = current_angle + angle_increment
            
            # Check boundaries and reverse direction if needed
            if next_angle >= max_angle:
                next_angle = max_angle
                direction = -1
            elif next_angle <= min_angle:
                next_angle = min_angle
                direction = 1
                
            current_angle = next_angle
            
            # STABLE CONTROL: Use actuator if available, otherwise VERY careful position control
            if r_elbow_actuator_id is not None:
                data.ctrl[r_elbow_actuator_id] = current_angle

            else:
                data.qpos[r_elbow_id] = current_angle
            
            mujoco.mj_step(model, data)
            step_count += 1
            
            # Render every RENDER_INTERVAL
            if step_count % STEPS_PER_RENDER == 0:
                actual_angle = data.qpos[r_elbow_id]
                actual_velocity = data.qvel[r_elbow_id]
                actual_acceleration = data.qacc[r_elbow_id]
                simulation_time = step_count * MUJOCO_TIMESTEP

                print(f"Time: {simulation_time:.3f}s, "
                      f"Position: {math.degrees(actual_angle):.2f}°, "
                      f"Velocity: {math.degrees(actual_velocity):.2f}°/s, "
                      f"Acceleration: {math.degrees(actual_acceleration):.2f}°/s²")

                viewer_instance.sync()
                render_count += 1
           
                actual_angle = data.qpos[r_elbow_id]
                actual_velocity = data.qvel[r_elbow_id]
                actual_acceleration = data.qacc[r_elbow_id]
                simulation_time = step_count * MUJOCO_TIMESTEP
                
                # Add data to plotter
                plotter.add_data(
                    simulation_time,
                    math.degrees(actual_angle),
                    math.degrees(actual_velocity),
                    math.degrees(actual_acceleration)
                )
                
                # Update plots only when needed (every 400ms)
                plotter.update_if_needed()
        
        print("Simulation ended.")

if __name__ == "__main__":
    main()
