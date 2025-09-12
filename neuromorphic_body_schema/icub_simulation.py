# Standard library imports
import math
import time
import threading
import argparse
from collections import deque

# Third-party imports
import mujoco
from mujoco import viewer
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import matplotlib.pyplot as plt
import tkinter as tk
# Constants
MODEL_PATH = "/home/fferrari-iit.local/Code/neuromorphic_body_schema/neuromorphic_body_schema/models/icub_v2_full_body.xml"
TARGET_VELOCITY_DEG_S = 20.0  # Target velocity in degrees per second
RENDER_INTERVAL = 0.05  # 50ms between renders
MUJOCO_TIMESTEP = 0.005  # MuJoCo internal timestep
STEPS_PER_RENDER = int(RENDER_INTERVAL / MUJOCO_TIMESTEP)  # 10 steps per render

# Global control variables for thread-safe communication
start_request = False
move_request = False
control_lock = threading.Lock()

def set_start_request(value):
    global start_request
    with control_lock:
        start_request = value

def get_start_request():
    global start_request
    with control_lock:
        return start_request

def set_move_request(value):
    global move_request
    with control_lock:
        move_request = value
        start_request= False  

def get_move_request():
    global move_request
    with control_lock:
        return move_request

def create_control_panel():
    """Create control panel that runs in a separate thread without blocking."""
    root = tk.Tk()
    root.title("Robot Control Panel")
    root.geometry("300x150")
    
    def request_start():
        set_start_request(True)
        status_label.config(text="START requested!")
        print("START button pressed!")
        
    def request_move():
        set_move_request(True)
        status_label.config(text="MOVE requested!")
        print("MOVE button pressed!")
    
    start_button = tk.Button(root, text="START", command=request_start,
                            bg="green", fg="white", height=2, width=15, font=("Arial", 12, "bold"))
    start_button.pack(pady=10)

    move_button = tk.Button(root, text="MOVE", command=request_move,
                           bg="blue", fg="white", height=2, width=15, font=("Arial", 12, "bold"))
    move_button.pack(pady=10)

    status_label = tk.Label(root, text="Control panel ready", font=("Arial", 10))
    status_label.pack(pady=5)
    
    # Make the GUI non-blocking by using after() instead of mainloop()
    def update_gui():
        try:
            root.update()  # Process pending events
            root.after(5, update_gui)  # Schedule next update in 50ms
        except tk.TclError:
            # Window was closed
            pass
    
    update_gui()  # Start the update cycle
    
  




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
        """Efficiently update the plot lines, skipping the first value."""
        if len(self.times) <= 1:
            return
        try:
            # Convert deques to arrays for plotting, skip the first value
            times_array = np.array(self.times)[1:]
            pos_array = np.array(self.positions)[1:]
            vel_array = np.array(self.velocities)[1:]
            acc_array = np.array(self.accelerations)[1:]

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
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="iCub neck joint motion simulation")
    parser.add_argument('--motion', choices=['sinusoidal', 'linear'], default='linear', help="Type of motion: sinusoidal or linear")
    args = parser.parse_args()

    # Start GUI in separate thread
    gui_thread = threading.Thread(target=create_control_panel, daemon=True)
    gui_thread.start()

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

    if neck_actuator_id is None:
        print("No right elbow actuator found, will use direct position control with smoothing")

    # Get joint limits
    joint_range = model.jnt_range[neck_id]
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
    target_velocity_rad_s = math.radians(TARGET_VELOCITY_DEG_S)
    angle_range = max_angle - min_angle
    T = (angle_range * math.pi) / (2 * target_velocity_rad_s)
    t = 0.0
    direction = 1  # 1 for forward, -1 for backward
    step_count = 0
    render_count = 0
    current_angle = min_angle

    # Initialize to default neutral pose first
    mujoco.mj_resetData(model, data)

    # Set initial elbow position to 5 degrees
    data.qpos[neck_id] = current_angle

    # Initialize MuJoCo data properly
    mujoco.mj_forward(model, data)  # Forward kinematics to ensure consistency

    # Setup efficient plotting
    plotter = EfficientPlotter(max_points=500)

    print("Simulation ready. Check control panel for START/MOVE buttons.")

    # Create viewer
    with viewer.launch_passive(model, data) as viewer_instance:
        # Initial settling
        for i in range(500):
            set_stable_pose(model, data)
            mujoco.mj_step(model, data)
            if i % 50 == 0:
                viewer_instance.sync()
                time.sleep(0.01)

        # Reset counters after initialization
        step_count = 0
        render_count = 0

        print("Robot initialized. Use control panel to start motion.")

        # For sinusoidal motion, use phase variable
        if args.motion == 'sinusoidal':
            if not hasattr(main, "phase"):
                main.phase = 0.0

        while viewer_instance.is_running():
            # Check for start/move requests from GUI
            if get_start_request():
                print("Processing START request...")
                set_start_request(False)  # Reset flag
            if get_move_request():
                print("Processing MOVE request...")
                set_move_request(False)  # Reset flag

            if args.motion == 'sinusoidal':
                # Sinusoidal (cosine) trajectory: position from min_angle to max_angle and back
                # Use a phase variable that increases continuously, and use cos(phase) to get smooth back-and-forth motion
                # q(t) = min_angle + (angle_range) * (1 - cos(phase)) / 2
                # phase goes from 0 to 2*pi for a full cycle (min->max->min)
                # The period for a full cycle is 2*T
                main.phase += math.pi * MUJOCO_TIMESTEP / T  # increment phase so that pi phase = T seconds
                if main.phase > 2 * math.pi:
                    main.phase -= 2 * math.pi
                current_angle = min_angle + angle_range * (1 - math.cos(main.phase)) / 2
            else:
                # Linear motion: increment/decrement at constant velocity, reverse at limits
                angle_increment = direction * target_velocity_rad_s * MUJOCO_TIMESTEP
                next_angle = current_angle + angle_increment
                if next_angle >= max_angle:
                    next_angle = max_angle
                    direction = -1
                    print(f"Reached max angle: {math.degrees(max_angle):.2f}°, reversing direction")
                elif next_angle <= min_angle:
                    next_angle = min_angle
                    direction = 1
                    print(f"Reached min angle: {math.degrees(min_angle):.2f}°, reversing direction")
                current_angle = next_angle

            # STABLE CONTROL: Use actuator if available, otherwise VERY careful position control
            if neck_actuator_id is not None:
                data.ctrl[neck_actuator_id] = current_angle
            else:
                data.qpos[neck_id] = current_angle

            mujoco.mj_step(model, data)
            step_count += 1

            # Render every RENDER_INTERVAL
            if step_count % STEPS_PER_RENDER == 0:
                actual_angle = data.qpos[neck_id]
                actual_velocity = data.qvel[neck_id]
                actual_acceleration = data.qacc[neck_id]
                simulation_time = step_count * MUJOCO_TIMESTEP

                print(f"Time: {simulation_time:.3f}s, "
                      f"Position: {math.degrees(actual_angle):.2f}°, "
                      f"Velocity: {math.degrees(actual_velocity):.2f}°/s, "
                      f"Acceleration: {math.degrees(actual_acceleration):.2f}°/s²")

                viewer_instance.sync()
                render_count += 1

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
