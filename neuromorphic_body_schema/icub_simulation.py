import math
import time
import mujoco
from mujoco import viewer
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import matplotlib.pyplot as plt

# Constants
MODEL_PATH = "/home/fferrari-iit.local/Code/neuromorphic_body_schema/neuromorphic_body_schema/models/icub_v2_full_body.xml"
TARGET_VELOCITY_DEG_S = 5.0  # Target velocity in degrees per second
RENDER_INTERVAL = 0.05  # 50ms between renders
MUJOCO_TIMESTEP = 0.005  # MuJoCo internal timestep
STEPS_PER_RENDER = int(RENDER_INTERVAL / MUJOCO_TIMESTEP)  # 10 steps per render

def set_stable_pose(model, data):
    """Set the robot to a stable standing pose - ONLY set velocities to zero, don't change positions."""
    data.qvel[:] = 0.0

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
    
    t=[]
    pos=[]
    vel=[]
    acc=[]
    
    plt.ion() 
    plt.style.use('fivethirtyeight')
    fig,ax=plt.subplots(3,1,layout='constrained',figsize=(10, 8))
    pos_graph=ax[0].plot([],[],'r-')
    vel_graph=ax[1].plot([],[],'b-')
    acc_graph=ax[2].plot([],[],'g-')
    ax[0].set_title('Right Elbow Joint Position')
    ax[1].set_title('Right Elbow Joint Velocity')
    ax[2].set_title('Right Elbow Joint Acceleration')
    ax[0].set_xlabel('Time (s)')
    ax[1].set_xlabel('Time (s)')
    ax[2].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position (deg)')
    ax[1].set_ylabel('Velocity (deg/s)')
    ax[2].set_ylabel('Acceleration (deg/s²)')
    plt.show()

  

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
                
                t.append(simulation_time)
                pos.append(math.degrees(actual_angle))
                vel.append(math.degrees(actual_velocity))
                acc.append(math.degrees(actual_acceleration))
                if len(t)>500:
                    t, pos, vel, acc = t[-500:], pos[-500:], vel[-500:], acc[-500:]
                pos_graph[0].set_xdata(np.arange(len(t)))
                pos_graph[0].set_ydata(pos)
                vel_graph[0].set_xdata(np.arange(len(t)))
                vel_graph[0].set_ydata(vel)
                acc_graph[0].set_xdata(np.arange(len(t)))
                acc_graph[0].set_ydata(acc)
                
                ax[0].relim()
                ax[0].autoscale_view()
                ax[1].relim()
                ax[1].autoscale_view()
                ax[2].relim()
                ax[2].autoscale_view()
                plt.draw()
                plt.pause(0.000000001)

if __name__ == "__main__":
    main()
