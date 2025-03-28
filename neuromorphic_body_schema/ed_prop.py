import logging

import cv2
import mujoco
import numpy as np
from proprioception import generalized_sigmoid, linear, proprioception

##############################################################################################################
TIME_WINDOW = 50000  # TODO make a global variable


def make_prop_event_frame(width, time, time_window, previous_events):
    # TODO do I need esim?
    tick_height = 20
    margin = 5
    height = 8 * (tick_height + margin)
    current_img = np.zeros((height, width, 3))
    
    t_left = time-time_window
    scale = width/time_window
        
    if len(previous_events):            
        for single_event in previous_events:
            y_val = single_event[0]*tick_height # TODO controlla that this is still not a problem of the idx 0 overriding the info in the event representation
            x_val = int((single_event[2]-t_left)*scale)
            neuron_idx = single_event[0]
            
            color_index = int((neuron_idx * 255/ 8) )
            color = cv2.applyColorMap(np.array([[color_index]], dtype=np.uint8), cv2.COLORMAP_HSV)[0][0]
            color = color/255

            cv2.line(current_img, (x_val, int(y_val)+margin), (x_val, int(y_val)+tick_height), color, 1) 

    return current_img



def visualize_proprioception(time, model, data, joint_list, previous_events, esim = None, show_proprioception=False, DEBUG = False):
    
    joint_pos = np.zeros(len(joint_list))
    joint_vel = np.zeros(len(joint_list))
    joint_load = np.zeros(len(joint_list))
    
    max_max_freq = 1e-4
    events_window_names= [] 

    # TODO replace with joint states I want to access
    for i in range(len(joint_list)):
        joint_pos[i] = data.joint(joint_list[i]).qpos # example for single joint
        joint_vel[i] = data.joint(joint_list[i]).qvel
        joint_load[i] = data.joint(joint_list[i]).qacc
        events_window_names.append(str(joint_list[i])+' proprioception events')

        

    if esim is None:
        # The values used for the next object are absolutely arbitrary, they will be changed in a later stage 
        # also based on the platform the code is going to be used on
        # max freq is 0.1 cause time is in ns
        freq = 0.0001
        
        esim = proprioception(n_joints=len(joint_list), position_limits= [ [model.actuator(joint).ctrlrange[0] for joint in joint_list ] , [model.actuator(joint).ctrlrange[1] for joint in joint_list ] ] , 
                                velocity_limit=20.0, load_limit= 10000000.0, position_max_freq=np.array(freq), velocity_max_freq=np.array(freq), 
                                load_max_freq=np.array(freq), limits_max_freq =np.array(freq))
        
        
        
        if show_proprioception:

            def on_thresh_slider(val):
            # adds slider to change online parameters of the class
            # TODO check values in slider
                val /= 100
                esim.position_max_freq = val
                esim.velocity_max_freq = val
                esim.load_max_freq = val
                esim.limits_max_freq = val
                
            for  i in range(len(joint_list)):
                events_window_name = events_window_names[i]
                cv2.namedWindow(events_window_name)
                
                cv2.createTrackbar("Threshold", events_window_name, int(
                    esim.position_max_freq * 100), 100, on_thresh_slider)
                cv2.setTrackbarMin("Threshold", events_window_name, 1)
                
        return esim
            
    else:
        events = esim.update(x = np.asarray(joint_pos), v = joint_vel, load = joint_load , time_stamp = time )
        previous_events.extend(events) 

        if len(previous_events):
            # TODO : nothing is being discarded cause I do not append them here 
            # and events is just the output of update
            events_array = np.array(previous_events)
            while events_array[0, 2] < time - TIME_WINDOW:
                events_array = np.delete(events_array, 0, axis=0)
            previous_events = list(events_array)
        if DEBUG:
            if len(events):
                logging.info(f"Generated {len(events)} proprioception events.")                
                
        
        if show_proprioception:
            if len(previous_events):
                # width is 1000, literally just window size
                for  i in range(len(joint_list)):
                    events_window_name = events_window_names[i]
                    width = 500
                    events_array = np.array(previous_events)
                    idc = np.where(events_array[:, 1] == i)
                    cv2.imshow(events_window_name, make_prop_event_frame(width=width, time=time, time_window=TIME_WINDOW,  previous_events=events_array[idc]))
                
                cv2.waitKey(1) 
            
        # only returns the event of the current timestamp, much like camera and skin. 
        #     
        return events
    





