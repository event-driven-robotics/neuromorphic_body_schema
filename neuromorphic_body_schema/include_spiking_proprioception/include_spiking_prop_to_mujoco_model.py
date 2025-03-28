import numpy as np
import matplotlib.pyplot as plt
import os as os
import matplotlib as mpl
from tqdm import tqdm
import cv2

import matplotlib.animation as animation

from proprioception import generalized_sigmoid, linear, proprioception



# test whether the proprioceptuion code works
time = np.linspace(0, 200, 2002) # time measure
dt = 0.1 # dt in ms
position = np.linspace(0, 100, 1001) # deg - position measure

position = np.append(position, np.linspace(100, 0, 1001)) # deg - position measure
vel = np.diff(position) # * dt
# vel = 0.1 * np.sign(position)# deg/time measure
position = position[:-1]
load = np.sin(0.02*np.pi*time)



fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12,8))
ax.set_xlim([0,time[-1]])
ax.set_ylim([0.5,12.0])

title = "Spiking proprioception test for show"

ax.set_title(title)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Load_N1   Load_N2   Load_e   V_N1   V_N2   V_e    Lim_N1   Lim_N2   Pos_N1   Pos_N2  Pos_e ")

# scat = ax.scatter([], [], s=200, marker = '|', label='Events')


scat1 = ax.scatter([], [], s=200, marker = '|', label='Events')
scat2 = ax.scatter([], [], s=200, marker = '|', label='Events')
scat3 = ax.scatter([], [], s=200, marker = '|', label='Events')
scat4 = ax.scatter([], [], s=200, marker = '|', label='Events')
scat5 = ax.scatter([], [], s=200, marker = '|', label='Events')
scat6 = ax.scatter([], [], s=200, marker = '|', label='Events')
scat7 = ax.scatter([], [], s=200, marker = '|', label='Events')
scat8 = ax.scatter([], [], s=200, marker = '|', label='Events')

position_encoded = 10.5 + 0.01*position
velocity_encoded = 6.0+ vel
load_encoded = 3 + 0.5*load

pos_encod = ax.plot(time[0], position_encoded[0], label= 'position encoder')[0]
vel_encod = ax.plot(time[0], velocity_encoded[0], label='velocity encoder')[0]
load_encod = ax.plot(time[0], load_encoded[0], label='load encoder')[0]

frames_dir = "/home/mbarborini/code/spiking_proprioception/saved_output"
frame_counter = 0

instance = proprioception(n_joints=1, position_limits= [ np.min(position)   , np.max(position) ] , velocity_limit=100.0, load_limit= 100.5, position_max_freq=np.array(100.), velocity_max_freq=np.array(100.), load_max_freq=np.array(100.), limits_max_freq =np.array(100.), DEBUG = False)

vertical_setup = np.array([10, 9, 5, 4, 2, 1, 8,7]) # np.arange(1,9,1) top down: pos_enc, pos_spk, Lim_spk, vel_enc, vel_spk, load_enc, load_spk 
times = np.tile(time, (len(vertical_setup),1))

spikes = np.ones((len(time+1), len(vertical_setup))) * vertical_setup



for t in range(len(time)-1):
    spikes[t+1] = instance.update(x = np.asarray(position[t]), v = vel[t], load = load[t] , time_stamp = time[t]).flatten() *vertical_setup


def animate(frame):

    global instance
    global frame_counter

    x = time[:frame]
    y = spikes[:frame]
    
    data1 = np.stack([x, y[:, 0]]).T
    scat1.set_offsets(data1)

    data2 = np.stack([x, y[:, 1]]).T
    scat2.set_offsets(data2)


    data3 = np.stack([x, y[:, 2]]).T
    scat3.set_offsets(data3)


    data4 = np.stack([x, y[:, 3]]).T
    scat4.set_offsets(data4)


    data5 = np.stack([x, y[:, 4]]).T
    scat5.set_offsets(data5)



    data6 = np.stack([x, y[:, 5]]).T
    scat6.set_offsets(data6)

    data7 = np.stack([x, y[:, 6]]).T
    scat7.set_offsets(data7)

    data8 = np.stack([x, y[:, 7]]).T
    scat8.set_offsets(data8)


    pos_encod.set_xdata(x)
    pos_encod.set_ydata(position_encoded[:frame])


    vel_encod.set_xdata(x)
    vel_encod.set_ydata(velocity_encoded[:frame])


    load_encod.set_xdata(x)
    load_encod.set_ydata(load_encoded[:frame])

    # data = np.column_stack([times[:,:frame+1],spikes[:frame+1].T])
    # print(data)

    # print(spikes[i])
    # scat.set_offsets(data)


    plt.savefig(os.path.join(frames_dir, f'frame_{frame_counter:04d}.png'))
    frame_counter += 1  

    return scat1, scat2, scat3, scat4, scat5, scat6, scat7, scat8, pos_encod, vel_encod, load_encod

anim = animation.FuncAnimation(fig, animate, interval=5, blit=False)

plt.show()


# Get the list of frame files
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
 
frame_height, frame_width = cv2.imread(os.path.join(frames_dir, frame_files[0])).shape[:2]
 
fps = 24  # Frames per second
 
# Define the codec and create a VideoWriter object
output_file = title+'.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
 
# Loop through all the frames and write them to the video file
for frame_file in frame_files:
    frame = cv2.imread(os.path.join(frames_dir, frame_file))
    out.write(frame)
 
# Release the VideoWriter object
out.release()




