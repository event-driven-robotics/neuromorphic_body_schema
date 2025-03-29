import logging

import cv2
import numpy as np

TIME_WINDOW = 50000  # TODO make a global variable


def generalized_sigmoid(x: np.array,  x_min: np.array, x_max: np.array, y_min: np.array, y_max: np.array, B=np.array) -> np.array:
    """
    x = where to calculate the sigmoid

    Upon further consideration we will get rid of everything except for the parameters K (ymax) and B, and we will add the limits of the sigmoid
    """

    x_norm = (x - x_min) / (x_max - x_min)

    # Apply sigmoid function centered at 0.5
    y_norm = 1 / (1 + np.exp(-B * (x_norm - 0.5)))

    y_min_local = 1 / (1 + np.exp(-B * (0 - 0.5)))

    y_max_local = 1 / (1 + np.exp(-B * (1 - 0.5)))

    # Stretch y_norm to ensure it spans from 0 to 1
    y_norm = (y_norm - y_min_local) / (y_max_local - y_min_local)

    # Scale y to [y_min, y_max]
    y_scaled = y_min + (y_max - y_min) * y_norm

    y_scaled = np.where(np.asarray(y_scaled) < 0.0, 0, y_scaled)

    return y_scaled


def linear(x, m=1, q=0):
    """
    x = where to calculate the linear value
    m = slope
    q = offset
    """

    return m*x + q


class ProprioceptionEventSimulator():

    """"
    This class defines the spiking representation of proprioceptive inputs.
    Inputs are idealized to be from either fictious data, MuJoCo simulations, or iCub's own values from YARP.
    time_stamp is independent from the proprioception class and depends on the timestamps of the external inputs simulator.

    """

    def __init__(self, position_limits, velocity_limit, load_limit,  position_max_freq=np.array(1000.), velocity_max_freq=np.array(1000.), load_max_freq=np.array(1000.), limits_max_freq=np.array(1000.), DEBUG=False):
        """
        Function to initialize the values of the proprioceptive output
        We start by setting all the proprioceptive values to zero.
        We then set up the constants used for the model. These can be modified in the future as we see fit.

        Position_limits in input is thought to be as follows: [[minimum limits per joint],
                                             [maximum limits per joint] ]

        Velocity_limit and load_limit are thought to be scalar and positive. See explanation in the function self.velocity + notes from 19/02/2025
        """

        self.pos = np.zeros(2)  # position
        self.v = np.zeros(2)  # velocity
        self.l = np.zeros(2)  # load
        self.lim = np.zeros(2)  # closeness to limits

        # measuring unity is in Hz. 1000Hz means we expect 1000 events per second (circa 1 per ms).
        self.position_max_freq = position_max_freq
        self.velocity_max_freq = velocity_max_freq
        self.load_max_freq = load_max_freq
        self.limits_max_freq = limits_max_freq

        # parameters of the system
        self.position_limit_min = np.array(
            position_limits[0])  # limits of the joint positions
        self.position_limit_max = np.array(
            position_limits[1])  # limits of the joint positions
        self.velocity_limit = velocity_limit
        self.load_limit = load_limit

        self.time_of_last_spike = np.zeros((8))
        self.previous_timestamp = 0.0

        self.DEBUG = DEBUG

    def position(self, x: float, B):
        """
        Here we set up the neuron model to translate the value of the joint angles (positions, x as input)
        into an output frequency.

        Since from a single value of position (aka joint angle) we want the spikes of the agonistic-antagonistic system
        the output will be the two frequencies of the two neurons representing the two muscles

        M has been derived as the approximator for the tails of the distribution. It helps modulating B as a function of the limits. See notes from 19/02/2025 for more details
        """

        if (x - self.position_limit_min).any() < 0:
            print("WARNING: joint value outside scope")
        if (-x + self.position_limit_max).any() < 0:
            print("WARNING: joint value outside scope")

        y1 = 0.1 * generalized_sigmoid(x=x,   x_min=self.position_limit_min,
                                       x_max=self.position_limit_max, y_min=0.0, y_max=self.position_max_freq, B=B)
        y2 = 0.1*(self.position_max_freq - 10 * y1)

        # if self.DEBUG:
        #     x_debug = np.linspace(self.position_limit_min, self.position_limit_max, 1000)
        #     y_debug = generalized_sigmoid(x=x_debug,   x_min = self.position_limit_min, x_max=self.position_limit_max, y_min= 0. , y_max=self.position_max_freq, B = B )
        #     plt.figure()
        #     plt.plot(x_debug, y_debug)
        #     plt.plot(x_debug, self.position_max_freq - y_debug)
        #     plt.xlabel("Joint position [deg]")
        #     plt.ylabel("Firing rate [Hz]")
        #     plt.title("Position")
        #     plt.show()

        return (y1, y2)

    def velocity(self, v):
        """
        Here we set up the neuron model to translate the value of the joint angle velocity (velocity, v as input)
        into an output frequency.

        Since from a single value of velocity (aka joint angle velocity) we want the spikes of the agonistic-antagonistic system
        the output will be the two information of the two neurons representing:
        - the module of the frequency 
        - clockwise or anticlockwise movement (respectively 0 or 1). 


        We assume that there is only one v_max for all the joints, resulting in a modulation of the output with respect of this velocity. 
        This is a choice we made and that, if needed, can be changed in the future. See notes from 19/02/2025
        """

        if np.min(v) < - self.velocity_limit:
            v = np.where(v < (-self.velocity_limit), - self.velocity_limit, v)
            print("WARNING: joint velocity value reached")
        if np.max(v) > self.velocity_limit:
            v = np.where(v > self.velocity_limit,  self.velocity_limit, v)
            print("WARNING: joint velocity value reached")

        module = linear(np.abs(v), m=self.velocity_max_freq /
                        self.velocity_limit)
        direction_pos = np.where(v > 0, 1.0, 0.0)
        direction_neg = np.where(v < 0, 1.0, 0.0)

        y1 = module * direction_pos  # element-wise multiplication
        y2 = module * direction_neg  # element-wise multiplication

        return (y1, y2)

    def load(self, load):
        """
        Here we set up the neuron model to translate the value of the joint angle load (load as input)
        into an output frequency.

        Since from a single value of load (aka joint angle load) we want the spikes of the agonistic-antagonistic system
        the output will be the two information of the two neurons representing:
        - the module of the frequency 
        - clockwise or anticlockwise movement (respectively 0 or 1). 

        We assume that there is only one load_max for all the joints, resulting in a modulation of the output with respect of this load. 
        This is a choice we made and that, if needed, can be changed in the future. See notes from 19/02/2025
        """

        if np.min(load) < - self.load_limit:
            print("WARNING: joint load value reached")
            load = np.where(load < (-self.load_limit), - self.load_limit, load)
        if np.max(load) > self.load_limit:
            print("WARNING: joint load value reached")
            load = np.where(load > self.load_limit,  self.load_limit, load)

        module = linear(np.abs(load), m=self.load_max_freq / self.load_limit)
        direction_pos = np.where(load > 0, 1.0, 0.0)
        direction_neg = np.where(load < 0, 1.0, 0.0)

        y1 = module * direction_pos  # element-wise multiplication
        y2 = module * direction_neg  # element-wise multiplication

        return (y1, y2)

    def limit(self, limit, B, delta_x):
        """
        Here we set up the neuron model to translate the value of the joint angle limits (limit as input)
        into an output frequency.

        Since from a single value of limit (aka joint angle limits) we want the spikes of the agonistic-antagonistic system
        the output will be the two information of one neuron representing how close we are to the joint limits. 

        It is independent from which limit, this info can be derived from self.pos
        """

        if (limit - self.position_limit_min).any() < 0:
            print("WARNING: joint value outside scope")
        if (-limit + self.position_limit_max).any() < 0:
            print("WARNING: joint value outside scope")

        # parameters of the limit function

        # below here the two B must be the first positive and the second negative
        y1 = generalized_sigmoid(x=limit,   x_min=self.position_limit_min + 2*delta_x,
                                 x_max=self.position_limit_min, y_min=0.0, y_max=self.limits_max_freq, B=-B)
        y2 = generalized_sigmoid(x=limit,   x_min=self.position_limit_max-2*delta_x,
                                 x_max=self.position_limit_max, y_min=0.0, y_max=self.position_max_freq, B=B)

        if self.DEBUG:
            x_debug = np.linspace(self.position_limit_min,
                                  self.position_limit_max, 1000)
            y_debug1 = generalized_sigmoid(x=x_debug,   x_min=self.position_limit_min + 2*delta_x,
                                           x_max=self.position_limit_min, y_min=0., y_max=self.limits_max_freq, B=-B)
            y_debug2 = generalized_sigmoid(x=x_debug,   x_min=self.position_limit_max-2*delta_x,
                                           x_max=self.position_limit_max, y_min=0., y_max=self.position_max_freq, B=B)

            plt.figure()
            plt.plot(x_debug, y_debug1, label="descending")
            plt.plot(x_debug, y_debug2, label="ascending")
            plt.xlabel("Joint position [deg]")
            plt.ylabel("Firing rate [Hz]")
            plt.title("Limit")
            plt.legend()
            plt.show()

        return (y1, y2)

    def proprioceptionCallback(self, x, v, load, time_stamp, B_pos=5.0, B_lim=20.0, delta_x=1.0):
        # update the joint positions values
        pos = self.position(x, B=B_pos)
        v = self.velocity(v)
        l = self.load(load)
        lim = self.limit(limit=x, B=B_lim, delta_x=delta_x)

        events = []

        def invert_tuple(frequ):
            # Check if the current element is a tuple
            if isinstance(frequ, tuple):
                # Apply inversion to each element recursively
                return tuple(invert_tuple(x) for x in frequ)
            else:
                # delta_t = np.where(frequ == 0.0, np.inf, 1/frequ)
                if frequ == 0.0:
                    delta_t = np.inf
                else:
                    delta_t = 1/frequ
                return delta_t

        # Develop the events according to the AER protocol

        # estimate of the delta time before we should have another spike
        is_it_time_to_spike = np.zeros(8)
        is_it_time_to_spike[0] = invert_tuple(pos[0])
        is_it_time_to_spike[1] = invert_tuple(pos[1])
        is_it_time_to_spike[2] = invert_tuple(v[0])
        is_it_time_to_spike[3] = invert_tuple(v[1])
        is_it_time_to_spike[4] = invert_tuple(l[0])
        is_it_time_to_spike[5] = invert_tuple(l[1])
        is_it_time_to_spike[6] = invert_tuple(lim[0])
        is_it_time_to_spike[7] = invert_tuple(lim[1])

        # if not working look for np.hstack or np.vstack
        # is_it_time_to_spike = place_holder

        for proprioc_output in range(8):
            while self.time_of_last_spike[proprioc_output] + is_it_time_to_spike[proprioc_output] < time_stamp:
                self.time_of_last_spike[proprioc_output] = self.time_of_last_spike[
                    proprioc_output] + is_it_time_to_spike[proprioc_output]
                events.append(
                    (proprioc_output, self.time_of_last_spike[proprioc_output], 0))

        if len(events):
            events = np.array(events)
            events = events[np.argsort(events[:, 2])]

        return events


def make_prop_event_frame(img, width, time, time_window, events):
    # here we take the old image, move it's content to the left and add the new events
    # remove old events
    if len(events):
        # TODO : nothing is being discarded cause I do not append them here
        # and events is just the output of update
        events_array = np.array(events)
        while events_array[0, 2] < time - TIME_WINDOW:
            events_array = np.delete(events_array, 0, axis=0)
        events = list(events_array)

    # TODO do I need esim?
    tick_height = 20
    margin = 5
    height = 8 * (tick_height + margin)
    current_img = np.zeros((height, width, 3))

    t_left = time-time_window
    scale = width/time_window

    if len(events):
        for single_event in events:
            # TODO controlla that this is still not a problem of the idx 0 overriding the info in the event representation
            y_val = single_event[0]*tick_height
            x_val = int((single_event[2]-t_left)*scale)
            neuron_idx = single_event[0]

            color_index = int((neuron_idx * 255 / 8))
            color = cv2.applyColorMap(
                np.array([[color_index]], dtype=np.uint8), cv2.COLORMAP_HSV)[0][0]
            color = color/255

            cv2.line(current_img, (x_val, int(y_val)+margin),
                     (x_val, int(y_val)+tick_height), color, 1)

    return current_img


class ICubProprioception:
    def __init__(self, model, joint_dict, show_proprioception=False, DEBUG=False):
        # nb_joints = len(joint_dict.keys())
        # TODO take arguments for the event encoding!
        # max_max_freq = 1e-4
        self.events_window_names = []
        self.esim = []
        self.joint_dict = joint_dict
        self.imgs = []
        self.show_proprioception = show_proprioception
        self.DEBUG = DEBUG

        # TODO replace with joint states I want to access
        for joint_name in list(joint_dict.keys()):
            position_max_freq = joint_dict[joint_name]['position_max_freq']
            velocity_max_freq = joint_dict[joint_name]['velocity_max_freq']
            load_max_freq = joint_dict[joint_name]['load_max_freq']
            limits_max_freq = joint_dict[joint_name]['limits_max_freq']
            # joint_pos[i] = data.joint(joint_list[i]).qpos # example for single joint
            # joint_vel[i] = data.joint(joint_list[i]).qvel
            # joint_load[i] = data.joint(joint_list[i]).qacc
            self.events_window_names.append(
                f'{joint_name} proprioception events')
            self.imgs.append(np.zeros((8, 500, 3), dtype=np.uint8))
            self.esim.append(ProprioceptionEventSimulator(position_limits=model.actuator(joint_name).ctrlrange,
                                                          velocity_limit=20.0, load_limit=10000000.0, position_max_freq=position_max_freq, velocity_max_freq=velocity_max_freq,
                                                          load_max_freq=load_max_freq, limits_max_freq=limits_max_freq, DEBUG=DEBUG))

        # The values used for the next object are absolutely arbitrary, they will be changed in a later stage
        # also based on the platform the code is going to be used on
        # max freq is 0.1 cause time is in ns

        if show_proprioception:

            # def on_thresh_slider(val):
            #     # adds slider to change online parameters of the class
            #     # TODO check values in slider
            #     val /= 100
            #     esim.position_max_freq = val
            #     esim.velocity_max_freq = val
            #     esim.load_max_freq = val
            #     esim.limits_max_freq = val

            for events_window_name in self.events_window_names:
                # events_window_name = events_window_names[i]
                cv2.namedWindow(events_window_name)

                # cv2.createTrackbar("Threshold", events_window_name, int(
                #     esim.position_max_freq * 100), 100, on_thresh_slider)
                # cv2.setTrackbarMin("Threshold", events_window_name, 1)

    def update_proprioception(self, time, data):
        all_events = []
        for esim_single, joint_name, img in zip(self.esim, list(self.joint_dict.keys()), self.imgs):
            joint_pos = data.joint(joint_name).qpos  # example for single joint
            joint_vel = data.joint(joint_name).qvel
            joint_load = data.joint(joint_name).qacc
            events = esim_single.proprioceptionCallback(
                x=joint_pos, v=joint_vel, load=joint_load, time_stamp=time)
            all_events.append(events)

            if self.DEBUG:
                if len(events):
                    logging.info(
                        f"Generated {len(events)} proprioception events.")

            if self.show_proprioception:
                if len(events):
                    # width is 1000, literally just window size
                    for events_window_name in self.events_window_names:
                        width = 500
                        events_array = np.array(events)
                        cv2.imshow(events_window_name, make_prop_event_frame(img=img,
                                                                             width=width, time=time, time_window=TIME_WINDOW,  events=events_array))

                    cv2.waitKey(1)

        # only returns the event of the current timestamp, much like camera and skin.
        #
        return all_events
