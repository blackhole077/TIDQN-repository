import os
import keras.backend as K

import numpy as np
import rl.core
from PIL import Image

INPUT_SHAPE = (84, 84)

class RajagopalProcessor(rl.core.Processor):

    def __init__(self, nb_conditional, testing, base_diver_reward, diver_model):
        self.cond_matrix = np.zeros(shape=(nb_conditional,))
        self.prev_step_cond_matrix = np.zeros(shape=(nb_conditional,))
        self.action_taken = None
        self.num_divers = 0
        self.prev_num_divers = 0
        self.shaping_reward = 0
        self.num_lives = None
        self.prev_num_lives = None
        ### BASE SHAPING SIGNALS ###
        self.base_reward_diver = base_diver_reward
        self.oxygen_reward = 0.2
        self.up_with_max_divers_reward = 0.3
        self.testing = testing
        ### DIVER LOCATOR INFO ###
        self.diver_model = diver_model
        self.location_action_mapping = self.generate_location_action_mapping()
        self.move_towards_diver_reward = 0.0001
        self.diver_output = np.zeros(shape=(5,))
        self.prev_diver_output = self.diver_output
        ### BDQN ARGUMENTS ###
        self.current_head = 0
        ### LOGGING ONLY ###
        self.total_divers = 0
        self.max_diver_count = 0

    def process_observation(self, observation):
        # Observation itself should be an RGB image of shape (210, 160, 3)
        # NOTE: Might be better to check if it's an instance of tuple instead
        if len(observation) == 2:
            # This observation has already been processed. No need to do more.
            return observation
        else:
            # This observation image is still an RGB of shape (210, 160, 3)
            ob = observation
            # Make sure it does have 3 dimensions
            assert ob.ndim == 3  # (height, width, channel)
            img = Image.fromarray(ob)
            # resize to (84, 84) and convert to grayscale
            img = img.resize(INPUT_SHAPE).convert('L')
            # Transform it into a Numpy array
            processed_observation = np.array(img)
            # Make sure the resizing is correctly done
            assert processed_observation.shape == INPUT_SHAPE
            # Set data type to uint8
            processed_observation.astype('uint8')
        threshold = 142.0 # The base value in the section of screen where the divers are.
        #4x4 kernels that are "center-of-mass" on each diver.
        #For each frame in the LazyFrame stack (4 total), count the number of divers.
        #If there is at least one, mark the first column ("Has at least one diver")
        #If there's exactly six, mark the second column ("Has max capacity")
        # for frame in range(processed_observation.shape[2]):
        lazy_frame_slice = processed_observation#[:,:,frame]
        diver_1 = lazy_frame_slice[72:74,32:34]
        diver_2 = lazy_frame_slice[72:74,36:38]
        diver_3 = lazy_frame_slice[72:74,40:42]
        diver_4 = lazy_frame_slice[72:74,44:46]
        diver_5 = lazy_frame_slice[72:74,48:50]
        diver_6 = lazy_frame_slice[72:74,52:54]
        divers = [diver_1, diver_2, diver_3, diver_4, diver_5, diver_6]
        oxygen_threshold_area = lazy_frame_slice[68:70,26:28]
        oxygen_threshold = 80.0
        num_divers = 0
        for diver in divers:
            if np.mean(diver) < threshold:
                num_divers +=1
        self.prev_num_divers = self.num_divers
        self.num_divers = num_divers
        #Set the previous step conditional matrix equal to the current step before modifications.
        for index, value in enumerate(self.cond_matrix):
            self.prev_step_cond_matrix[index] = value

        # Shape of conditional matrix [>=1, >=2, >=3, >=4, >=5, ==6, o2, diver_onscreen]
        #                               0   1    2    3    4    5   6        7
        # Iterate of the number of possible non-zero diver counts [1, 6] and adjust the conditional matrix
        for i in range(6):
            if (i+1) <= self.num_divers:
                self.cond_matrix[i] = 1
            else:
                self.cond_matrix[i] = 0
        if np.mean(oxygen_threshold_area) < oxygen_threshold:
            self.cond_matrix[-2] = 1
        else:
            self.cond_matrix[-2] = 0
        
        # Use the diver locator to determine where the divers are (if any)
        self.predict_divers(processed_observation)

        # If testing, manually change the heads to either 0-head (default) or diver-head (1)
        if self.testing:
            if self.cond_matrix[-1] == 1:
                self.current_head = 1
            else:
                self.current_head = 0

        return processed_observation, self.cond_matrix  # saves storage in experience memory

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it. Resets the DFA if it was in a violating state.
        # Arguments
            reward (float): A reward as obtained by the environment
        # Returns
            Reward obtained by the environment processed
        """
        up_actions = [2, 6, 7, 10, 14, 15]
        
        if self.current_head == 0:
            self.shaping_reward = 0.0
            return np.clip(reward, -1., 1.)
        else:
            # Set the base reward to zero
            shaping_reward = 0.0
            # If we gain a diver, increase reward (If the first diver is picked up, reward is 1.5)
            diver_delta = np.sum(self.cond_matrix[:-1] - self.prev_step_cond_matrix[:-1])
            # If we've got the number of lives at the current time step and previous time step calculate the delta
            if self.num_lives and self.prev_num_lives:
                life_delta = self.num_lives - self.prev_num_lives
            else:
                # If we do not, assume the delta is 0 (no change)
                life_delta = 0

            if diver_delta > 1 or diver_delta < -1:
                diver_delta = 0
            diver_index = None
            # If a diver is picked up this frame (noted by positive delta value)
            if diver_delta > 0:
                # Increment by one (safety against changes larger than one which should not happen)
                self.total_divers += 1
                # Locate which diver of six was picked up (+1 to account for 0-indexing)
                diver_index = int(np.argwhere(self.cond_matrix[:-1] == 1)[-1]) + 1
            # This condition should only hold if the agent surfaces and loses a diver as a result.
            elif diver_delta < 0:
                if life_delta == 0 and self.cond_matrix[-1] == 1:
                    diver_index = 0
                else:
                    # Figure out which diver of six was lost to determine points taken
                    diver_index = int(np.argwhere(self.prev_step_cond_matrix[:-1] == 1)[-1]) + 1
            else:
                diver_index = 0
            # Add shaping reward equal to the base_diver_reward multiplied by the index and the delta value
            shaping_reward += diver_delta * diver_index * self.base_reward_diver
            # Add a reward if the agent is moving closer to a diver (if one is present)
            shaping_reward += self.determine_diver_movement_reward()
            # Incentivize going to the surface
            # if self.cond_matrix[-2] == 1 and self.cond_matrix[0] == 1:
            #     if self.action_taken in up_actions:
                    # shaping_reward += self.oxygen_reward
            # Punish the agent for losing lives
            if life_delta < 0:
                shaping_reward -= 0.5

            # For logging purposes, take note of the shaping reward
            self.shaping_reward = shaping_reward
            if not self.testing:
                return np.clip(reward, -1., 1.) + shaping_reward
            else:
                return np.clip(reward, -1., 1.)

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            info (dict): An info as obtained by the environment
            Currently only contains ale.lives in Seaquest, which
            refers to the number of lives left.
        # Returns
            Info obtained by the environment processed
        """
        info['shaping_reward'] = self.shaping_reward
        info['num_divers'] = self.num_divers
        info['total_divers'] = self.total_divers
        info['max_divers_for_episode'] = self.max_diver_count
        if self.num_lives:
            self.prev_num_lives = self.num_lives
        self.num_lives = info.get('ale.lives')

        return info
    
    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.
        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.
        # Returns
            The tuple (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        if done:
            self.max_diver_count = 0

        return observation, reward, done, info

    def process_experience(self, observation, action, reward, done):
        """
            Parameters
            ----------
            observation : tuple
                A tuple that should contain a processed image of shape
                (84,84) as opposed to (210, 160, 3) and a conditional
                matrix of shape (nb_conditional, ) to be processed.
        """
        #Observation is a tuple of length 2 (that's why the check exists!)
        '''
        print("Observation ChecK!\n")
        print("Length of Observation: {}\n".format(len(observation)))
        if len(observation) == 2:
            print("Length of Observation(0): {}\n".format(observation[0].shape))       
            print("Shape of Observation(1): {}\n".format(observation[1].shape))
        else:
            print("Shape of Observation: {}\n".format(observation.shape))
        '''
        observation = self.process_observation(observation)
        if done and not self.testing:
            self.current_head += 1
        return observation, action, reward, done

    def process_action(self, action, q_value):
        self.action_taken = action
        return action

    def process_state_batch(self, batch):
        """
            Process a state batch (with augmentations).

        """
        # The image section of the batch
        batch_images = []
        # The augmentation section of the batch
        batch_augmentation = []
        for b in batch:
            b_images = b[:, 0]
            b_images = np.array(b_images.tolist())
            # Apply the transformation needed by DQN
            b_images = b_images.astype('float32') / 255.
            # Append it back to batch_images
            batch_images += [b_images]

            # Augmentation section
            b_augmentations = b[:, 1]
            b_augmentations = np.array(b_augmentations.tolist())
            temp = []
            for bb in b_augmentations:
                temp += [bb.flatten()]
            batch_augmentation += [temp]
        batch_images = np.array(batch_images)
        batch_augmentation = np.array(batch_augmentation)
        return [batch_images, batch_augmentation]

    def predict_divers(self, processed_observation):
        obs = np.reshape(processed_observation, (1, 1,) + processed_observation.shape)
        assert obs.shape == (1, 1, 84, 84)
        diver_locations = np.squeeze(self.diver_model.predict(obs), axis=0)
        self.prev_diver_output = self.diver_output
        if not np.any(diver_locations):
            self.diver_output = np.array([1, 0, 0, 0, 0])
            self.cond_matrix[-1] = 0
        else:
            self.diver_output = diver_locations
            self.cond_matrix[-1] = 1

    def generate_location_action_mapping(self):
        # No diver, UL, DL, UR, DR
        list_mapping = [
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False,  True, False,  True, False],
            [False, False, False,  True,  True],
            [False,  True,  True, False, False],
            [False, False,  True, False,  True],
            [False, False, False,  True, False],
            [False,  True, False, False, False],
            [False, False, False, False, False],
            [False, False,  True, False, False],
            [False,  True, False,  True, False],
            [False, False, False,  True, False],
            [False,  True,  True, False, False],
            [False, False,  True, False,  True],
            [False, False, False,  True, False],
            [False,  True, False, False, False],
            [False, False, False, False,  True],
            [False, False,  True, False, False]]

        return np.array(list_mapping, dtype=bool)

    def determine_diver_movement_reward(self):
        if self.diver_output[0] == 1:
            return 0.0
        else:
            actions_to_reward = np.zeros(shape=(18,), dtype='float32')
            np.any((self.location_action_mapping * self.diver_output), axis=1, out=actions_to_reward)
            return self.move_towards_diver_reward * actions_to_reward[self.action_taken]