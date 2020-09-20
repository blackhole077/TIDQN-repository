import os
import keras.backend as K

import numpy as np
import rl.core
from PIL import Image

INPUT_SHAPE = (84, 84)

class RajagopalProcessor(rl.core.Processor):

    def __init__(self, nb_conditional, testing, base_diver_reward):
        self.cond_matrix = np.zeros(shape=(nb_conditional,))
        self.prev_step_cond_matrix = np.zeros(shape=(nb_conditional,))
        self.action_taken = None
        self.prev_actions = []
        self.num_divers = 0
        self.prev_num_divers = 0
        self.shaping_reward = 0
        ### BASE SHAPING SIGNALS ###
        self.base_reward_diver = base_diver_reward
        self.oxygen_reward = 0.2
        self.up_with_max_divers_reward = 0.3
        self.testing = testing

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

        #Shape of conditional matrix [>=1, >=2, >=3, >=4, >=5, ==6, o2]
        #                               0   1    2    3    4    5   6
        # Iterate of the number of possible non-zero diver counts [1, 6] and adjust the conditional matrix
        for i in range(6):
            if (i+1) <= self.num_divers:
                self.cond_matrix[i] = 1
            else:
                self.cond_matrix[i] = 0
        if np.mean(oxygen_threshold_area) < oxygen_threshold:
            self.cond_matrix[-1] = 1
        else:
            self.cond_matrix[-1] = 0
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
        down_actions = [5, 8, 9, 13, 16, 17]
        dither_constraint_1 = [0,1,0,1]
        dither_constraint_2 = [1,0,1,0]
        
        shaping_reward = 0.0
        # If we gain a diver, increase reward (If the first diver is picked up, reward is 1.5)
        # If a diver was picked up within the previous frame, add a reward signal.
        if self.num_divers - self.prev_num_divers > 0:
            if self.cond_matrix[0] == 1 and self.prev_step_cond_matrix[0] == 0:
                shaping_reward += self.base_reward_diver
            elif self.cond_matrix[1] == 1 and self.prev_step_cond_matrix[1] == 0:
                shaping_reward += self.base_reward_diver
            else:
                shaping_reward += self.base_reward_diver
        # Don't increment for no increase/decrease
        elif self.num_divers - self.prev_num_divers == 0:
            shaping_reward += 0
        # If you lose a diver
        else:
            # If the agent had at least one diver within the last frame, and then lost it
            if self.cond_matrix[0] == 0 and self.prev_step_cond_matrix[0] == 1:
                shaping_reward -= (2 * self.base_reward_diver)
            elif self.cond_matrix[1] == 0 and self.prev_step_cond_matrix[1] == 1:
                shaping_reward -= (2 * self.base_reward_diver)
            else:
                shaping_reward -= self.base_reward_diver
        # If the agent has picked up 6 divers, add a larger reward signal.
        if self.cond_matrix[5] == 1 and self.prev_step_cond_matrix[5] == 0:
            shaping_reward += (3 * self.base_reward_diver)
        # Attempt to address if the submarine successfully sufraces with 6 divers.
        if self.cond_matrix[5] == 0 and self.cond_matrix[0] == 0:
            if self.prev_step_cond_matrix[5] == 1 and self.cond_matrix[0] == 1:
                shaping_reward += (5 * self.base_reward_diver)
        # Attempt to increase reward for taking an UP related action with 6 divers.
        if self.cond_matrix[5] == 1:
            if self.action_taken is not None:
                if self.action_taken in up_actions:
                    shaping_reward += self.up_with_max_divers_reward

        # If the oxygen level is low, and the agent has at least one diver.
        if self.cond_matrix[-1] == 1 and self.cond_matrix[0] == 1:
            if self.action_taken in up_actions:
                shaping_reward += self.oxygen_reward
            else:
                shaping_reward -= self.oxygen_reward
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
        return observation, action, reward, done

    def process_action(self, action, q_value):
        up_actions = [2, 6, 7, 10, 14, 15]
        down_actions = [5, 8, 9, 13, 16, 17]
        if len(self.prev_actions) >= 4:
            del self.prev_actions[0]
        if action in up_actions:
            self.prev_actions.append(0)
        elif action in down_actions:
            self.prev_actions.append(1)
        else:
            self.prev_actions.append(2)
        self.action_taken = action
        return action

    def process_state_batch(self, batch):
        """
            Process a state batch (with augmentations).

        """
        #The image section of the batch
        batch_images = []
        #The augmentation section of the batch
        batch_augmentation = []
        for b in batch:
            b_images = b[:, 0]
            b_images = np.array(b_images.tolist())
            #Apply the transformation needed by DQN
            b_images = b_images.astype('float32') / 255.
            #Append it back to batch_images
            batch_images += [b_images]

            #Augmentation section
            b_augmentations = b[:, 1]
            b_augmentations = np.array(b_augmentations.tolist())
            temp = []
            for bb in b_augmentations:
                temp += [bb.flatten()]
            batch_augmentation += [temp]
        batch_images = np.array(batch_images)
        batch_augmentation = np.array(batch_augmentation)
        return [batch_images, batch_augmentation]