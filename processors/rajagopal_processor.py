import os

import numpy as np
import rl.core
from PIL import Image

INPUT_SHAPE = (84, 84)

class RajagopalProcessor(rl.core.Processor):

    def __init__(self, nb_conditional, testing, diver_model):
        self.cond_matrix = np.zeros(shape=(nb_conditional,))
        self.prev_step_cond_matrix = np.zeros(shape=(nb_conditional,))
        self.action_taken = None
        self.num_divers = 0
        self.prev_num_divers = 0
        self.shaping_reward = 0
        self.total_divers = 0
        self.num_lives = None
        self.prev_num_lives = None
        self.diver_locator = diver_model
        self.diver_output = None

        ### BASE SHAPING SIGNALS ###
        self.base_reward_diver = 1.0
        self.oxygen_reward = 0.2
        self.up_with_max_divers_reward = 0.3
        ### TESTING SIGNAL ###
        self.testing = testing
    def process_observation(self, observation):
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

        instance = generate_single_image_instance(processed_observation)
        self.diver_output = fetch_feature_extractor_outputs(instance, self.diver_locator)

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
            diver_change = 0
        else:
            diver_change = np.minimum(diver_delta, 1)
        oxygen_change = np.sum(self.cond_matrix[-1] - self.prev_step_cond_matrix[-1])

        # If a diver is picked up this frame (noted by positive delta value)
        if diver_change > 0:
            # Increment by one (safety against changes larger than one which should not happen)
            self.total_divers += 1
            # Locate which diver of six was picked up (+1 to account for 0-indexing)
            diver_index = int(np.argwhere(self.cond_matrix[:-1] == 1)[-1]) + 1
        # This condition should only hold if the agent surfaces and loses a diver as a result.
        elif life_delta == 0 and diver_change < 0:
            if oxygen_change > 0:
                shaping_reward += 0
            else:
                # Placeholder value since there are more nuanced cases that need handling first
                diver_index = int(np.argwhere(self.prev_step_cond_matrix[:-1] == 1)[-1]) + 1
                shaping_reward += diver_change * diver_index * (self.base_reward_diver + 0.001)
        # No (acutal) change in divers picked up
        elif diver_change == 0:
            diver_index = 0
        else:
            # Figure out which diver of six was lost to determine points taken
            diver_index = int(np.argwhere(self.prev_step_cond_matrix[:-1] == 1)[-1]) + 1
        # Add shaping reward equal to the base_diver_reward mulitplied by the index and the delta value
        shaping_reward += diver_change * diver_index * self.base_reward_diver
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
        # Punish the agent for losing lives
        if self.num_lives and self.prev_num_lives:
            if (self.num_lives - self.prev_num_lives) < 0:
                shaping_reward -= 1.0
        full_correct_reward = 0.00001
        half_correct_reward = full_correct_reward / 2.
        up_left_full = [7, 15]
        up_left_half = [2, 4, 6, 9, 10, 12, 14, 17]

        down_left_full = [9, 17]
        down_left_half = [4, 5, 7, 8, 12, 13, 15, 16]

        up_right_full = [6, 14]
        up_right_half = [2, 3, 7, 8, 10, 11, 15, 16]

        down_right_full = [8, 16]
        down_right_half = [3, 5, 6, 9, 11, 13, 14, 17]
        diver_quadrant_rewards = [[up_left_full, up_left_half],
                                  [down_left_full, down_left_half],
                                  [up_right_full, up_right_half],
                                  [down_right_full, down_right_half]
                                ]
        #Will output value between [0, 4]
        diver_quadrant = np.argmax(self.diver_output)
        if diver_quadrant > 0:
            if self.action_taken in diver_quadrant_rewards[(diver_quadrant-1)][0]:
                shaping_reward += full_correct_reward
            # elif self.action_taken in diver_quadrant_rewards[(diver_quadrant-1)][1]:
            #     shaping_reward += half_correct_reward
            else:
                shaping_reward += 0.0
        else:
            shaping_reward += 0.0
                
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
        self.action_taken = action
        return action

    def process_state_batch(self, batch):
        """
            Process a state batch (with augmentations).

            Batch comes in with shape (1,4,2)
            which is 1 batch of (4,2) instances
            So it seems this is a little more complex.
            batch[0][0] gives you the first tuple of the lazy frame (so [0][n] will give each of the four images)
            batch[0][0][0] gives you the first IMAGE of the first lazy frame stack
            batch[0][0][1] gives you the condition matrix (see above)
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

'''MISC. FUNCTIONS'''

def generate_single_image_instance(processed_observation):
    single_instance_image = np.expand_dims(np.repeat(processed_observation[:, :, np.newaxis], 4, axis=2), axis=0)
    single_instance_image = np.reshape(single_instance_image, (-1, 4, 84, 84))
    return single_instance_image

def fetch_feature_extractor_outputs(instance, feature_extractor):
    # Should give a softmax vector
    return feature_extractor.predict(instance)
