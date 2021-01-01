import os
import keras.backend as K

import numpy as np
import rl.core
from PIL import Image

INPUT_SHAPE = (84, 84)

class RajagopalProcessor(rl.core.Processor):
    """
        An experimental processor that incorporates domain knowledge into the agent's behavior.

        Inheriting from Keras RL Processor, this class serves as a liaison between the
        agent and the environment. If instantiated this processor attempts to incorporate
        domain knowledge of the environment Seaquest (Atari 2600) to improve the resulting policy.

        NOTE: Since this processor is hand-crafted for Seaquest, it will not work as intended with
        other environments.

        Parameters
        ----------
        nb_conditional: int
            The number of conditions present within the completion vector.
            Note that this may be larger than the number of conditions present
            in the conditional matrix.
        testing: bool
            A flag indicating whether the agent is training or testing.
            Used to ensure shaping rewards and other modifications (e.g., option heads)
            are properly handled.
        base_diver_reward: float
            The reward value that is used when the agent picks up a diver.
        diver_model: Keras.Model
            A Keras Model (built with Functional API) that detects diver positions
            via multi-label image classification. The classes, in order, are as follows:
            (no_divers, up_left, down_left, up_right, down_right).
    """

    def __init__(self, nb_conditional, testing, base_diver_reward, diver_model=None, use_state_augmentation=True, use_action_shaping=False):
        self.num_cond = nb_conditional
        # The conditional matrix as of the current time-step
        self.cond_matrix = np.zeros(shape=(nb_conditional,))
        # The conditional matrix as of the previous time-step
        self.prev_step_cond_matrix = np.zeros(shape=(nb_conditional,))
        # The action the agent has taken. Used for reward shaping
        self.action_taken = None
        # The number of divers the agent has in the current time-step
        self.num_divers = 0
        # The number of divers the agent had in the previous time-step
        self.prev_num_divers = 0
        # A variable to hold the shaping reward generated at the current time-step
        self.shaping_reward = 0
        # The current number of lives the agent has
        self.num_lives = None
        # The number of lives the agent had in the previous time-step
        self.prev_num_lives = None
        ### BASE SHAPING SIGNALS ###
        self.base_reward_diver = base_diver_reward
        self.oxygen_reward = 0.2
        self.up_with_max_divers_reward = 0.3
        ### DIVER LOCATOR INFO ###
        self.diver_model = diver_model
        if self.diver_model is None:
            self.diver_output = np.array([1, 0, 0, 0, 0])
        else:
            self.diver_output = np.zeros(shape=(5,))
        self.prev_diver_output = self.diver_output
        self.location_action_mapping = self.generate_location_action_mapping()
        self.move_towards_diver_reward = 0.0001
        ### BDQN ARGUMENTS ###
        self.current_head = 0
        ### TESTING FLAG ###
        self.testing = testing
        ### STATE AUGMENTATION FLAG ###
        self.use_state_augmentation = use_state_augmentation
        ### ACTION SHAPING FLAG ###
        self.use_action_shaping = use_action_shaping
        ### LOGGING ONLY ###
        self.total_divers = 0
        self.max_diver_count = 0

    def process_observation(self, observation):
        """
            Process the observation.
            
            Parameters
            ----------
            observation: numpy.ndarray
                A numpy array with dimensions (210, 160, 3) that
                corresponds to an RGB image of the current game frame.
            
            Returns
            -------
            processed_observation: numpy.ndarray
                A numpy array with dimensions (84, 84) that
                corresponds to a grayscale image of the current game frame.
            cond_matrix: numpy.ndarray
                The conditional matrix as of the current time-step.
                Contains information about the number of divers the
                agent has currently, along with if divers are present.
        """

        # Observation itself should be an RGB image of shape (210, 160, 3)
        # NOTE: Might be better to check if it's an instance of tuple instead
        if self.use_state_augmentation:
            if len(observation) == 2:
                # This observation has already been processed. No need to do more.
                return observation
        else:
            if observation.ndim < 3:
                # Observation was already processed.
                return observation
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        threshold = 142.0 # The base value in the section of screen where the divers are.
        lazy_frame_slice = processed_observation
        diver_1 = lazy_frame_slice[72:74,32:34]
        diver_2 = lazy_frame_slice[72:74,36:38]
        diver_3 = lazy_frame_slice[72:74,40:42]
        diver_4 = lazy_frame_slice[72:74,44:46]
        diver_5 = lazy_frame_slice[72:74,48:50]
        diver_6 = lazy_frame_slice[72:74,52:54]
        divers = [diver_1, diver_2, diver_3, diver_4, diver_5, diver_6]
        # Determine if the oxygen is below a certain percentage (e.g., 10%)
        oxygen_threshold_area = lazy_frame_slice[68:70,26:28]
        oxygen_threshold = 80.0
        # Count the number of divers present by checking if the mean value of the kernel is below the flat value
        num_divers = 0
        for diver in divers:
            if np.mean(diver) < threshold:
                num_divers +=1
        # Set the previous step conditional matrix equal to the current step before modifications.
        self.prev_num_divers = self.num_divers
        self.num_divers = num_divers
        # Update the max diver count if applicable.
        if self.num_divers > self.max_diver_count:
            self.max_diver_count = self.num_divers
        # Update the conditional matrix to match the number of divers present.
        for index, value in enumerate(self.cond_matrix):
            self.prev_step_cond_matrix[index] = value

        # Shape of conditional matrix [>=1, >=2, >=3, >=4, >=5, ==6, o2, diver_onscreen]
        #                               0   1    2    3    4    5   6        7
        # First fill in the number of divers with 1 values (e.g., 3 divers -> [1, 1, 1]), then zero pad until the number of conditions in the CV is present
        self.cond_matrix = np.pad(np.ones(self.num_divers), (0, (self.num_cond - self.num_divers)), 'constant', constant_values=0)
        if np.mean(oxygen_threshold_area) < oxygen_threshold:
            self.cond_matrix[-2] = 1
        else:
            self.cond_matrix[-2] = 0
        
        # Use the diver locator to determine where the divers are (if any)
        if self.diver_model:
            self.predict_divers(processed_observation)
        if self.use_state_augmentation:
            return processed_observation, self.cond_matrix  # saves storage in experience memory
        else:
            return processed_observation

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it. Resets the DFA if it was in a violating state.
        # Arguments
            reward (float): A reward as obtained by the environment
        # Returns
            Reward obtained by the environment processed
        """

        # Set the base reward to zero
        shaping_reward = 0.0
        # Determine the change in diver counts between the two time-steps
        diver_delta = np.sum(self.cond_matrix[:-2] - self.prev_step_cond_matrix[:-2])
        # If we've got the number of lives at the current time step and previous time step calculate the delta
        if self.num_lives and self.prev_num_lives:
            life_delta = self.num_lives - self.prev_num_lives
        else:
            # If we do not, assume the delta is 0 (no change)
            life_delta = 0
        # Realistically, it isn't possible to gain or lose more than one diver between two game frames.
        # This is also done due to the diver count flashing at six divers every game frame (meaning the change is +- 6 every 2 frames)
        if diver_delta > 1 or diver_delta < -1:
            diver_delta = 0
        diver_index = None
        # If a diver is picked up this frame (noted by positive delta value)
        if diver_delta > 0:
            # Increment by one (safety against changes larger than one which should not happen)
            self.total_divers += 1
            # Locate which diver of six was picked up (+1 to account for 0-indexing). The [-1] here is fetching the last index where a value of 1 is found.
            diver_index = int(np.argwhere(self.cond_matrix[:-2] == 1)[-1]) + 1
        # This condition should only hold if the agent surfaces and loses a diver as a result.
        elif diver_delta < 0:
            if life_delta == 0 and self.cond_matrix[-2] == 1:
                diver_index = 0
            else:
                # Figure out which diver of six was lost to determine points taken. The [-1] here is fetching the last index where a value of 1 is found.
                diver_index = int(np.argwhere(self.prev_step_cond_matrix[:-2] == 1)[-1]) + 1
        else:
            diver_index = 0
        # Add shaping reward equal to the base_diver_reward multiplied by the index and the delta value
        shaping_reward += diver_delta * diver_index * self.base_reward_diver
        # Add a reward if the agent is moving closer to a diver (if one is present)
        # shaping_reward += self.determine_diver_movement_reward()
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

        # Observation is a tuple of length 2 (that's why the check exists!)
        observation = self.process_observation(observation)
        return observation, action, reward, done

    def process_action(self, action, q_value):
        self.action_taken = action
        return action

    def process_state_batch(self, batch):
        """
            Process a state batch (with augmentations).

        """
        if self.use_state_augmentation:
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
        else:
            # We could perform this processing step in `process_observation`. In this case, however,
            # we would need to store a `float32` array instead, which is 4x more memory intensive than
            # an `uint8` array. This matters if we store 1M observations.
            processed_batch = batch.astype('float32') / 255.
            return processed_batch

    def predict_divers(self, processed_observation):
        """
            Determine where divers are located, if any.

            Using a processed observation at the current time-step,
            this function uses the diver model provided to predict if divers
            are present in the current observation, outputting their locations.

            NOTE: The diver model should be multi-label and pre-trained for optimal results.
            
            Parameters
            ----------
            processed_observation: numpy.ndarray
                A numpy array with dimensions (84, 84) that
                corresponds to a grayscale image of the current game frame.
            
            Returns
            -------
            None.
        """

        # Add a batch shape and channel dimension back to the image.
        obs = np.reshape(processed_observation, (1, 1,) + processed_observation.shape)
        assert obs.shape == (1, 1, 84, 84)
        # Run a single image prediction and transform it to a list
        diver_locations = np.squeeze(self.diver_model.predict(obs), axis=0).round()
        # Update the previous output before replacing current output
        self.prev_diver_output = self.diver_output
        # If none of the outputs were above the threshold (i.e., zero vector output), default to 'no_divers'
        if not np.any(diver_locations):
            self.diver_output = np.array([1, 0, 0, 0, 0])
            self.cond_matrix[-1] = 0
        # Set the current output and update conditional matrix
        else:
            self.diver_output = diver_locations
            self.cond_matrix[-1] = 1

    def generate_location_action_mapping(self):
        """
            Generate a boolean mask for each possible action in Seaquest.

            This function generates a boolean mask for each action in Seaquest.
            Each row corresponds to a specific action (e.g., up, down, etc.), and
            each column corresponds to the diver location where such an action is applicable.
            A simple example is, if divers are detected in the upper left quadrant and lower left
            quadrant, then the actions that correspond to moving left, up, or down would be applicable.

            NOTE: This function should only be called once.

            Parameters
            ----------
            None.

            Returns
            -------
            location_action_mapping: numpy.ndarray
                An array with shape (nb_actions, len(diver_locations)) that maps
                a boolean mask to each action, based on the diver location.
        """

        # No diver, up_left, down_left, up_right, down_right
        list_mapping = [
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False,  True, False,  True, False],
            [False, False, False,  True,  True],
            [False,  True,  True, False, False],
            [False, False,  True, False,  True],
            [False, False, False,  True, False],
            [False,  True, False, False, False],
            [False, False, False, False, True],
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
        """
            Determine what shaping reward should be generated for moving towards a diver.

            Based on the action taken at the current time-step, this function determines if
            a shaping signal should be generated by determining if the action taken corresponds
            with movement towards a quadrant that contains a diver.

            Paramters
            ---------
            None.

            Returns
            -------
            diver_movement_signal: float
                A shaping signal value for moving towards a quadrant that contains a diver.
        """
        
        # If the diver output has the 'no_divers' bit on, then ignore remaining bits and return 0.
        if self.diver_output[0] == 1 or np.all(self.diver_output == 0):
            return 0.0
        else:
            actions_to_reward = np.zeros(shape=(18,), dtype='float32')
            np.any((self.location_action_mapping * self.diver_output), axis=1, out=actions_to_reward)
            return self.move_towards_diver_reward * actions_to_reward[self.action_taken]

    def determine_action_shaping(self):
        return self.determine_diver_locator_action_shaping()

    def determine_diver_locator_action_shaping(self):
        """
            Determine what shaping reward should be generated for moving towards a diver.

            Based on the action taken at the current time-step, this function determines if
            a shaping signal should be generated by determining if the action taken corresponds
            with movement towards a quadrant that contains a diver.

            Paramters
            ---------
            None.

            Returns
            -------
            diver_movement_signal: float
                A shaping signal value for moving towards a quadrant that contains a diver.
        """
        # If the diver output has the 'no_divers' bit on, return a mask of all values.
        if self.diver_output[0] == 1 or np.all(self.diver_output == 0):
            return np.ones(shape=(18,), dtype='bool')
        else:
            diver_output = self.diver_output
        actions_to_reward = np.zeros(shape=(18,), dtype='float32')
        np.any((self.location_action_mapping * diver_output), axis=1, out=actions_to_reward)
        return actions_to_reward