import numpy as np
import processors.contract_processor as contract_processor
import PIL

INPUT_SHAPE = (84, 84)

class ContractProcessorWithActionHistory(contract_processor.ContractProcessor):

    def __init__(self, reg_ex, mode, log_root, nb_actions, enforce_contract, action_history_size):
        super(ContractProcessorWithActionHistory, self).__init__(reg_ex, mode, log_root, nb_actions, enforce_contract)

        self._nb_actions = nb_actions
        self._action_history = np.zeros((1, action_history_size, nb_actions))

    def process_action(self, action, q_value):
        super(ContractProcessorWithActionHistory, self).process_action(action, q_value)

        self._action_history = np.delete(self._action_history[0], 1, 0)
        temp_array = np.zeros(self._nb_actions)
        temp_array[action] = 1
        self._action_history = np.append(self._action_history, [temp_array], axis=0)
        self._action_history = np.array([self._action_history])

        return action

    def process_observation(self, observation):
        if len(observation) == 2: return observation
        
        processed_observation = super(ContractProcessorWithActionHistory, self).process_observation(observation)

        return processed_observation, self._action_history  # saves storage in experience memory

    def process_experience(self, observation, action, reward, done):
        super(ContractProcessorWithActionHistory, self).process_experience(observation, action, reward, done)
        observation = self.process_observation(observation)
        return observation, action, reward, done

    def process_state_batch(self, batch):
        batch0 = []
        batch1 = []
        for b in batch:
            b0 = b[:, 0]
            b0 = np.array(b0.tolist())
            b0 = b0.astype('float32') / 255.
            batch0 += [b0]

            b1 = b[:, 1][-1]
            b1 = b1.reshape(-1, b1.shape[-1])
            batch1 += [b1]

        batch0 = np.array(batch0)
        batch1 = np.array(batch1)

        return [batch0, batch1]
