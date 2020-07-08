import numpy as np
import processors.contract_processor as contract_processor

class DFAGraphEmbeddingProcessor(contract_processor.ContractProcessor):

    def __init__(self, reg_ex, mode, log_root, nb_actions, enforce_contract, emb):
        super(DFAGraphEmbeddingProcessor, self).__init__(reg_ex, mode, log_root, nb_actions, enforce_contract)
        self.emb = emb

    def get_num_states(self):
        return self._dfa.num_states()

    def process_observation(self, observation):
        if len(observation) == 2: return observation

        processed_observation = super(DFAGraphEmbeddingProcessor, self).process_observation(observation)

        state_emb = np.array([self.emb.loc[self._dfa._current_state].values]) # converts shape to (1,n)

        return processed_observation, state_emb  # saves storage in experience memory

    def process_experience(self, observation, action, reward, done):
        super(DFAGraphEmbeddingProcessor, self).process_experience(observation, action, reward, done)
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

            b1 = b[:, 1]
            b1 = np.array(b1.tolist())
            temp = []
            for bb in b1:
                temp += [bb.flatten()]
            batch1 += [temp]

        batch0 = np.array(batch0)
        batch1 = np.array(batch1)

        return [batch0, batch1]
