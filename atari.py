import os
import argparse
import yaml

import numpy as np
import pandas as pd
import gym
import tensorflow as tf

import models.atari_model as atari_model
import models.merged_model as merged_model
# import processors.contract_processor as contract_processor
# import processors.stateful_contract_processor as stateful_contract_processor
# import processors.action_history_contract_processor as action_history_contract_processor
# import processors.graph_emb_processor as graph_emb_processor
import processors.rajagopal_processor as rajagopal_processor
import processors.atari_processor as atari_processor
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from callbacks.contract_callbacks import ContractLogger
from rl.callbacks import FileLogger, RajagopalFileLogger
from gym import wrappers
# from envs.doom import make_env as make_doom_env

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

ENFORCE_CONTRACT = True
enforce_contract = ENFORCE_CONTRACT
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

VISUALIZE = False
VERBOSE = 1

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--env-name', type=str,)
parser.add_argument('--contract', type=str)
parser.add_argument('--architecture', type=str)
parser.add_argument('--contract-mode', type=str, choices=['off', 'punish', 'halt'])
parser.add_argument('--steps', type=int)
parser.add_argument('--train_seed', type=int)
parser.add_argument('--test_seed', type=int)
parser.add_argument('--emb', type=str)
parser.add_argument('--enforce_contract', type=bool, default=False)
parser.add_argument('--doom_scenario', type=str)
parser.add_argument('--log_root', type=str, default='./slurmlogs/')
parser.add_argument('--weights_root', type=str, default='./weights/')
parser.add_argument('--weights_suffix', type=str, default='_weights.h5f')

args = parser.parse_args()


def filename_prefix_fn(env_name, contract, architecture, contract_mode, steps, train_seed, test_seed=None):
    root = 'env={}-c={}-arc={}-mode={}-ns={}-seed={}'.format(env_name, contract, architecture, contract_mode, steps, train_seed)
    if test_seed is not None:
        root += '-test_seed=' + str(test_seed)
    return root


def build_dqn(env_name, contract, architecture, contract_mode, steps, nb_actions, emb, enforce_contract, log_prefix, testing=False):
    # map from contract name to regex
    # config = yaml.load(open('./pipeline/config.yaml'))
    # contract = config[env_name][contract]['regular']
    print('ARCHITECTURE: {}'.format(architecture))
    number_conditionals = 3
    if architecture == 'original':
        processor = atari_processor.AtariProcessor(testing=testing)
        model = atari_model.atari_model(INPUT_SHAPE, WINDOW_LENGTH, nb_actions)
    elif architecture == 'rajagopal_processor':
        # 3.) DFA STATE NETWORK USING ONE-HOT
        # Model=merged model; Processor=contract processor
        processor = rajagopal_processor.RajagopalProcessor(
            nb_conditional=number_conditionals,
            testing=testing
        )
        cond_input_shape = (WINDOW_LENGTH, number_conditionals)
        model = merged_model.merged_model(INPUT_SHAPE, WINDOW_LENGTH, nb_actions, cond_input_shape)
    else:
        assert False, 'unknown architecture'

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=.1,
        value_test=.05,
        nb_steps=steps)

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    return dqn


def build_env(env_name, scenario):
    if env_name == 'doom' and scenario is not None:
        # env = make_doom_env(scenario = scenario, grayscale=False, input_shape=INPUT_SHAPE)
        pass
    else:
        env = gym.make(env_name)
    return env

def train(env_name=None, contract=None, architecture=None, contract_mode=None, steps=None, train_seed=None, emb=None, enforce_contract=None, doom_scenario=None):
    env = build_env(env_name, doom_scenario)
    np.random.seed(train_seed)
    env.seed(train_seed)
    nb_actions = env.action_space.n

    filename_prefix = filename_prefix_fn(env_name, contract, architecture, contract_mode, steps, train_seed, None)
    log_prefix = os.path.join(args.log_root, filename_prefix)
    weights_prefix = os.path.join(args.weights_root, filename_prefix)

    dqn = build_dqn(env_name, contract, architecture, contract_mode, steps, nb_actions, None, enforce_contract, log_prefix)

    callbacks = [FileLogger(log_prefix + '_information.json', 1), RajagopalFileLogger(log_prefix + '_info_dict_information.json', 1)]
    # callbacks = [ContractLogger(log_prefix + '_callback.csv'), RajagopalTrainIntervalLogger(log_prefix + '_information.csv', 10000)]
    dqn.fit(
        env,
        nb_steps=steps,
        log_interval=10000,
        visualize=VISUALIZE,
        verbose=VERBOSE,
        callbacks=callbacks)
    weights_filename = weights_prefix + '_weights.h5f'
    return dqn.save_weights(weights_filename, overwrite=True)


def test(env_name=None, contract=None, architecture=None, contract_mode=None, steps=None, train_seed=None, test_seed=None, emb=None, enforce_contract=None, doom_scenario=None, weights_suffix='_weights.h5f'):
    log_prefix = os.path.join(args.log_root, filename_prefix_fn(env_name, contract, architecture, contract_mode, steps, train_seed, test_seed))
    env = build_env(env_name, doom_scenario)
    env = wrappers.Monitor(env, log_prefix + '_video_test')
    np.random.seed(test_seed)
    env.seed(test_seed)
    nb_actions = env.action_space.n

    log_prefix = os.path.join(args.log_root, filename_prefix_fn(env_name, contract, architecture, contract_mode, steps, train_seed, test_seed))
    weights_filename = os.path.join(args.weights_root, filename_prefix_fn(env_name, contract, architecture, contract_mode, steps, train_seed, None)) + weights_suffix

    dqn = build_dqn(env_name, contract, architecture, contract_mode, steps, nb_actions, emb, enforce_contract, log_prefix, True)
    dqn.load_weights(weights_filename)

    callbacks = [ContractLogger(log_prefix + '_callback_test.csv')]
    # callbacks = [FileLogger(log_prefix + '_information_test.json', 1)]

    dqn.test(env, nb_episodes=100, visualize=False, nb_max_start_steps=100, callbacks=callbacks)


if __name__ == '__main__':
    if args.task == 'train':
        train(args.env_name, args.contract, args.architecture, args.contract_mode, args.steps, args.train_seed, args.emb, args.enforce_contract, args.doom_scenario)
    elif args.task == 'test':
        test(args.env_name, args.contract, args.architecture, args.contract_mode, args.steps, args.train_seed, args.test_seed, args.emb, args.enforce_contract, args.doom_scenario, args.weights_suffix)
    else:
        assert False, 'unkown task'
            
    # elif architecture == 'contract_dfa_state':
    #     # 3.) DFA STATE NETWORK USING ONE-HOT
    #     # Model=merged model; Processor=contract processor
    #     processor = stateful_contract_processor.ContractProcessorWithState(
    #         reg_ex = contract,
    #         mode = contract_mode,
    #         log_root = log_prefix,
    #         nb_actions = nb_actions,
    #         enforce_contract = enforce_contract,
    #     )
    #     dfa_input_shape = (
    #         WINDOW_LENGTH,
    #         processor.get_num_states(),
    #     )
    #     model = merged_model.merged_model(INPUT_SHAPE, WINDOW_LENGTH, nb_actions, dfa_input_shape)

    # elif architecture == 'contract_graph_emb':
    #     # 4.) DFA STATE USING NODE2VEC
    #     # Model=graph embedding model; Processor=contract processor
    #     if emb is not None:
    #         emb = pd.read_csv(emb, header=-1, index_col=0, skiprows=1, delimiter=' ')
    #     processor = graph_emb_processor.DFAGraphEmbeddingProcessor(
    #         reg_ex = contract,
    #         mode = contract_mode,
    #         log_root = log_prefix,
    #         nb_actions = nb_actions,
    #         enforce_contract = enforce_contract,
    #         emb = emb
    #     )
    #     model = merged_model.merged_model(INPUT_SHAPE, WINDOW_LENGTH, nb_actions, (WINDOW_LENGTH, emb.shape[1], )
    #     )
    # if architecture == 'contract':
    #     # 1.) BASELINE WITH 0 AUGMENTATION (NO STATE NETWORK)
    #     # Model=default atari model; Processor=contract processor
    #     processor = contract_processor.ContractProcessor(
    #         reg_ex = contract,
    #         mode = contract_mode,
    #         log_root = log_prefix,
    #         nb_actions = nb_actions,
    #         enforce_contract = enforce_contract
    #     )
    #     model = atari_model.atari_model(INPUT_SHAPE, WINDOW_LENGTH, nb_actions)

    # elif architecture == 'contract_action_history':
    #     # 2.) BASELINE WITH CONSTANT ACTION HISTORY NETWORK
    #     # Model=merged model; Processor=contract processor
    #     ACTION_HISTORY_SIZE = 10
    #     processor = action_history_contract_processor.ContractProcessorWithActionHistory(
    #         reg_ex = contract,
    #         mode = contract_mode,
    #         log_root = log_prefix,
    #         nb_actions = nb_actions,
    #         enforce_contract = enforce_contract,
    #         action_history_size = ACTION_HISTORY_SIZE
    #     )
    #     dfa_input_shape = (
    #         ACTION_HISTORY_SIZE,
    #         nb_actions,
    #     )
    #     model = merged_model.merged_model(INPUT_SHAPE, WINDOW_LENGTH, nb_actions, dfa_input_shape)
