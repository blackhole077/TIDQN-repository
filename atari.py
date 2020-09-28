import argparse
import os

import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from gym import wrappers
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam

import models.atari_model as atari_model
import models.merged_model as merged_model
import models.diver_model as diver_model
import processors.atari_processor as atari_processor
import processors.rajagopal_processor as rajagopal_processor
from callbacks.contract_callbacks import ContractLogger
from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger, RajagopalFileLogger
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

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

def filename_prefix_fn(env_name, contract, architecture, contract_mode, steps, train_seed, test_seed=None):
    root = 'env={}-c={}-arc={}-mode={}-ns={}-seed={}'.format(env_name, contract, architecture, contract_mode, steps, train_seed)
    if test_seed is not None:
        root += '-test_seed=' + str(test_seed)
    return root


def build_dqn(env_name, architecture, steps, nb_actions, dqn_arguments, testing=False):
    print('ARCHITECTURE: {}'.format(architecture))
    number_conditionals = 7
    if architecture == 'original':
        processor = atari_processor.AtariProcessor(testing=testing)
        model = atari_model.atari_model(INPUT_SHAPE, WINDOW_LENGTH, nb_actions)
    elif architecture == 'rajagopal_processor':
        processor = rajagopal_processor.RajagopalProcessor(
            nb_conditional=number_conditionals,
            testing=testing,
            base_diver_reward=dqn_arguments.get('reward_signal')
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
    """
        Build the environment for the model to use.

        Builds the environment specified by the user arguments.
        Currently only supports creating environments that exist
        in OpenAI Gym, namely the Atari Learning Environment (ALE).
    """
    if env_name == 'doom' and scenario is not None:
        # env = make_doom_env(scenario = scenario, grayscale=False, input_shape=INPUT_SHAPE)
        pass
    else:
        env = gym.make(env_name)
    return env

def generate_filename_prefix(atari_arguments=None, prefix_type=None):
    if atari_arguments is None:
        raise ValueError("No arguments provided.")
    root = 'env={}-c={}-arc={}-mode={}-ns={}-seed={}-r={}'.format(atari_arguments.get("environment"),
                                                            atari_arguments.get("contract"),
                                                            atari_arguments.get("architecture"),
                                                            atari_arguments.get("contract_mode"),
                                                            atari_arguments.get("steps"),
                                                            atari_arguments.get("train_seed"),
                                                            atari_arguments.get("reward_signal"))
    if prefix_type == 'log':
        if atari_arguments.get("test_seed") is not None:
            root += '-test_seed=' + str(atari_arguments.get("test_seed"))
    elif prefix_type == 'weight':
        if atari_arguments.get("weights_suffix") is not None:
            root += atari_arguments.get("weights_suffix")
    else:
        raise ValueError("Expected either 'log' or 'weight', got {}\n".format(prefix_type))
    return root


def train(atari_arguments=None):
    if atari_arguments is None:
        raise ValueError("No arguments provided.")
    # Build the environment specified by the user.
    env = build_env(atari_arguments.get("environment"), atari_arguments.get("doom_scenario"))
    # Seed the random number generator (RNG) for both Numpy and the environment
    np.random.seed(atari_arguments.get("train_seed"))
    env.seed(atari_arguments.get("train_seed"))
    # Determine the number of actions available to the agent
    nb_actions = env.action_space.n

    # Construct the file name prefix for saving all things related to the model
    log_location = os.path.join(atari_arguments.get("log_directory"), generate_filename_prefix(atari_arguments, 'log'))
    weights_location = os.path.join(atari_arguments.get("weights_directory"), generate_filename_prefix(atari_arguments, 'weight'))

    # Build the DQN model provided the parameters
    dqn_model = build_dqn(atari_arguments.get("environment"), atari_arguments.get("architecture"), atari_arguments.get("steps"), nb_actions, atari_arguments, testing=False)
    
    # Define the callback functions
    callbacks = [FileLogger(log_location + '_information.json', 1), RajagopalFileLogger(log_location + '_info_dict_information.json', 1)]
    
    dqn_model.fit(
        env,
        nb_steps=atari_arguments.get("steps"),
        log_interval=10000,
        visualize=VISUALIZE,
        verbose=VERBOSE,
        callbacks=callbacks)
    
    return dqn_model.save_weights(weights_location, overwrite=True)

def test(atari_arguments=None):
    if atari_arguments is None:
        raise ValueError("No arguments provided.")

    # Construct the file name prefix for saving all things related to the model
    log_location = os.path.join(atari_arguments.get("log_directory"), generate_filename_prefix(atari_arguments, 'log'))
    weights_location = os.path.join(atari_arguments.get("weights_directory"), generate_filename_prefix(atari_arguments, 'weight'))

    # Build the environment specified by the user.
    env = build_env(atari_arguments.get("environment"), atari_arguments.get("doom_scenario"))
    # Start up a Monitor that will record videos of the gameplay for empirical analysis.
    env = wrappers.Monitor(env, log_location + '_video_test')
    # Seed the random number generator (RNG) for both Numpy and the environment
    np.random.seed(atari_arguments.get("test_seed"))
    env.seed(atari_arguments.get("test_seed"))
    # Determine the number of actions available to the agent
    nb_actions = env.action_space.n
    # Build the DQN model provided the parameters
    dqn_model = build_dqn(atari_arguments.get("environment"), atari_arguments.get("architecture"), atari_arguments.get("steps"), nb_actions, atari_arguments, testing=True)
    # Load the saved weights
    dqn_model.load_weights(weights_location)
    # Define the callback functions
    callbacks = [ContractLogger(log_location + '_callback_test.csv')]

    dqn_model.test(env, nb_episodes=100, visualize=False, nb_max_start_steps=100, callbacks=callbacks)
