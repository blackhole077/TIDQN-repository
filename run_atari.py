import argparse
import atari

parser = argparse.ArgumentParser()
parser.add_argument('--env-name',
                    type=str,
                    dest='environment',
                    choices=['BreakoutDeterministic-v4', 'SpaceInvadersDeterministic-v4', 'SeaquestDeterministic-v4', 'doom'])
parser.add_argument('--task',
                    type=str,
                    dest='task',
                    help="Choose if you are training a model, or testing one.",
                    choices=['train', 'test'],
                    required=True)
parser.add_argument('--contract',
                    type=str,
                    dest='contract',
                    choices=['dithering', 'actuation', 'upto4'])
parser.add_argument('--doom_scenario',
                    type=str,
                    dest='doom_scenario',
                    choices=['DoomBasic-v0', 'DoomCorridor-v0',
                             'DoomDefendCenter-v0', 'DoomDefendLine-v0',
                             'DoomHealthGathering-v0', 'DoomMyWayHome-v0',
                             'DoomPredictPosition-v0', 'DoomTakeCover-v0'],
                    help="If using the VizDoom environment, which scenario should be utilized?",
                    default=None)
parser.add_argument('--arch',
                    type=str,
                    dest='architecture',
                    choices=['original', 'rajagopal_processor', 'option_heads'],
                    help="Choose which architecture you wish to utilize.\
                        By default, 'original' (Vanilla DQN) is used.",
                    default='original')
parser.add_argument('--train_seed',
                    type=int,
                    dest='train_seed',
                    help="What seed should be used during training.")
parser.add_argument('--test_seed',
                    type=int,
                    dest='test_seed',
                    help="What seed should be used during testing.")
parser.add_argument('--steps',
                    type=int,
                    dest='steps',
                    help="How many time steps should the model run?\
                        By default, it will run for 10M steps.\
                        This is done for both training and testing.",
                    default=1000000)
parser.add_argument('--reward',
                    type=float,
                    dest='reward_signal',
                    help='How much reward will be set for rescuing a diver?',
                    default=1.0)
parser.add_argument('--enforce_contract',
                    type=bool,
                    dest='enforce_contract',
                    default=False)

parser.add_argument('--use_state_augmentation',
                    dest='use_state_augmentation',
                    action='store_true')

parser.add_argument('--use_action_shaping',
                    dest='use_action_shaping',
                    action='store_true')

parser.add_argument('--use_aux_extractor',
                    dest='use_aux_extractor',
                    action='store_true')

# Logging-related arguments. Can be left as default.
parser.add_argument('--log_root',
                    type=str,
                    dest='log_directory',
                    default='./slurmlogs/')
parser.add_argument('--weights_root',
                    type=str,
                    dest='weights_directory',
                    default='./weights/')
parser.add_argument('--weights_suffix',
                    type=str,
                    dest='weights_suffix',
                    default='_weights.h5f')
args = vars(parser.parse_args())


arch = args.get("architecture")
if arch == 'original':
    # arch = 'contract'
    mode = 'off'
else:
    mode = 'punish'

if args.get("task") =='train':
    atari.train(args)
else:
    atari.test(args)
