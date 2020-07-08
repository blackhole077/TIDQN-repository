import argparse
#import luigi
import atari
#import pipeline.train_task as train_task
#import pipeline.test_task as test_task


parser = argparse.ArgumentParser()
parser.add_argument('--env-name',
                    type=str,
                    choices=['BreakoutDeterministic-v4', 'SpaceInvadersDeterministic-v4', 'SeaquestDeterministic-v4', 'doom'])
parser.add_argument('--task',
                    type=str,
                    choices=['train', 'test'])
parser.add_argument('--contract',
                    type=str,
                    choices=['dithering', 'actuation', 'upto4'])
parser.add_argument('--doom_scenario', type=str, default=None,
                    choices=['DoomBasic-v0', 'DoomCorridor-v0',
                             'DoomDefendCenter-v0', 'DoomDefendLine-v0',
                             'DoomHealthGathering-v0', 'DoomMyWayHome-v0',
                             'DoomPredictPosition-v0', 'DoomTakeCover-v0'])
parser.add_argument('--arch',
                    type=str,
                    choices=['original', 'contract', 'contract_action_history', 'contract_dfa_state', 'rajagopal_processor', 'contract_graph_emb'],
                    default='original')
parser.add_argument('--train_seed', type=int)
parser.add_argument('--test_seed', type=int)
parser.add_argument('--steps', type=int, default=1000000)
parser.add_argument('--enforce_contract', type=bool, default=False)
parser.add_argument('--weights_suffix', type=str, default='_weights.h5f')
args = parser.parse_args()


arch = args.arch
if arch == 'original':
    # arch = 'contract'
    mode = 'off'
else:
    mode = 'punish'

if args.task=='train':
    atari.train(env_name=args.env_name, contract=args.contract, steps=args.steps, architecture=arch, contract_mode=mode, train_seed=args.train_seed, enforce_contract=args.enforce_contract, doom_scenario=args.doom_scenario)
else:
    # def test(env_name, contract, architecture, contract_mode, steps, train_seed, test_seed, emb, enforce_contract, doom_scenario):
    atari.test(env_name=args.env_name, contract=args.contract, steps=args.steps, architecture=arch, contract_mode=mode, train_seed=args.train_seed, test_seed=args.test_seed, enforce_contract=args.enforce_contract, doom_scenario=args.doom_scenario, weights_suffix=args.weights_suffix)
#if args.task=='train':
#    tasks = [train_task.TrainTask(env_name=args.env_name, contract=args.contract, steps=args.steps, architecture=arch, contract_mode=mode, train_seed=args.train_seed, enforce_contract=args.enforce_contract, doom_scenario=args.doom_scenario)]
#else:
#    tasks = [test_task.TestTask(env_name=args.env_name, contract=args.contract, steps=args.steps, architecture=arch, contract_mode=mode, train_seed=args.train_seed, test_seed=args.test_seed, enforce_contract=args.enforce_contract, doom_scenario=args.doom_scenario)]

# luigi.build(tasks, local_scheduler=True)

# python run_atari.py --task=train --env-name BreakoutDeterministic-v4 --contract dithering --arch contrac_dfa_state --train_seed 1 --enforce_contract True

# python run_atari.py --task=train --env-name doom --contract dithering --arch contract_dfa_state --train_seed 1  --doom_scenario DoomCorridor-v0
