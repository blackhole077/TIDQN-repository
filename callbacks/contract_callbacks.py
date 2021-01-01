from rl.callbacks import Callback
import numpy as np
import timeit
import json

class RajagopalTestLogger(Callback):
    def __init__(self, filepath, interval=1):
        self.filepath = filepath
        self.interval = interval
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dict that maps from episode to metrics array.
        self.episodes = {}

    def on_train_begin(self, logs):
        pass
    def on_train_end(self, logs):
        """ Save model at the end of training """
        self.save_data()

    def on_episode_begin(self, episode, logs):
        """Called at beginning of each episode"""
        self.episodes[episode] = dict()
        self.episodes[episode]['actions'] = []
        self.episodes[episode]['info'] = []
        self._action_list = []
        self._info_list = []
    
    def on_episode_end(self, episode, logs):
        if episode % self.interval == 0:
            logs['total_divers_for_episode'] = self._info_list[-1].get('total_divers')
            logs['max_divers_for_episode'] = self._info_list[-1].get('max_divers_for_episode')
            self.episodes[episode]['logs'] = logs
            self.episodes[episode]['actions'] = self._action_list
            self.episodes[episode]['info'] = self._info_list
            self.save_data()

    def on_step_begin(self, step, logs):
        pass

    def on_step_end(self, step, logs):
        """
            Append metric at the end of each step

            Parameters
            ----------
            step : 

            logs : dict
                Contains the following:
                - action (what action was taken)
                - observation
                - reward
                - info
        """
        for key in logs['info']:
            logs['info'][key] = np.asscalar(logs['info'][key])
        self._info_list += [logs['info']]
        
    def on_action_begin(self, action, logs):
        """Called at beginning of each action"""
        self._action_list.append(int(action))

    def save_data(self):
        """ Save metrics in a json file """
        if len(self.episodes.keys()) == 0:
            return
        # Sort everything by episode.
        # Overwrite already open file. We can simply seek to the beginning since the file will
        # grow strictly monotonously.
        with open(self.filepath, 'w') as f:
            json.dump(self.episodes, f)

class ContractLogger(Callback):
    def __init__(self, filename, interval=1):
        self._interval = interval
        self._file = open(filename, 'w')

        
    def on_episode_begin(self, episode, logs):
        """Called at beginning of each episode"""
        self._action_list = []
        self._info_list = []
        pass

    
    def on_episode_end(self, episode, logs):
        """Called at end of each episode"""
        if episode % self._interval == 0:
            #assert len(self._action_list) == len(self._info_list)
            self._file.write('{}\t{}\t{}\t{}\n'.format(episode, logs, self._action_list, self._info_list))
            self._file.flush()
        pass

    
    def on_step_begin(self, step, logs):
        """Called at beginning of each step"""
        pass

    
    def on_step_end(self, step, logs):
        """Called at end of each step"""
        self._info_list += [logs['info']]

    
    def on_action_begin(self, action, logs):
        """Called at beginning of each action"""
        self._action_list += [action]


    def on_action_end(self, action, logs):
        """Called at end of each action"""
        pass


class RajagopalTrainIntervalLogger(Callback):
    def __init__(self, filename, interval=10000):
        self.interval = interval
        self._file = open(filename, 'w')
        self.step = 0
        self.reset()

    def reset(self):
        """ Reset statistics """
        self.interval_start = timeit.default_timer()
        # self.progbar = Progbar(target=self.interval)
        self.metrics = []
        self.infos = []
        self.info_names = None
        self.episode_rewards = []

    def on_train_begin(self, logs):
        """ Initialize training statistics at beginning of training """
        pass
        # self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        # print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_train_end(self, logs):
        """ Print training duration at end of training """
        pass
        # duration = timeit.default_timer() - self.train_start
        # print('done, took {:.3f} seconds'.format(duration))

    def on_step_begin(self, step, logs):
        """ Print metrics if interval is over """
        if self.step % self.interval == 0:
            if len(self.episode_rewards) > 0:
                metrics = np.array(self.metrics)
                assert metrics.shape == (self.interval, len(self.metrics_names))
                formatted_metrics = ''
                if not np.isnan(metrics).all():  # not all values are means
                    means = np.nanmean(self.metrics, axis=0)
                    assert means.shape == (len(self.metrics_names),)
                    for name, mean in zip(self.metrics_names, means):
                        formatted_metrics += ' - {}: {:.3f}'.format(name, mean)
                
                formatted_infos = ''
                if len(self.infos) > 0:
                    infos = np.array(self.infos)
                    if not np.isnan(infos).all():  # not all values are means
                        means = np.nanmean(self.infos, axis=0)
                        assert means.shape == (len(self.info_names),)
                        for name, mean in zip(self.info_names, means):
                            formatted_infos += ',{}, {:.3f}'.format(name, mean)
                self._file.write('{} episodes - episode_reward: {:.3f} [{:.3f} - {:.3f}],{},{}'.format(len(self.episode_rewards), np.mean(self.episode_rewards), np.min(self.episode_rewards), np.max(self.episode_rewards), formatted_metrics, formatted_infos))
                self._file.flush()
                print('')
            self.reset()
            # print('Interval {} ({} steps performed)'.format(self.step // self.interval + 1, self.step))

    def on_step_end(self, step, logs):
        """ Update progression bar at the end of each step """
        if self.info_names is None:
            self.info_names = logs['info'].keys()
        values = [('reward', logs['reward'])]
        # if KERAS_VERSION > '2.1.3':
        #     self.progbar.update((self.step % self.interval) + 1, values=values)
        # else:
        #     self.progbar.update((self.step % self.interval) + 1, values=values, force=True)
        self.step += 1
        self.metrics.append(logs['metrics'])
        if len(self.info_names) > 0:
            self.infos.append([logs['info'][k] for k in self.info_names])

    def on_episode_end(self, episode, logs):
        """ Update reward value at the end of each episode """
        self.episode_rewards.append(logs['episode_reward'])
