import os.path as op
from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial
from psychopy.visual import TextStim, ImageStim
from psychopy.event import waitKeys, getKeys
from psychopy import event

from exptools2 import utils
from psychopy import core

import numpy as np
import glob
import random
import sys
import os

class BlockTrial(Trial):
    """
    Trial with blocks in 3 phases:
    - p0 starts .5 TR before the block, preps ands waits for TR
    - p1 is the stimulus block, flashing snake textures at a certain speed 
    - p2 is the empty block, is of len(stimulus block) - .5 TR
    """
    def __init__(self, session, trial_nr, phase_durations, txt=None, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        self.txt = TextStim(self.session.win, txt)
	
        # get indeces based on amount of session level images
        self.image_idxs = list(range(len(self.session.images)))
        # shuffle them
        random.shuffle(self.image_idxs)
        self.flicker_speed = self.parameters['flicker_speed']
        self.i = 0

        if self.session.debug:
            self.trial_timer = core.Clock()


    def draw(self):
        """ Draws stimuli """
        if self.phase == 1:
            
            if (self.session.flick_timer.getTime()*1000) < self.flicker_speed:
                self.session.images[self.image_idxs[self.i%len(self.session.images)]].draw()
            else:
                # increase counter
                self.i += 1
                # draw new img
                self.session.images[self.image_idxs[self.i%len(self.session.images)]].draw()
                # reset timer
                self.session.flick_timer.reset()
        
        # switch color of fix
        self.session.switch_fix_color()
        # draw fix
        self.session.default_fix.draw()

        # debug message, counting time
        if self.session.debug:
            self.session.debug_message.setText(f"trial {self.trial_nr}, phase {self.phase}, time {self.trial_timer.getTime():.2f}")
            self.session.debug_message.draw()
    
    
    def get_events(self):
        """ Logs responses/triggers """
        events = event.getKeys(timeStamped=self.session.clock)
        
        if events:
            if 'q' in [ev[0] for ev in events]:  # specific key in settings?
                self.session.close()
                self.session.quit()

            for key, t in events:

                if key == self.session.mri_trigger:
                    event_type = 'pulse'

                    if (self.phase == 0) and (self.session.scanner_sync == True):
                
                        self.exit_phase = True

                else:
                    event_type = 'response'
                    # TODO calculate the dt to last fix color switch!

                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = key

                # for param, val in self.parameters.items():
                    # self.session.global_log.loc[idx, param] = val
                for param, val in self.parameters.items():  # add parameters to log
                    if type(val) == np.ndarray or type(val) == list:
                        for i, x in enumerate(val):
                            self.session.global_log.loc[idx, param+'_%4i'%i] = x 
                    else:       
                        self.session.global_log.loc[idx, param] = val

                if self.eyetracker_on:  # send msg to eyetracker
                    msg = f'start_type-{event_type}_trial-{self.trial_nr}_phase-{self.phase}_key-{key}_time-{t}'
                    self.session.tracker.sendMessage(msg)

                #self.trial_log['response_key'][self.phase].append(key)
                #self.trial_log['response_onset'][self.phase].append(t)
                #self.trial_log['response_time'][self.phase].append(t - self.start_trial)

                if key != self.session.mri_trigger:
                    self.last_resp = key
                    self.last_resp_onset = t

        return events


class BlockSession(PylinkEyetrackerSession):
    """ Implements block session"""
    def __init__(self, output_str, output_dir=None, eyetracker_on=True, debug = False, settings_file=None, n_trials=10):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        super().__init__(output_str, output_dir=None, settings_file=settings_file, eyetracker_on=eyetracker_on)

        # unpack a few crucial settings
        self.debug = debug
        self.TR = self.settings['mri']['TR']
        self.n_trials = self.settings['stimuli']['n_blocks']
        self.block_length = self.settings['stimuli']['block_length_TR']
        self.scanner_sync = self.settings['stimuli']['scanner_sync']
        self.flicker_speed = self.settings['stimuli']['flicker_speed']
        x_offset = self.settings['stimuli']['x_offset'] # positive is right, negative is left
        y_offset = self.settings['stimuli']['y_offset'] # positive is up, negative is down

        if self.settings['stimuli']['start_with_phase'] == 2:
            self.total_TRs = self.block_length * 2 + self.n_trials - self.block_length
        else:
            self.total_TRs = self.block_length * 2 + self.n_trials

        # set fix dot params
        self.fix_dot_color_idx = 0
        self.fix_dot_colors = ['green', 'red']
        self.default_fix.setSize(self.settings['task']['fix_dot_size'])
        self.default_fix.setPos((0+x_offset, 0+y_offset))
        self.default_fix.setColor('red') # starting color
        self.total_exp_duration_s = self.total_TRs * self.TR
        self.total_fix_duration = self.total_exp_duration_s
        print(self.total_fix_duration)
        if self.settings['mri']['topup_scan']: 
            self.total_fix_duration += self.settings['mri']['topup_duration']
        print(self.total_fix_duration)


        # init timer for flicking imgs
        self.flick_timer = core.Clock()

        if debug:
            self.debug_message = TextStim(self.win, text = "debug text", pos = (6.0,5.0), height = .3,
                                       opacity = .5) 
        
        # load all images here to avoid excessive memory usage
        self.texture_paths = glob.glob(f"../textures/{self.settings['stimuli']['tex_type']}/density-{self.settings['stimuli']['snake_density']}/*") # get paths to textures
        self.images = [ImageStim(self.win, texture_path,  pos = (0+x_offset,0+y_offset), #size = 10,
                             mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) for texture_path in self.texture_paths]  # proportion that will be blurred

        # self.images = [ImageStim(self.win, texture_path) for texture_path in self.texture_paths]  # proportion that will be blurred

        print(f"loaded {len(self.images)} images at session level")

    def create_trials(self, durations=None, timing='seconds'):
        if durations is None:
            
            if self.scanner_sync == False:
                durations = (self.TR/2, self.TR *self.block_length, (self.TR *self.block_length) - self.TR/2)
            else:
                durations = (100, self.TR *self.block_length, (self.TR *self.block_length) - self.TR/2)
        

        # first trial has no p0, to start the block immediately after first trigger
        if self.settings['stimuli']['start_with_phase'] == 1:
            first_trial_durations=(0, self.TR *self.block_length, (self.TR *self.block_length) - self.TR/2)
        elif self.settings['stimuli']['start_with_phase'] == 2:
            first_trial_durations=(0, 0, (self.TR *self.block_length) - self.TR/2)
        else:
            first_trial_durations=durations

        self.trials = []
        for trial_nr in range(self.n_trials):
            
            if trial_nr == 0 :
                self.trials.append(
                    BlockTrial(session=self,
                            trial_nr=trial_nr,
                            phase_durations=first_trial_durations,
                            txt='Trial %i' % trial_nr,
                            parameters=dict(flicker_speed=self.flicker_speed),
                            #   verbose=True,
                            timing=timing)
                )
            
            else:
                self.trials.append(
                    BlockTrial(session=self,
                            trial_nr=trial_nr,
                            phase_durations=durations,
                            txt='Trial %i' % trial_nr,
                            parameters=dict(flicker_speed=self.flicker_speed),
                            #   verbose=True,
                            timing=timing)
                )

    def _make_fix_dot_color_timings(self, total_time=None):

        if total_time is None:
            total_time = self.total_fix_duration
        # Inspired by Marco's fixation task
        dot_switch_color_times = np.arange(3, total_time, self.settings['task']['color switch interval'])
        # adding randomness
        dot_switch_color_times += (2*np.random.rand(len(dot_switch_color_times))-1) # adding uniform noise [-1, 1] 
        # last one will be total time, ending it all
        dot_switch_color_times[-1] = total_time
        # transforming to frames 
        # dot_switch_color_times = (dot_switch_color_times*120).astype(int)

        return dot_switch_color_times
    
    def switch_fix_color(self, atol = 1e-1):
        """
        change color of default fix
        """
        # print(self.clock.getTime())
        if np.isclose(self.clock.getTime(), self.fix_dot_color_timings[self.fix_dot_color_idx%len(self.fix_dot_color_timings)], atol = atol):
        
            # change color
            self.fix_dot_color_idx += 1
            self.default_fix.setColor(self.fix_dot_colors[self.fix_dot_color_idx % len(self.fix_dot_colors)])


    def end_experiment(self):
        """
        simply takes the fixation task to the end
        """
        finish_fix_task = True
        while finish_fix_task:
            
            self.switch_fix_color()
            # if int(self.clock.getTime()*120) in self.fix_dot_color_timings:
            
            #     # change color
            #     self.fix_dot_color_idx += 1
            #     self.default_fix.setColor(self.fix_dot_colors[self.fix_dot_color_idx % len(self.fix_dot_colors)])

            self.default_fix.draw()
            if self.debug:
                self.debug_message.setText(f"ending fix, time: {self.clock.getTime(): .2f}, last one: {self.fix_dot_color_timings[-1]}\n, topup time to do: {self.settings['mri']['topup_duration']}")
                self.debug_message.draw()

            self.win.flip()

            if int(self.clock.getTime()) > self.fix_dot_color_timings[-1]:
                finish_fix_task = False
        
        return

    def run(self):
        """ Runs experiment. """
        keys = []
        if self.debug:
            self.debug_message.setText("preparing to run, awaiting trigger")
        self.clock.reset()

        # TODO test with eyetracker

        if self.eyetracker_on:
            self.calibrate_eyetracker()
            
            # make fix timings
            self.fix_dot_color_timings = self._make_fix_dot_color_timings()
            if self.debug:
                print(f"created fix timings: {list(self.fix_dot_color_timings)}")
                print(f"created fix timings (s): {[time/120 for time in self.fix_dot_color_timings]}")
            
            while self.settings['mri']['sync'] not in keys:
                keys = getKeys()
                # print(keys)

                # fix switch color needed
                self.switch_fix_color()

                if self.debug:
                    self.debug_message.draw()
                self.default_fix.draw()

                self.win.flip()

            self.start_experiment() # resets global clock as well
            self.start_recording_eyetracker()
        else:
            # make fix timings
            self.fix_dot_color_timings = self._make_fix_dot_color_timings()
            
            if self.debug:
                print(f"created fix timings (f): {list(self.fix_dot_color_timings)}")
                print(f"created fix timings (s): {[time/120 for time in self.fix_dot_color_timings]}")
            
            while self.settings['mri']['sync'] not in keys:
                keys = getKeys()
                # print(keys)

                # fix switch color needed
                self.switch_fix_color()

                if self.debug:
                    self.debug_message.draw()
                self.default_fix.draw()

                self.win.flip()

            self.start_experiment() # resets global clock as well

        for trial in self.trials:
            if self.debug:
                self.debug_message.setText("running trial {}".format(trial.trial_nr))
            
            trial.run()

        # finish task here, instead of dummy trial in the end
        self.end_experiment()

        self.close()

if __name__ == '__main__':

    subject = sys.argv[1] # which subject
    sess =  sys.argv[2] # which session
    task = 'CTS_block_exp' 
    run = sys.argv[3] # which run    
    # output_str = 'sub-{}_sess-{}_task-{}_run-{}'.format(subject.zfill(2), sess.zfill(2), task, run.zfill(2))
    output_str = f'sub-{subject.zfill(2)}_sess-{sess.zfill(2)}_task-{task}_run-{run.zfill(2)}'
    output_dir = f'{task}_pilot/sub-{subject}/ses-{sess}'
    
    # Check if the directory already exists
    if not os.path.exists(output_dir):
        # Create the directory
        os.makedirs(output_dir)
        print("output_dir created successfully!")
    else:
        print("output_dir already exists!")

    settings = op.join(op.dirname(__file__), 'settings_block.yml')
    session = BlockSession(output_str, output_dir=output_dir,# n_trials=2,
                           settings_file=settings, eyetracker_on=False,
                           debug = False)

    session.create_trials()
    #session.create_trials(durations=(3, 3), timing='frames')
    session.run()
    session.quit()
