from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial
from exptools2.stimuli import create_circle_fixation

from psychopy.visual import TextStim, ImageStim, Circle
from psychopy import event
from psychopy import visual, core
from psychopy import sound

import psychtoolbox as ptb

import random
import pyglet
import pandas as pd
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools

class TempIntTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, txt=None, parameters=None, preschedule = False, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        self.txt = TextStim(self.session.win, txt) 
        # get the 1/f texture
        # self.img = ImageStim(self.session.win, self.parameters['oneOverF_texture_path'], size = 10,
        #                      mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred
        self.img = ImageStim(self.session.win, f"../textures/{self.session.settings['stimuli']['tex_type']}/density-4/snake_bp-3_density-4_id-0_new.bmp", size = 10,
                             mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred
        self.blank = TextStim(self.session.win, '')
        self.sound_played = False
        self.parameters = parameters
        self.preschedule = preschedule
        self.draw_visual = False if self.parameters['order'] == 'AV' else True
        self.frames_drawn_visual = 0 # counter to draw visual stimulus

        # set frame index of start second stimulus: duration first + soa - 1 for proper indexing
        self.start_second_stimulus_frame = self.session.stim_dur + self.parameters['soa'] - 1

        print(f'initialized trial {trial_nr} with durations {phase_durations} and parameters {parameters}')

    def draw(self):
        """ Draws stimuli 
        potentially, the sound_played flag can be removed now as we are only playing on one specific frame
        """

        # debug message
        if self.session.debug:
            self.session.debug_message.setText(f"trial {self.trial_nr}, phase {self.phase}\norder {self.parameters['order']}, soa {self.parameters['soa']}")
            self.session.debug_message.draw()

        if self.phase == 0:
            # jittered blank
            self.session.default_fix.draw()

            if (self.parameters["order"] == "AV") and (self.preschedule == True):
                # on last flip of p1
                if self.phase_durations[0]-1 == self.session.on_phase_frame:
                    # get next flip time
                    nextFlip = self.session.win.getFutureFlipTime(clock='ptb')
                    # schedule sound
                    self.session.sound.play(when=nextFlip)
                    self.sound_played = True

        elif self.phase == 1:
            if self.preschedule == True:
                if self.parameters["order"] == "VA":
                    # visual first
                    if self.draw_visual:
                        self.img.draw()

                        self.frames_drawn_visual += 1
                        # print(self.frames_drawn_visual)
                        if self.frames_drawn_visual == self.session.stim_dur:
                            self.draw_visual = False
                    
                    # audio second
                    # get the frame before audio should play
                    if self.start_second_stimulus_frame - 1 == self.session.on_phase_frame:
                        if not self.sound_played:
                            # schedule
                            # get next flip time
                            nextFlip = self.session.win.getFutureFlipTime(clock='ptb')
                            # schedule sound
                            self.session.sound.play(when=nextFlip)
                            self.sound_played = True

                elif self.parameters["order"] == "AV":
                    # audio first, should already play
                    
                    # visual second
                    if self.start_second_stimulus_frame == self.session.on_phase_frame:
                        self.draw_visual = True
                    
                    if self.draw_visual:
                        self.img.draw()
                        self.frames_drawn_visual += 1

                        if self.frames_drawn_visual == self.session.stim_dur:
                            self.draw_visual = False

                else:
                    raise ValueError(f"The only supported stimulus orders are 'VA' and 'AV'. You requested {self.parameters['order']}")
            
            else:
                if self.parameters["order"] == "VA":
                    # visual first
                    if self.draw_visual:
                        self.img.draw()
                        self.frames_drawn_visual += 1
                        # print(self.frames_drawn_visual)
                        if self.frames_drawn_visual == self.session.stim_dur:
                            self.draw_visual = False
                    
                    # audio second
                    if self.start_second_stimulus_frame == self.session.on_phase_frame:
                        if not self.sound_played:
                            # print('play sound')
                            # now = ptb.GetSecs()
                            # self.mySound.play(when=now+4)  # play in EXACTLY 4s
                            self.session.sound.play()
                            self.sound_played = True


                elif self.parameters["order"] == "AV":
                    # audio first
                    if not self.sound_played:
                        # print('play sound')
                        # now = ptb.GetSecs()
                        # self.mySound.play(when=now+4)  # play in EXACTLY 4s
                        self.session.sound.play()
                        self.sound_played = True
                    
                    # visual second
                    if self.start_second_stimulus_frame == self.session.on_phase_frame:
                        self.draw_visual = True
                    
                    if self.draw_visual:
                        self.img.draw()
                        self.frames_drawn_visual += 1

                        if self.frames_drawn_visual == self.session.stim_dur:
                            self.draw_visual = False

                else:
                    raise ValueError(f"The only supported stimulus orders are 'VA' and 'AV'. You requested {self.parameters['order']}")
            # always draw default fix
            self.session.default_fix.draw()
                
        elif self.phase == 2:
            # response
            self.session.green_fix.draw()
          

    def get_events(self):
        # TODO implement handling of responses
        # events = super().get_events()
        
        ## code potentially useful for tutorial
        # if (self.phase == 0) and (self.trial_nr == 0):
        #     for key, t in events:
        #         if key in ['space']:
        #             self.stop_phase()

        events = event.getKeys(timeStamped=self.session.clock)
        if events:
            if 'q' in [ev[0] for ev in events]:  # specific key in settings?
                self.session.close()
                self.session.quit()

            for key, t in events:

                if key == self.session.mri_trigger:
                    event_type = 'pulse'
                else:
                    event_type = 'response'

                print(self.phase)
                if self.phase == 2:
                    print(f"t is: {t}")
                    # some logic depending on the task and response
                    if self.session.settings['task']['type'] == 'TOJ':
                        # temporal order judgement: responses are a/v first
                        if key == self.session.response_button_mapping['audio_first']:
                            # pp indicated audio first
                            if self.parameters['order'] == 'AV':
                                # correct
                                self.parameters['correct'] = 1
                                self.parameters['response'] = 'A'
                                
                                
                                # enter into global log
                                idx = self.session.global_log.shape[0]
                                # get start of response window
                                resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                                (self.session.global_log['event_type']=='response_window'), 'onset'].to_numpy()

                                RT = t - resp_onset
                                self.parameters['RT'] = RT
                                print(f"rt is: {RT}, resp_onset is {resp_onset}")

                                # self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                                # self.session.global_log.loc[idx, 'onset'] = t
                                # self.session.global_log.loc[idx, 'event_type'] = 'response'
                                # self.session.global_log.loc[idx, 'phase'] = self.phase
                                # self.session.global_log.loc[idx, 'key'] = key
                                # self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                                # self.session.global_log.loc[idx, 'response'] = self.parameters['response']
                                # self.session.global_log.loc[idx, 'RT'] = RT

                                # self.session.global_log.loc[idx, 'correct'] = self.parameters['correct']
                                
                                # stop phase, ending response window
                                self.stop_phase()

                            else:
                                # incorrect
                                self.parameters['correct'] = 0
                                self.parameters['response'] = 'A'
                                
                                # enter into global log
                                idx = self.session.global_log.shape[0]
                                # get start of response window
                                resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                                (self.session.global_log['event_type']=='response_window'), 'onset'].to_numpy()

                                RT = t - resp_onset
                                self.parameters['RT'] = RT
                                print(f"rt is: {RT}, resp_onset is {resp_onset}")

                                # self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                                # self.session.global_log.loc[idx, 'onset'] = t
                                # self.session.global_log.loc[idx, 'event_type'] = 'response'
                                # self.session.global_log.loc[idx, 'phase'] = self.phase
                                # self.session.global_log.loc[idx, 'key'] = key
                                # self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                                # self.session.global_log.loc[idx, 'response'] = self.parameters['response']
                                # self.session.global_log.loc[idx, 'RT'] = RT

                                # self.session.global_log.loc[idx, 'correct'] = self.parameters['correct']
                                
                                # stop phase, ending response window
                                self.stop_phase()

                        elif key == self.session.response_button_mapping['visual_first']:
                            # pp indicated visual first
                            if self.parameters['order'] == 'VA':
                                # correct
                                self.parameters['correct'] = 1
                                self.parameters['response'] = 'V'

                                # enter into global log
                                idx = self.session.global_log.shape[0]
                                # get start of response window
                                resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                                (self.session.global_log['event_type']=='response_window'), 'onset'].to_numpy()

                                RT = t - resp_onset
                                self.parameters['RT'] = RT
                                print(f"rt is: {RT}, resp_onset is {resp_onset}")

                                # self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                                # self.session.global_log.loc[idx, 'onset'] = t
                                # self.session.global_log.loc[idx, 'event_type'] = 'response'
                                # self.session.global_log.loc[idx, 'phase'] = self.phase
                                # self.session.global_log.loc[idx, 'key'] = key
                                # self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                                # self.session.global_log.loc[idx, 'response'] = self.parameters['response']
                                # self.session.global_log.loc[idx, 'RT'] = RT
                                
                                # stop phase, ending response window
                                self.stop_phase()

                            else:
                                # incorrect
                                self.parameters['correct'] = 0
                                self.parameters['response'] = 'V'

                                # enter into global log
                                idx = self.session.global_log.shape[0]
                                # get start of response window
                                resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                                (self.session.global_log['event_type']=='response_window'), 'onset'].to_numpy()

                                RT = t - resp_onset
                                self.parameters['RT'] = RT
                                print(f"rt is: {RT}, resp_onset is {resp_onset}")

                                # self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                                # self.session.global_log.loc[idx, 'onset'] = t
                                # self.session.global_log.loc[idx, 'event_type'] = 'response'
                                # self.session.global_log.loc[idx, 'phase'] = self.phase
                                # self.session.global_log.loc[idx, 'key'] = key
                                # self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                                # self.session.global_log.loc[idx, 'response'] = self.parameters['response']
                                # self.session.global_log.loc[idx, 'RT'] = RT

                                # stop phase, ending response window
                                self.stop_phase()

                    if self.session.settings['task']['type'] == 'SJ':
                        # temporal order judgement: responses are a/v first
                        if key == self.session.response_button_mapping['synchronous']:
                            # pp indicated synchonous
                            self.parameters['response'] = 'synchronous'

                            # enter into global log
                            idx = self.session.global_log.shape[0]
                            # get start of response window
                            resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                            (self.session.global_log['event_type']=='response_window'), 'onset'].to_numpy()

                            RT = t - resp_onset
                            self.parameters['RT'] = RT
                            print(f"rt is: {RT}, resp_onset is {resp_onset}")

                            # self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                            # self.session.global_log.loc[idx, 'onset'] = t
                            # self.session.global_log.loc[idx, 'event_type'] = 'response'
                            # self.session.global_log.loc[idx, 'phase'] = self.phase
                            # self.session.global_log.loc[idx, 'key'] = key
                            # self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                            # self.session.global_log.loc[idx, 'response'] = self.parameters['response']
                            # self.session.global_log.loc[idx, 'RT'] = RT

                            # stop phase, ending response window
                            self.stop_phase()

                        elif key == self.session.response_button_mapping['asynchronous']:
                            # pp indicated asynchronous
                            self.parameters['response'] = 'asynchronous'

                    
                            # enter into global log
                            idx = self.session.global_log.shape[0]
                            # get start of response window
                            resp_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.trial_nr) & \
                                                                            (self.session.global_log['event_type']=='response_window'), 'onset'].to_numpy()

                            RT = t - resp_onset
                            self.parameters['RT'] = RT
                            print(f"rt is: {RT}, resp_onset is {resp_onset}")


                            # self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                            # self.session.global_log.loc[idx, 'onset'] = t
                            # self.session.global_log.loc[idx, 'event_type'] = 'response'
                            # self.session.global_log.loc[idx, 'phase'] = self.phase
                            # self.session.global_log.loc[idx, 'key'] = key
                            # self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                            # self.session.global_log.loc[idx, 'response'] = self.parameters['response']
                            # self.session.global_log.loc[idx, 'RT'] = RT
                                
                            # stop phase, ending response window
                            self.stop_phase()
                
                # else:
                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'key'] = key

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

                if key != self.session.mri_trigger:
                    self.last_resp = key
                    self.last_resp_onset = t

        return events

    def run(self):
        """ Runs through phases. Should not be subclassed unless
        really necessary. """

        if self.eyetracker_on:  # Sets status message
            cmd = f"record_status_message 'trial {self.trial_nr}'"
            self.session.tracker.sendCommand(cmd)

        # Because the first flip happens when the experiment starts,
        # we need to compensate for this during the first trial/phase
        if self.session.first_trial:
            # must be first trial/phase
            if self.timing == 'seconds':  # subtract duration of one frame
                self.phase_durations[0] -= 1./self.session.actual_framerate * 1.1  # +10% to be sure
            else:  # if timing == 'frames', subtract one frame 
                self.phase_durations[0] -= 1
            
            self.session.first_trial = False

        for phase_dur in self.phase_durations:  # loop over phase durations
            # pass self.phase *now* instead of while logging the phase info.
            self.session.win.callOnFlip(self.log_phase_info, phase=self.phase)
            self.session.on_phase_frame = 0

            # Start loading in next trial during this phase (if not None)
            if self.load_next_during_phase == self.phase:
                self.load_next_trial(phase_dur)

            if self.timing == 'seconds':
                # Loop until timer is at 0!
                self.session.timer.add(phase_dur)
                while self.session.timer.getTime() < 0 and not self.exit_phase and not self.exit_trial:
                    self.draw()
                    if self.draw_each_frame:
                        self.session.win.flip()
                        self.session.nr_frames += 1
                    self.get_events()
            else:
                # Loop for a predetermined number of frames
                # Note: only works when you're sure you're not 
                # dropping frames
                for _ in range(phase_dur):

                    if self.exit_phase or self.exit_trial:
                        break

                    self.draw()
                    self.session.win.flip()
                    self.get_events()
                    self.session.nr_frames += 1
                    self.session.on_phase_frame += 1


            if self.exit_phase:  # broke out of phase loop
                self.session.timer.reset()  # reset timer!
                self.exit_phase = False  # reset exit_phase
            if self.exit_trial:
                self.session.timer.reset()
                break

            self.phase += 1  # advance phase

class TempIntSession(PylinkEyetrackerSession):
    """ Simple session with x trials. """

    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10, eyetracker_on=True, debug = False):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        super().__init__(output_str, output_dir=output_dir,
                         settings_file=settings_file, eyetracker_on=eyetracker_on)
        
        self.green_fix = Circle(self.win, radius=.075, edges = 100, lineWidth=0)
        self.green_fix.setColor((0, 128, 0), 'rgb255')
        self.black_fix = Circle(self.win, radius=.075, edges = 100, lineWidth=0)
        self.black_fix.setColor((0, 0, 0), 'rgb255')

        # get stimulus params
        self.conds = self.settings['stimuli']['stim_conds']
        self.n_conds = len(self.conds)
        self.n_repeats = self.settings['stimuli']['n_repeats']
        self.stim_dur = self.settings['stimuli']['stim_dur']
        self.stim_onset_asynch = self.settings['stimuli']['stim_onset_asynch']
        self.conds_tuple = list(itertools.product(self.conds, self.stim_onset_asynch))
        try:
            self.conds_tuple.remove(('AV',0)) # removing one of the simultaneous conds, as we only need one
        except ValueError:
            print("No (AV, 0) condition specified. Please check if this is intended.")

        # make sound
        self.sound = sound.Sound(secs = self.stim_dur/120)

        # set phase frame counter
        self.on_phase_frame = 0
        # set phase names for logs 
        self.phase_names = ['ITI','stim','response_window']

        if n_trials is None:
            self.n_trials = self.n_repeats * len(self.conds_tuple)

        self.debug = debug
        # setting a debug message
        if self.debug:
            self.debug_message = TextStim(self.win, text = "debug text", pos = (6.0,5.0), height = .3,
                                       opacity = .5) 

        # make response button mapping
        if self.settings['task']['response_device'] == 'keyboard':
            
            if self.settings['task']['type'] == 'SJ':
                # synchrony judgement mapping
                self.response_button_mapping = {'synchronous' : self.settings['task']['response_keys'][0],
                                            'asynchronous' : self.settings['task']['response_keys'][1]}
            elif self.settings['task']['type'] == 'TOJ':
                self.response_button_mapping = {'audio_first' : self.settings['task']['response_keys'][0],
                                            'visual_first' : self.settings['task']['response_keys'][1]}
            else:
                raise ValueError(f"{self.settings['task']['type']} is not supported as a task")
            
        elif self.settings['task']['response_device'] == 'button_box':
            raise NotImplementedError("Button box not implemented yet")
            # self.response_button_mapping = {'present_confident' : '2',
            #                                 'present_not_confident' : '4',
            #                                 'absent_confident' : '1',
            #                                 'absent_not_confident' : '3'}

        # init result lists TODO use
        self.response_times = []
        self.target_times = []
        self.target_times_f = []

    def create_trials(self, durations=None, timing='frames', preschedule = False):
        # make trial parameters (AV/VA and SOA)
        all_trial_parameters = self.conds_tuple * self.n_repeats
        # shuffle
        random.shuffle(all_trial_parameters)
        print(all_trial_parameters)

        # p0_durs = [120] * self.n_trials # constant
        jits = np.arange(54, 79) # jittered times between 54 and 79 frames, corr. to 425 - 600 ms, as in Yankieva
        p0_durs = [int(jit) for jit in np.random.choice(jits, self.n_trials)]

        # this might not be the right solution, p1 might need to be variable
        # p1_durs = [96] * self.n_trials # fixed duration within which stimulus occurs, 800 ms
        # variable p1
        p1_durs = [self.stim_dur * 2 + params[-1] for params in all_trial_parameters]

        p2_durs = [144] * self.n_trials # fixed duration for answers, 144 frames are 1.2 s

        self.trials = []
        durations = list(zip(p0_durs, p1_durs, p2_durs))
        for trial_nr in range(self.n_trials):
            
            # debug datatypes of durations
            # types = [type(dur) for dur in durations[trial_nr]]
            # print(types)

            self.trials.append(
                TempIntTrial(session=self,
                          trial_nr=trial_nr,
                          phase_durations=durations[trial_nr],
                          phase_names = self.phase_names,
                          txt='Trial %i' % trial_nr,
                          verbose=False,
                          parameters=dict(order=all_trial_parameters[trial_nr][0],
                                          soa=all_trial_parameters[trial_nr][1]),
                          timing=timing,
                          preschedule=preschedule)
            )

    def close(self):
        """'Closes' experiment. Should always be called, even when10
        experiment is quit manually (saves onsets to file)."""

        if self.closed:  # already closed!
            return None

        self.win.callOnFlip(self._set_exp_stop)
        self.win.flip()
        self.win.recordFrameIntervals = False

        print(f"\nDuration experiment: {self.exp_stop:.3f}\n")

        if not op.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.global_log = pd.DataFrame(self.global_log).set_index("trial_nr")
        self.global_log["onset_abs"] = self.global_log["onset"] + self.exp_start

        # Only non-responses have a duration
        nonresp_idx = ~self.global_log.event_type.isin(["response", "trigger", "pulse"])
        last_phase_onset = self.global_log.loc[nonresp_idx, "onset"].iloc[-1]
        dur_last_phase = self.exp_stop - last_phase_onset
        durations = np.append(
            self.global_log.loc[nonresp_idx, "onset"].diff().values[1:], dur_last_phase
        )
        self.global_log.loc[nonresp_idx, "duration"] = durations

        # Same for nr frames
        nr_frames = np.append(
            self.global_log.loc[nonresp_idx, "nr_frames"].values[1:], self.nr_frames
        )
        self.global_log.loc[nonresp_idx, "nr_frames"] = nr_frames.astype(int)

        # Round for readability and save to disk
        self.global_log = self.global_log.round(
            {"onset": 5, "onset_abs": 5, "duration": 5}
        )
        f_out = op.join(self.output_dir, self.output_str + "_events.tsv")
        self.global_log.to_csv(f_out, sep="\t", index=True)

        # Create figure with frametimes (to check for dropped frames)
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(self.win.frameIntervals)
        ax.axhline(1.0 / self.actual_framerate, c="r")
        ax.axhline(
            1.0 / self.actual_framerate + 1.0 / self.actual_framerate, c="r", ls="--"
        )
        ax.set(
            xlim=(0, len(self.win.frameIntervals) + 1),
            xlabel="Frame nr",
            ylabel="Interval (sec.)",
            ylim=(-0.01, 0.125),
        )
        fig.savefig(op.join(self.output_dir, self.output_str + "_frames.pdf"))

        if self.mri_simulator is not None:
            self.mri_simulator.stop()

        # create results df
        results = {'target_times_s' : self.target_times,
                   'target_times_f' : self.target_times_f,
                   'response_times_s' : self.response_times,
                   'response_times_f' : [int(round(time * 120)) for time in self.response_times],
                   }
        
        results_df = pd.DataFrame(results)
        results_df['response_diff_f'] = [frames_target - frames_resp for frames_target, frames_resp in zip(results_df['target_times_f'], results_df['response_times_f'])]

        self.results = results_df

        self.win.close()
        
        if self.eyetracker_on:
            self.stop_recording_eyetracker()
            self.tracker.setOfflineMode()
            core.wait(.5)
            f_out = op.join(self.output_dir, self.output_str + '.edf')
            self.tracker.receiveDataFile(self.edf_name, f_out)
            self.tracker.close()
        
        self.closed = True



    def run(self):
        """ Runs experiment. """
        if self.eyetracker_on:
            self.calibrate_eyetracker()
            self.start_experiment()
            self.start_recording_eyetracker()
        else:
            self.start_experiment()

        for trial in self.trials:
            trial.run()

        self.close()  # contains tracker.stopRecording()


if __name__ == '__main__':

    subject = sys.argv[1]
    sess =  sys.argv[2]
    task = 'TempInt' # different settings -> now implemented as saving the actual settings
    run = sys.argv[3] # which run    
    output_str = f'sub-{subject}_sess-{sess}_task-{task}_run-{run}'
    print(output_str)
    # output_str = 'sub-{}_sess-{}_task-{}_run-{}'.format(subject.zfill(2), sess.zfill(2), task, run.zfill(2))
    # output_str = 'sub-TempIntTest99_sess-TempIntTest_task-TempInt_run-TempIntTest'
    # save results
    results_folder = f'{task}_pilot/sub-{subject}/ses-{sess}'
    # print(results_folder) 

    # Check if the directory already exists
    if not os.path.exists(results_folder):
        # Create the directory
        os.makedirs(results_folder)
        print("results_folder created successfully!")
    else:
        print("results_folder already exists!")

    session = TempIntSession(output_str, output_dir = results_folder,  eyetracker_on=False,
                              n_trials=None, settings_file='settings_TempInt.yml', debug=False)

    print(session.n_conds)
    print(session.n_repeats)
    print(session.stim_onset_asynch)
    print(session.n_trials)
    print(session.conds_tuple)
    print(sound.Sound)

    session.create_trials(preschedule=False)
    session.run()
    # print(session.results)

    
    # results_out = f'{results_folder}/{output_str}_results.csv'
    # session.results.to_csv(results_out, index = False)
    # print(results_out)

