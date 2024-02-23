import os.path as op
from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial
from psychopy.visual import Rect, TextStim, ImageStim
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from psychopy import event

class DelayedNormTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        
        # get the 1/f texture
        self.img = ImageStim(self.session.win, self.parameters['oneOverF_texture_path'],
                             mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred
        self.blank = TextStim(self.session.win, text='') 
        self.check_frames = np.zeros((96, 3))

        # get the stimulus array with self.session.var_isi/dur_dict and save it into self.stimulus_frames
        if self.parameters['trial_type'] == 'dur':
            # self.stimulus_frames = self.session.var_dur_dict[self.parameters['stim_dur']]
            self.stimulus_frames = self.session.var_dur_dict_flip[self.parameters['stim_dur']]

        else:
            # self.stimulus_frames = self.session.var_isi_dict[self.parameters['stim_dur']]
            self.stimulus_frames = self.session.var_isi_dict_flip[self.parameters['stim_dur']]

        # squares for Photodiode
        if self.session.photodiode_check is True:
            self.white_square = Rect(self.session.win, 1, 1, pos = (-10,-8))
            self.black_square = Rect(self.session.win, 1, 1, pos = (-10,-8), fillColor = 'black')


    def draw_flip(self, current_frame):
        """ 
        Draws stimuli and flips the window 
        This function implements a potentially more resourceful drawing behaviour, where frames are only 
        drawn when something on the screen actually changes.
        Otherwise, we simply flip without clearing the buffer, meaning that a blank screen will be flipped on top
        of what is already drawn on the screen.


        This method is still buggy on fullscreen, for undiscovered reasons
        """

        # fixation dot color change
        if int((self.session.clock.getTime()*120)) in self.parameters['dot_color_timings']:
            # change color
            self.session.fix_dot_color_idx += 1
            self.session.default_fix.setColor(self.session.fix_dot_colors[self.session.fix_dot_color_idx % len(self.session.fix_dot_colors)])

        if self.phase == 0: # we are in phase 0, prep time
            
            self.session.default_fix.draw()
            
            if self.photodiode_check is True:
                # draw photodiode square
                self.black_square.draw()

            
            self.session.win.flip()

            # checking if scanner t arrives
            if 't' in event.getKeys(keyList = ['t']):
                self.exit_phase = True

        elif self.phase == 1: # we are in phase 1, stimulus presentation
            
            # if the self.stimulus_frames array at this frame index is 1, draw the stimulus and flip it
            if self.stimulus_frames[current_frame] == 1:
                
                if self.photodiode_check is True:
                    self.white_square.draw()
    
                self.img.draw()
                self.session.default_fix.draw()

                self.session.trialwise_frame_timings[current_frame, self.trial_nr] = self.session.win.flip(clearBuffer = False)

            # if the self.stimulus_frames array at this frame index is -1, draw only fix and flip it
            elif self.stimulus_frames[current_frame] == -1:

                if self.photodiode_check is True:
                    self.black_square.draw()
    
                self.session.default_fix.draw()

                self.session.trialwise_frame_timings[current_frame, self.trial_nr] = self.session.win.flip()

                if self.photodiode_check is True:
                    self.black_square.draw()
    
                self.session.default_fix.draw()

            # if the self.stimulus_frames array at this frame index is neither [1, -1], draw nothing and flip with clearBuffer = False
            else:
                if self.photodiode_check is True:
                    self.black_square.draw()

                # self.session.default_fix.draw()
                self.session.trialwise_frame_timings[current_frame, self.trial_nr] = self.session.win.flip(clearBuffer = False)
            
        
        else: # we are in phase 2, iti

            self.black_square.draw()
            # draw fixation
            self.session.default_fix.draw()
            self.session.win.flip()


    def draw(self):
        """ Draws stimuli 
        This is to be used when flipping on every frame
        """
        # fixation dot color change
        if int((self.session.clock.getTime()*120)) in self.parameters['dot_color_timings']:
            # change color
            self.session.fix_dot_color_idx += 1
            self.session.default_fix.setColor(self.session.fix_dot_colors[self.session.fix_dot_color_idx % len(self.session.fix_dot_colors)])

        if self.phase == 0: # we are in phase 0, prep time

            if self.photodiode_check is True:
                self.black_square.draw()

            self.session.default_fix.draw()
            
            
            if 't' in event.getKeys(keyList = 't'):
                self.exit_phase = True


        elif self.phase == 1: # we are in phase 1, stimulus presentation
            
            ## if the self.frame array at this frame index is one, show the texture, otherwise blank or fix
            if self.stimulus_frames[self.session.nr_frames] == 1:
                
                if self.photodiode_check is True:
                    self.white_square.draw()    

                # draw texture
                self.img.draw()

                # draw fixation
                self.session.default_fix.draw()
                
            else:
                
                if self.photodiode_check is True:
                    self.black_square.draw()

                # draw fixation 
                self.session.default_fix.draw()


        else: # we are in phase 2, iti
            if self.photodiode_check is True:
                self.black_square.draw()
    
            # draw fixation
            self.session.default_fix.draw()


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
            self.session.total_nr_frames += self.session.nr_frames
            self.session.nr_frames = 0
            # pass self.phase *now* instead of while logging the phase info.
            #self.session.win.callOnFlip(self.log_phase_info, phase=self.phase)

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
                # self.flip_counter = 0

                for frame in range(phase_dur):
                    
                    if frame == 0:
                        event.clearEvents()

                    if self.exit_phase or self.exit_trial:
                        break
                    # here, either draw_flip() or (draw() and session.win.flip()) should be used
                    #self.draw_flip(current_frame = frame)
                    
                    self.draw()
                    
                    # keeping track of flip timings
                    if self.phase == 1:
                        self.session.trialwise_frame_timings[frame, self.trial_nr] = self.session.win.flip()
                    else:
                        self.session.win.flip()                
                    
                    #self.get_events()
                    self.session.nr_frames += 1

            if self.exit_phase:  # broke out of phase loop
                self.session.timer.reset()  # reset timer!
                self.exit_phase = False  # reset exit_phase
            if self.exit_trial:
                self.session.timer.reset()
                break

            self.phase += 1  # advance phase        


class DelayedNormSession(PylinkEyetrackerSession):
    """ Simple session with x trials. """
    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10, eyetracker_on=True):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        self.total_nr_frames = 0
        self.fix_dot_color_idx = 0
        self.fix_dot_colors = ['green', 'red']

        self.trialwise_frame_timings = np.zeros((96, n_trials))

        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file, eyetracker_on=eyetracker_on)

    def create_trials(self, timing='frames'):
        self.trials = []
        # 3 phases: prep, ITI and trial
        trial_duration = 96
        frames_TR = int(round(self.settings['mri']['TR']*120))
        full_iti_duration = 3 * frames_TR

        prep_durations = self.n_trials * [120] #[100000] # just a very large duration, as we wait for t # [frames_TR//2] # 1/2 of a TR, setting up and waiting for t # also 1.33/2 will be 79.8 frames, rounding to 80
        iti_durations = self.n_trials * [full_iti_duration - trial_duration - frames_TR//2] # placeholder for TODO implement itis note they will be the time from onset of a stimulus, so it will be ITI - 1/2 TR - 96
        trial_durations = self.n_trials * [trial_duration] # this will be constant TODO confirm length

        # make durations list of tuples for prep, iti, trial
        durations = list(zip(prep_durations, trial_durations, iti_durations))
        print("prep durations: {}".format(prep_durations))
        ## making stimulus arrays
        stim_durations = [0, 2, 4, 8, 16, 32, 64] # frames in 120 FPS, either duration or isi times
        fixed_duration = 16 # fixed duration for isi trials
        self.total_duration = 96 # (<800 ms in total) in exp design; 800 ms = .8*120 = 96 frames
        var_duration = np.vstack([np.hstack((np.ones(duration), # showing stimulus
                                             np.zeros(self.total_duration - duration))) # no stimulus for the remaining frames
                                             for duration in stim_durations])
        
        var_isi = np.vstack([np.hstack((np.ones(fixed_duration), # show stimulus
                                        np.zeros(duration), # isi
                                        np.ones(fixed_duration), # show stimulus again
                                        np.zeros(self.total_duration - duration - 2*fixed_duration))) # no stimulus for remaining frames 
                                        for duration in stim_durations])
        
        # these dicts are integer indexable with the current number of trial frames 
        self.var_isi_dict = {dur:frames for dur, frames in zip(stim_durations, var_isi)}
        self.var_dur_dict = {dur:frames for dur, frames in zip(stim_durations, var_duration)}

        # these are to check the timings        
        self.check_isi_dict = {dur:np.zeros(96) for dur in stim_durations}
        self.check_dur_dict = {dur:np.zeros(96) for dur in stim_durations}


        var_duration_flip = np.zeros((len(stim_durations), self.total_duration))
        for i in range(len(stim_durations)):
            #print(i)
            var_duration_flip[i, 0] = 1 # on
            var_duration_flip[i, stim_durations[i]] = -1 # off
            
            if i == 0:
                var_duration_flip[i, 0] = 0
        
        var_isi_flip = np.zeros((len(stim_durations), self.total_duration))

        for i in range(len(stim_durations)):
            #print(i)
            
            if i == 0:
                var_isi_flip[i, 0] = 1 # on
                var_isi_flip[i, 2 * fixed_duration] = -1 # off
                
            else:
                try:
                    # fixed 16 frames
                    var_isi_flip[i, 0] = 1 # on
                    var_isi_flip[i, 0 + fixed_duration] = -1 # off
                    var_isi_flip[i, 0 + fixed_duration + stim_durations[i]] = 1 # on
                    var_isi_flip[i, 0 + fixed_duration + stim_durations[i] + fixed_duration] = -1 # off
                except IndexError:
                    continue
        
        # these dicts are integer indexable with the current number of trial frames 
        self.var_isi_dict_flip = {dur:frames for dur, frames in zip(stim_durations, var_isi_flip)}
        self.var_dur_dict_flip = {dur:frames for dur, frames in zip(stim_durations, var_duration_flip)}
        
        print("var isi durations: {}".format(self.var_isi_dict_flip))

        ## making parameters
        # there are two parameters:
        # type: ['isi', 'dur']
        # stim_dur: [0, 2, 4, ... , 64] stimulus durations in frames for 120 FPS

        # for now, this is just a list of trialwise dictionaries specifying
        # these two params. Will be made definitive at some later point

        self.texture_paths = glob.glob('./textures/*') # get paths to textures
        params = [dict(trial_type='dur' if trial_nr % 2 == 0 else 'isi', # back-to-back isi-dur
                       stim_dur = stim_durations[trial_nr%len(stim_durations)], # increasing durations
                       oneOverF_texture_path = self.texture_paths[trial_nr%10], # one after the other stimulus
                       dot_color_timings = self._make_dot_color_timings()) 
                  for trial_nr in range(self.n_trials)] # this makes back-to-back isi-dur with increasing durations

        for trial_nr in range(self.n_trials):
            self.trials.append(
                DelayedNormTrial(session=self,
                          trial_nr=trial_nr,
                          phase_durations=durations[trial_nr],
                          txt='Trial %i' % trial_nr,
                          parameters=params[trial_nr],
                          verbose=True,
                          timing=timing)
            )
            print("made trial {} with params: {} phase duration {} and timing: {}".format(trial_nr, params[trial_nr], durations[trial_nr], timing))

    def _make_dot_color_timings(self):
        # Inspired by Marco's fixation task
        dot_switch_color_times = np.arange(3, 360, float(3.5))
        # adding randomness
        dot_switch_color_times += (2*np.random.rand(len(dot_switch_color_times))-1) # adding uniform noise [-1, 1] 
        # transforming to frames
        dot_switch_color_times = (dot_switch_color_times*120).astype(int)

        return dot_switch_color_times


    def run(self):
        """ Runs experiment. """
        if self.eyetracker_on:
            self.calibrate_eyetracker()
            self.start_recording_eyetracker()

        self.start_experiment()

        for trial in self.trials:
            trial.run()        

        self.close()

    def close(self):
        """'Closes' experiment. Should always be called, even when
        experiment is quit manually (saves onsets to file)."""

        if self.closed:  # already closed!
            return None

        self.win.callOnFlip(self._set_exp_stop)
        self.win.flip()
        self.win.recordFrameIntervals = False

        print(f"\nDuration experiment: {self.exp_stop:.3f}\n")

        if not op.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # self.global_log = pd.DataFrame(self.global_log).set_index("trial_nr")
        # self.global_log["onset_abs"] = self.global_log["onset"] + self.exp_start

        # # Only non-responses have a duration
        # nonresp_idx = ~self.global_log.event_type.isin(["response", "trigger", "pulse"])
        # last_phase_onset = self.global_log.loc[nonresp_idx, "onset"].iloc[-1]
        # dur_last_phase = self.exp_stop - last_phase_onset
        # durations = np.append(
        #     self.global_log.loc[nonresp_idx, "onset"].diff().values[1:], dur_last_phase
        # )
        # self.global_log.loc[nonresp_idx, "duration"] = durations

        # # Same for nr frames
        # nr_frames = np.append(
        #     self.global_log.loc[nonresp_idx, "nr_frames"].values[1:], self.nr_frames
        # )
        # self.global_log.loc[nonresp_idx, "nr_frames"] = nr_frames.astype(int)

        # # Round for readability and save to disk
        # self.global_log = self.global_log.round(
        #     {"onset": 5, "onset_abs": 5, "duration": 5}
        # )
        # f_out = op.join(self.output_dir, self.output_str + "_events.tsv")
        # self.global_log.to_csv(f_out, sep="\t", index=True)

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

        # save frame timings
        frametimings_df = pd.DataFrame(self.trialwise_frame_timings, columns = ["trial {}".format(str(i).zfill(2)) for i in range(self.n_trials)] )
        frametimings_df.to_csv(op.join(self.output_dir, self.output_str + "_frametimings.csv"), index = False)

        if self.mri_simulator is not None:
            self.mri_simulator.stop()

        self.win.close()
        self.closed = True


if __name__ == '__main__':

    settings = op.join(op.dirname(__file__), 'settings.yml')
    session = DelayedNormSession('sub-02', n_trials=100, settings_file=settings,
                                 eyetracker_on = False) # False if testing without eyetracker!
    # session.create_trials(durations=(.25, .25), timing='seconds')
    session.create_trials()
    print(session.var_dur_dict_flip)
    print(session.var_isi_dict_flip)

    #session.create_trials(durations=(3, 3), timing='frames')
    session.run()
    session.quit()