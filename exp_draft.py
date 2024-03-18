import os.path as op
from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial
from psychopy.visual import Rect, TextStim, ImageStim
from psychopy.sound import Microphone
from scipy.signal import find_peaks

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from psychopy import event
from datetime import datetime

class DelayedNormTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, txt = None, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        
        # get the 1/f texture
        self.img = ImageStim(self.session.win, self.parameters['oneOverF_texture_path'], size = 10,
                             mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred
        self.check_frames = np.zeros((96, 3))

        # get the stimulus array with self.session.var_isi/dur_dict and save it into self.stimulus_frames
        if self.parameters['trial_type'] == 'dur':
            self.stimulus_frames = self.session.var_dur_dict[self.parameters['stim_dur']]
            # self.stimulus_frames = self.session.var_dur_dict_flip[self.parameters['stim_dur']]

        else:
            self.stimulus_frames = self.session.var_isi_dict[self.parameters['stim_dur']]
            # self.stimulus_frames = self.session.var_isi_dict_flip[self.parameters['stim_dur']]

        # squares for Photodiode
        if self.session.photodiode_check is True:
            self.white_square = Rect(self.session.win, 2, 2, pos = (5.5,-2.5))
            self.black_square = Rect(self.session.win, 2, 2, pos = (5.5,-2.5), fillColor = 'black')
            if self.parameters['trial_type'] == 'dur':
                self.square_flip_frames = self.session.var_dur_dict_flip[self.parameters['stim_dur']]
                print(self.square_flip_frames)
            else:
                self.square_flip_frames = self.session.var_isi_dict_flip[self.parameters['stim_dur']]


    def draw(self):
        """ Draws stimuli 
        This is to be used when flipping on every frame
        """
        # fixation dot color change
        # if int((self.session.clock.getTime()*120)) in self.parameters['dot_color_timings']:
        if int((self.session.clock.getTime()*120)) in self.session.dot_color_timings:
        
            # change color
            self.session.fix_dot_color_idx += 1
            self.session.default_fix.setColor(self.session.fix_dot_colors[self.session.fix_dot_color_idx % len(self.session.fix_dot_colors)])

        if self.phase == 0: # we are in phase 0, prep time

            if self.session.photodiode_check is True:
                self.black_square.draw()

            self.session.default_fix.draw()
            
            # trigger 't' breaks out of phase 0, now in get_events
            # if 't' in event.getKeys(keyList = 't'):
                
            #     if self.session.photodiode_check is True:
            #         # start recording
            #         self.session.mic.start()
                
            #     self.exit_phase = True


        elif self.phase == 1: # we are in phase 1, stimulus presentation
            
            ## if the self.stimulus_frames array at this frame index is one, show the texture, otherwise fix
            if self.stimulus_frames[self.session.nr_frames] == 1:
                # draw texture
                self.img.draw()

                # draw fixation
                self.session.default_fix.draw()
                
            else:
                # draw fixation 
                self.session.default_fix.draw()

            if self.session.photodiode_check is True:
                if self.square_flip_frames[self.session.nr_frames]:
                    self.white_square.draw()
                else:
                    self.black_square.draw()

        else: # we are in phase 2, iti
            if self.session.photodiode_check is True:

                # TODO this will oversample! -> correct by putting it outside of draw!
                self.black_square.draw()
                self.session.mic.stop()
                audioClip = self.session.mic.getRecording()
                # plotting for debugging
                # t = np.linspace(0, audioClip.duration, int(np.round(audioClip.sampleRateHz * audioClip.duration)))
                # fig, ax = plt.subplots()
                # ax.plot(t, audioClip.samples[:,1])
                # plt.savefig('audio_recordings/audio_plot_exp.png')

                peaks, _ = find_peaks(audioClip.samples[:,1], height = .3, distance = audioClip.sampleRateHz*1/120) 
                self.session.conditions.append(self.parameters['stim_dur'])
                self.session.trial_type.append(self.parameters['trial_type'])
                self.session.recording_durations.append(audioClip.duration)
                self.session.delta_peaks.append((peaks[1] - peaks[0])/audioClip.sampleRateHz)
                self.session.n_peaks_found.append(len(peaks))

                # self.session.recordings[self.parameters['trial_type']][self.parameters['stim_dur']].append(self.session.mic.getRecording()) # get recording and save into dict 
    
            # draw fixation
            self.session.default_fix.draw()

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

                    if self.phase == 0:
                
                        if self.session.photodiode_check is True:
                           # start recording
                            self.session.mic.start()
                
                        self.exit_phase = True

                else:
                    event_type = 'response'

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
            self.session.win.callOnFlip(self.log_phase_info, phase=self.phase)

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

                for frame in range(phase_dur):
                    
                    if frame == 0:
                        event.clearEvents()

                    if self.exit_phase or self.exit_trial:
                        break
                    
                    # draw stimuli
                    self.draw()
                    
                    # keeping track of flip timings
                    if self.phase == 1:
                        self.session.trialwise_frame_timings[frame, self.trial_nr] = self.session.win.flip()
                    else:
                        self.session.win.flip()                
                        # getting events only outside phase 1 makes a difference for frame timings?
                        self.get_events()   
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
    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=None, eyetracker_on=True, photodiode_check = False):
        """ Initializes TestSession object. """
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file, eyetracker_on=eyetracker_on)
        
        # load params from sequence df
        self.trial_sequence_df = pd.read_csv(self.settings['stimuli']['trial_sequence'])
        # get TR in frames
        self.TR = self.settings['mri']['TR']

        if n_trials is None:
            self.n_trials = len(self.trial_sequence_df)
        else:
            self.n_trials = n_trials
            self.trial_sequence_df = self.trial_sequence_df[:n_trials]

        self.total_nr_frames = 0
        self.fix_dot_color_idx = 0
        self.fix_dot_colors = ['green', 'red']
        self.photodiode_check = True if photodiode_check else False
        self.trialwise_frame_timings = np.zeros((96, self.n_trials))
        self.total_exp_duration = np.sum(self.trial_sequence_df.iti_TR) * self.TR
        
        if photodiode_check == True:
            # only duration
            self.trial_sequence_df = self.trial_sequence_df[self.trial_sequence_df.type == 'dur']
            # quick
            self.trial_sequence_df.iti_TR = [3 for i in range(len(self.trial_sequence_df))]
            # triple
            self.trial_sequence_df = pd.concat([self.trial_sequence_df, self.trial_sequence_df, self.trial_sequence_df])

            self.mic = Microphone(streamBufferSecs=6.0)  # open the microphone
            self.recordings = {"dur" : {timing: [] for timing in [0, 2, 4, 8, 16, 32, 64]}, # hardcoded for now
                               "var" : {timing: [] for timing in [0, 2, 4, 8, 16, 32, 64]}}
            
            self.conditions = [] 
            self.trial_type = [] 
            self.recording_durations = [] 
            self.delta_peaks = [] 
            self.n_peaks_found = [] 


    def create_trials(self, timing='frames'):
        self.trials = []
        TR_in_frames = int(round(self.TR*120))

        # load params from sequence df
        # self.trial_sequence_df = pd.read_csv(self.settings['stimuli']['trial_sequence'])
        # iti in TRs
        iti_TRs = self.trial_sequence_df['iti_TR']
        # iti in frames
        iti_frames = [int(iti_TR * TR_in_frames) for iti_TR in iti_TRs]

        ## phase durations of the 3 phases: prep (p0), trial (p1) and ITI (p2)
        if self.settings['stimuli']['scanner_sync']:
            # just a very large duration, as we wait for t
            prep_durations_p0 = self.n_trials * [100000] 
        else:
            # 1/2 of a TR, setting up and waiting for t # also 1.32/2 will be 79.6 frames, rounding to 80
            prep_durations_p0 = self.n_trials * [TR_in_frames//2]

        # stim_duration_p1 is the duration of a single stimulus, 96 frames or 800 ms for every trial
        stim_duration_p1 = self.settings['stimuli']['stim_duration']
        stim_durations_p1 = self.n_trials * [stim_duration_p1]
        # itis are the time from onset of a stimulus, while iti_durations_p2 are between the end of p1 and start of p0 so it will be ITI - 1/2 TR - 96
        iti_durations_p2 = [iti_frames_trial - stim_duration_p1 - TR_in_frames//2 for iti_frames_trial in iti_frames] 

        # make phase durations list of tuples for prep, iti, trial
        phase_durations = list(zip(prep_durations_p0, stim_durations_p1, iti_durations_p2))

        ## making stimulus arrays
        self.stim_conds = self.settings['stimuli']['stim_conds'] # frames in 120 FPS, either duration or isi times
        self.fixed_duration = self.settings['stimuli']['fixed_duration'] # fixed duration for isi trials
        self.total_duration = self.settings['stimuli']['stim_duration'] # (<800 ms in total) in exp design; 800 ms = .8*120 = 96 frames
        self._make_trial_frame_timings()

        # get paths to textures
        self.texture_paths = glob.glob('./textures/{}/*'.format(self.settings['stimuli']['tex_type'])) # get paths to textures

        # read trial_sequence_df for trial parameters
        params = [dict(trial_type = row.type,
                       stim_dur = row.cond_frames, 
                       oneOverF_texture_path = self.texture_paths[row.texture_id])
                  for i, row in self.trial_sequence_df.iterrows()] 

        # params = [dict(trial_type='dur' if trial_nr % 2 == 0 else 'isi', # back-to-back isi-dur
        #                stim_dur = self.stim_conds[trial_nr%len(self.stim_conds)], # increasing durations
        #                oneOverF_texture_path = self.texture_paths[trial_nr%10], # one after the other stimulus
        #                dot_color_timings = self._make_dot_color_timings()) 
        #           for trial_nr in range(self.n_trials)] # this makes back-to-back isi-dur with increasing durations

        # construct trials
        for trial_nr in range(self.n_trials):
            self.trials.append(
                DelayedNormTrial(session=self,
                          trial_nr=trial_nr,
                          phase_durations=phase_durations[trial_nr],
                          txt='Trial %i' % trial_nr,
                          parameters=params[trial_nr],
                          verbose=False,
                          timing=timing)
            )
            # debugging printout
            print("made trial {} with params: {} phase duration {} and timing: {}".format(trial_nr, params[trial_nr], phase_durations[trial_nr], timing))
        
        # make a dummy trial at the start
        dummy = DelayedNormTrial(session=self,
                          trial_nr=999,
                          phase_durations=(0, 0, self.settings['stimuli']['dummy_trial_trs']*120 - int(self.TR//2)),
                          txt='Trial %i: Dummy' % trial_nr,
                          parameters=dict(trial_type = 'dur',
                                            stim_dur = 0, 
                                            oneOverF_texture_path = self.texture_paths[0]),
                          verbose=False,
                          timing=timing)
        
        self.trials = [dummy] + self.trials

        self.dot_color_timings = self._make_dot_color_timings()

    def _make_trial_frame_timings(self):
        """
        makes frame-wise sequences for stimulus presentation
        flip versions are needed for photodiode
        """
        var_duration = np.vstack([np.hstack((np.ones(stim_cond), # showing stimulus
                                             np.zeros(self.total_duration - stim_cond))) # no stimulus for the remaining frames
                                             for stim_cond in self.stim_conds])
        
        var_isi = np.vstack([np.hstack((np.ones(self.fixed_duration), # show stimulus
                                        np.zeros(stim_cond), # isi
                                        np.ones(self.fixed_duration), # show stimulus again
                                        np.zeros(self.total_duration - stim_cond - 2*self.fixed_duration))) # no stimulus for remaining frames 
                                        for stim_cond in self.stim_conds])
        
        # these dicts are integer indexable with the current number of trial frames 
        self.var_isi_dict = {dur:frames for dur, frames in zip(self.stim_conds, var_isi)}
        self.var_dur_dict = {dur:frames for dur, frames in zip(self.stim_conds, var_duration)}

        var_duration_flip = np.zeros((len(self.stim_conds), self.total_duration))
        for i in range(len(self.stim_conds)):
            #print(i)
            var_duration_flip[i, 0] = 1 # on
            var_duration_flip[i, self.stim_conds[i]] = -1 # off
            
            if self.stim_conds[i] == 0:
                var_duration_flip[i, 0] = 0
        
        var_isi_flip = np.zeros((len(self.stim_conds), self.total_duration))

        for i in range(len(self.stim_conds)):
            #print(i)
            
            if i == 0:
                var_isi_flip[i, 0] = 1 # on
                var_isi_flip[i, 2 * self.fixed_duration] = -1 # off
                
            else:
                try:
                    # fixed 16 frames
                    var_isi_flip[i, 0] = 1 # on
                    var_isi_flip[i, 0 + self.fixed_duration] = -1 # off
                    var_isi_flip[i, 0 + self.fixed_duration + self.stim_conds[i]] = 1 # on
                    var_isi_flip[i, 0 + self.fixed_duration + self.stim_conds[i] + self.fixed_duration] = -1 # off
                except IndexError:
                    continue
        
        # these dicts are integer indexable with the current number of trial frames 
        self.var_isi_dict_flip = {dur:frames for dur, frames in zip(self.stim_conds, var_isi_flip)}
        self.var_dur_dict_flip = {dur:frames for dur, frames in zip(self.stim_conds, var_duration_flip)}

        return

    def _make_dot_color_timings(self, total_time=None):

        if total_time is None:
            total_time = self.total_exp_duration
        # Inspired by Marco's fixation task
        dot_switch_color_times = np.arange(3, total_time, float(3.5))
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
        
        self.display_text('Waiting for scanner \n(remove this text)', keys=self.settings['mri'].get('sync', 't'))

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

        # save frame timings
        frametimings_df = pd.DataFrame(self.trialwise_frame_timings, columns = ["trial {}".format(str(i).zfill(2)) for i in range(self.n_trials)] )
        frametimings_df.to_csv(op.join(self.output_dir, self.output_str + "_frametimings.csv"), index = False)

        current_datetime = datetime.now()
        # plot and save audio
        if self.photodiode_check:
            photo_data = pd.DataFrame({'conditions':self.conditions,
                        'trial_type': self.trial_type,
                        'duration': self.recording_durations,
                        'delta_peaks': self.delta_peaks,
                        'n_peaks_found': self.n_peaks_found})
            
            photo_data.to_csv('photodiode_test_results/timing_photo_exp_results_{}.csv'.format(current_datetime.strftime("%Y-%m-%d-%H-%M")), index = False)

        if self.mri_simulator is not None:
            self.mri_simulator.stop()

        self.win.close()
        self.closed = True


if __name__ == '__main__':

    settings = op.join(op.dirname(__file__), 'settings.yml')
    session = DelayedNormSession('sub-02', settings_file=settings, n_trials = None,
                                 eyetracker_on = False, photodiode_check = False)

    session.create_trials()

    session.run()
    session.quit()