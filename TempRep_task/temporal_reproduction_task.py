from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial
from exptools2.stimuli import create_circle_fixation

from psychopy.visual import TextStim, ImageStim, Circle
from psychopy import event
from psychopy import visual, core

import random
import pyglet
import pandas as pd
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import sys

class TemRepTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, txt=None, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        self.txt = TextStim(self.session.win, txt) 
        # get the 1/f texture
        # self.img = ImageStim(self.session.win, self.parameters['oneOverF_texture_path'], size = 10,
        #                      mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred
        self.img = ImageStim(self.session.win, f"textures/{self.session.settings['stimuli']['tex_type']}/oneOverF_texture_1_1024_{trial_nr%100}.bmp", size = 10,
                             mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred
        self.blank = TextStim(self.session.win, '')

    def draw_flip(self):
        """ Draws stimuli """
        if self.phase == 0:
            # cue
            # TODO replace with eye
            # self.txt.draw()
            # self.session.eye.draw()
            self.session.default_fix.draw()
            self.session.win.flip()
            self.get_events()

        
        elif self.phase == 1:
            # jittered blank
            # self.blank.draw()
            self.session.default_fix.draw()
            self.session.win.flip()
            self.get_events()


        elif self.phase == 2:
            # show stimulus for some time
            self.img.draw()
            self.session.default_fix.draw()
            self.session.win.flip()
            self.get_events()

        elif self.phase == 3:
            # fixed blank
            # self.session.default_fix.draw()
            # self.blank.draw()
            self.session.default_fix.draw()
            self.session.win.flip()
            self.get_events()
            event.clearEvents()
          
        elif self.phase == 4:
            # collect answer
            # on keypress show the stimulus
            # if space gets pressed, we count the time

            frame_count = 0
            events = self.get_events()
            # print(events)

            # if 'space' in event.getKeys(keyList = 'space'):
            if events:
                if ('space' in events[0]):

                    # start timer
                    self.session.response_timer.reset()
                    print(f"key pressed, started timer, {self.session.response_timer.getTime()}")

                    # start while loop that will run as long as the key stays pressed
                    key_pressed = True
                    while key_pressed:
                        # if key is pressed, we draw the stimulus
                        if self.session.keyboard[self.session.key.SPACE]: # this evaluates to true as long as the key is pressed
                            # draw stimulus
                            self.img.draw()
                            # green fix on top
                            self.session.green_fix.draw()
                            # counting frames, helpful for debugging
                            frame_count += 1 
                            # flip
                            self.session.win.flip()

                        else:
                            # if key is released, key_pressed will be set to false, exiting the while loop
                            key_pressed = False
                            print(f"key released, ending timer {self.session.response_timer.getTime()}")

                    # getting the passed time
                    response_time = self.session.response_timer.getTime()

                    # saving passed time, target time
                    self.session.response_times.append(response_time)
                    self.session.target_times_f.append(self.phase_durations[2])
                    self.session.target_times.append(self.phase_durations[2]/120)
                    
                    # printout comparing passed time with counted frames
                    print(f"this time has passed with pressed key: {response_time} and frames: {frame_count} so fc/120 {frame_count/120}")
                    print(f"the trial was frames {self.phase_durations[2]}, and s {self.phase_durations[2] / 120}")
                    # exit phase
                    self.exit_phase = True

            # as long as nothing has been pressed, a green button icon ecourages pressing the button
            else:
                # button
                # self.session.button.draw()
                # green fix
                self.session.green_fix.draw()

                # TODO implement condition where time runs out


            # print(f'flipped {i}') # debug message    
            self.session.win.flip()
            # pass

        else:
            # iti
            # self.session.default_fix.draw()
            # self.blank.draw()
            self.session.black_fix.draw()
            self.session.win.flip()
            self.get_events()


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
                    
                    self.draw_flip()
                    # self.draw()
                    # self.session.win.flip()
                    # self.get_events()
                    self.session.nr_frames += 1

            if self.exit_phase:  # broke out of phase loop
                self.session.timer.reset()  # reset timer!
                self.exit_phase = False  # reset exit_phase
            if self.exit_trial:
                self.session.timer.reset()
                break

            self.phase += 1  # advance phase

class TemRepSession(PylinkEyetrackerSession):
    """ Simple session with x trials. """

    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10, eyetracker_on=True):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        super().__init__(output_str, output_dir=output_dir,
                         settings_file=settings_file, eyetracker_on=eyetracker_on)
        
        # icons
        self.button = visual.ImageStim(self.win, 'button.png', size = self.settings['stimuli']['button_size'])
        self.button.setColor((255, 128, 255), 'rgb255')
        # self.eye = visual.ImageStim(self.win, 'witness.png', size = self.settings['stimuli']['eye_size'])
        self.eye = visual.ImageStim(self.win, 'eye.png', size = self.settings['stimuli']['eye_size'], opacity = .5)
        self.green_fix = Circle(self.win, radius=.075, edges = 100, lineWidth=0)
        self.green_fix.setColor((0, 128, 0), 'rgb255')
        self.black_fix = Circle(self.win, radius=.075, edges = 100, lineWidth=0)
        self.black_fix.setColor((0, 0, 0), 'rgb255')

        # keyboard workaround
        self.key = pyglet.window.key
        self.keyboard = self.key.KeyStateHandler()
        self.win.winHandle.push_handlers(self.keyboard)
        
        # response timer
        self.response_timer = core.Clock()
        
        # get stimulus params
        self.conds = self.settings['stimuli']['stim_conds']
        self.n_conds = len(self.conds)
        self.n_repeats = self.settings['stimuli']['n_repeats']
        
        if n_trials is None:
            self.n_trials = self.n_conds * self.n_repeats

        # init result lists
        self.response_times = []
        self.target_times = []
        self.target_times_f = []

    def create_trials(self, durations=None, timing='frames'):
        # p0_durs = [120] * self.n_trials # constant
        p0_durs = [120] * self.n_conds * self.n_repeats # cue

        jits = np.arange(54, 79) # jittered times between 54 and 79 frames, corr. to 425 - 600 ms, as in Yankieva
        p1_durs = [int(jit) for jit in np.random.choice(jits, self.n_conds * self.n_repeats)] # jittered blank
        p2_durs = self.conds * self.n_repeats # random durations showing stim
        random.shuffle(p2_durs) # randomize, if not already
        p3_durs = [60] * self.n_conds * self.n_repeats # 500 ms/60 frames blank
        p4_durs = [3 * 120] * self.n_conds * self.n_repeats #  a few seconds to start answer
        # p5_durs = [2 * 120] * self.n_conds * self.n_repeats # iti, to be scattered maybe, maybe start upon click (TODO parametrize)
        p5_durs = [60] * self.n_conds * self.n_repeats # iti, to be scattered maybe, maybe start upon click (TODO parametrize)
        

        self.trials = []
        durations = list(zip(p0_durs, p1_durs, p2_durs, p3_durs, p4_durs, p5_durs))
        for trial_nr in range(self.n_trials):
            types = [type(dur) for dur in durations[trial_nr]]
            print(types)
            self.trials.append(
                TemRepTrial(session=self,
                          trial_nr=trial_nr,
                          phase_durations=durations[trial_nr],
                          txt='Trial %i' % trial_nr,
                          verbose=False,
                          timing=timing)
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
        results_df['resonse_diff_f'] = [frames_target - frames_resp for frames_target, frames_resp in zip(results_df['target_times_f'], results_df['response_times_f'])]

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
    task = 'TempRep' # different settings -> now implemented as saving the actual settings
    run = sys.argv[3] # which run    
    output_str = f'sub-{subject}_sess-{sess}_task-{task}_run-{run}'
    print(output_str)
    # output_str = 'sub-{}_sess-{}_task-{}_run-{}'.format(subject.zfill(2), sess.zfill(2), task, run.zfill(2))
    # output_str = 'sub-TemRepTest99_sess-TemRepTest_task-TemRep_run-TemRepTest'
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

    session = TemRepSession(output_str, output_dir = results_folder,  eyetracker_on=True, n_trials=None, settings_file='settings_TemRep.yml')

    # print(session.n_conds)
    # print(session.n_repeats)
    session.create_trials()
    session.run()
    # print(session.results)

    
    results_out = f'{results_folder}/{output_str}_results.csv'
    session.results.to_csv(results_out, index = False)
    # print(results_out)

