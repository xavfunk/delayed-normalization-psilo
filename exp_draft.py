import os.path as op
from exptools2.core import Session
from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial
from psychopy.visual import TextStim, ImageStim
from exptools2 import utils
import numpy as np
import glob
from psychopy import event

class TestTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, txt=None, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        
        # get the 1/f texture
        self.img = ImageStim(self.session.win, self.parameters['oneOverF_texture_path'],
                             mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred
        self.blank = TextStim(self.session.win, text='') 


        # get the stimulus array with self.session.var_isi/dur_dict and safe it into self.frames
        if self.parameters['trial_type'] == 'dur':
            self.frames = self.session.var_dur_dict[self.parameters['stim_dur']]
        else:
            self.frames = self.session.var_isi_dict[self.parameters['stim_dur']]

        #self.fix_color = "red"
        self.change_back_counter = 0 # counting frames to change back fixation color
        self.overt_counter_trial = TextStim(self.session.win, text='{}'.format(self.session.nr_frames), color = 'green') # for debugging
        self.overt_counter_iti = TextStim(self.session.win, text='{}'.format(self.session.nr_frames), color = 'red') # for debugging
        self.overt_counter_prep = TextStim(self.session.win, text='{}'.format(self.session.nr_frames), color = 'blue') # for debugging
    
    def wait_print_t(self):
        # wait for t
        events = event.getKeys(keyList=['t'])
        if events: 
            self.overt_counter_prep.setText('saw t')
            self.overt_counter_prep.draw()   

    def draw(self):
        """ Draws stimuli """
        if self.phase == 0: # prep time
            self.overt_counter_prep.setText(self.session.nr_frames) 
            self.overt_counter_prep.draw() 

            self.wait_print_t()

        elif self.phase == 1: # we are in stimulus presentation
            
            ## if the self.frame array at this frame index is one, show the texture, otherwise blank or fix
            if self.frames[self.session.nr_frames%self.session.total_duration] == 1:
#            if self.frames[self.session.nr_frames] == 1:  

                # draw texture
                self.img.draw()
                # draw fixation TODO confirm we want this
                self.session.default_fix.draw()

                # debug
                self.overt_counter_trial.setText(self.session.nr_frames) 
                self.overt_counter_trial.draw() 
                self.wait_print_t()

            else:
                # draw blank TODO confirm if this is needded
                self.blank.draw()            
                # draw fixation 
                self.session.default_fix.draw()
                
                # debug
                self.overt_counter_trial.setText(self.session.nr_frames) 
                self.overt_counter_trial.draw() 
                self.wait_print_t()



        else: # we are in iti
            # draw fixation
            self.session.default_fix.draw()

            # debug
            self.overt_counter_iti.setText(self.session.nr_frames) 
            self.overt_counter_iti.draw()
            self.wait_print_t()
            

        # fixation dot color change; makes frames drop
        # if self.session.nr_frames in self.parameters['dot_color_timings']:
        #     # change color
        #     self.session.default_fix.setColor('red')
        #     self.change_back_counter = 96 # keep red for 96 frames, ie 800 ms
        
        # # count down frames left before changing color back
        # if self.change_back_counter != 0:
        #     self.change_back_counter -= 1

        # # change color back if not yet black
        # else:
        #     if self.session.default_fix.color != 'black':
                
        #         self.session.default_fix.setColor('black')    

        # overt_counter = TextStim(self.session.win, text='{}'.format(self.session.nr_frames)) # for debugging
        # overt_counter.draw()        


class EyetrackerSession(PylinkEyetrackerSession):
    """ Simple session with x trials. """
    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10, eyetracker_on=True):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file, eyetracker_on=eyetracker_on)

    def create_trials(self, timing='frames'):
        self.trials = []
        # 3 phases: prep, ITI and trial
        trial_duration = 96
        frames_TR = int(round(self.settings['mri']['TR']*120))
        full_iti_duration = 3 * frames_TR

        prep_durations = self.n_trials * [frames_TR//2] # 1/2 of a TR, setting up and waiting for t # also 1.33/2 will be 79.8 frames, rounding to 80
        iti_durations = self.n_trials * [full_iti_duration - trial_duration - frames_TR//2] # placeholder for TODO implement itis note they will be the time from onset of a stimulus, so it will be ITI - 1/2 TR - 96
        trial_durations = self.n_trials * [trial_duration] # this will be constant TODO confirm length

        # make durations list of tuples for prep, iti, trial
        durations = list(zip(prep_durations, trial_durations, iti_durations))
        
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
                TestTrial(session=self,
                          trial_nr=trial_nr,
                          phase_durations=durations[trial_nr],
                          txt='Trial %i' % trial_nr,
                          parameters=params[trial_nr],
                          verbose=True,
                          timing=timing)
            )

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
            # TODO wait for scanner t
            trial.run()            

        self.close()


if __name__ == '__main__':

    settings = op.join(op.dirname(__file__), 'settings.yml')
    session = EyetrackerSession('sub-01', n_trials=100, settings_file=settings,
                                 eyetracker_on = False) # False if testing without eyetracker!
    # session.create_trials(durations=(.25, .25), timing='seconds')
    session.create_trials()

    #session.create_trials(durations=(3, 3), timing='frames')
    session.run()
    session.quit()