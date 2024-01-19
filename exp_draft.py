import os.path as op
from exptools2.core import Session
from exptools2.core import Trial
from psychopy.visual import TextStim, ImageStim
from exptools2 import utils
import numpy as np
import glob

class TestTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, txt=None, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        
        # get the 1/f texture
        self.img = ImageStim(self.session.win, self.parameters['oneOverF_texture_path']) 
        self.blank = TextStim(self.session.win, text='') 


        # get the stimulus array with self.session.var_isi/dur_dict and safe it into self.frames
        if self.parameters['trial_type'] == 'dur':
            self.frames = self.session.var_dur_dict[self.parameters['stim_dur']]
        else:
            self.frames = self.session.var_isi_dict[self.parameters['stim_dur']]
        
    def draw(self):
        """ Draws stimuli """
        if self.phase == 0: # we are in stimulus presentation
            
            ## if the self.frame array at this frame index is one, show the texture, otherwise blank or fix
            if self.frames[self.session.nr_frames%self.session.total_duration] == 1:  
                # draw texture
                self.img.draw()
            else:
                # draw blank
                self.blank.draw()            

        else: # we are in iti
            # draw fixation
            self.session.default_fix.draw()


class TestSession(Session):
    """ Simple session with x trials. """
    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        super().__init__(output_str, output_dir=None, settings_file=settings_file)

    def create_trials(self, timing='frames'):
        self.trials = []
        # 2 phases: ITI and trial
        iti_durations = self.n_trials * [120] # placeholder for TODO implement itis
        trial_durations = self.n_trials * [96] # this will be constant TODO confirm length

        # make durations list of tuples for iti, trial
        durations = list(zip(trial_durations, iti_durations))
        
        ## making stimulus arrays
        stim_durations = [0, 2, 4, 8, 16, 32, 64] # frames in 120 FPS, either duration or isi times
        fixed_duration = 16 # fixed duration for isi trials
        self.total_duration = 96 # (<800 ms in total) in exp design; 800 ms = .8*120 = 96 frames
        var_duration = np.vstack([np.hstack((np.ones(duration), # showing stimulus
                                             np.zeros(self.total_duration - duration))) # no stimulus for the remaining frames
                                             for duration in stim_durations])
        
        var_isi = np.vstack([np.hstack((np.ones(fixed_duration), # show stimulus
                                        np.zeros(duration), # isi
                                        np.ones(fixed_duration), # show atimulus again
                                        np.zeros(self.total_duration - duration - 2*fixed_duration))) # no stimulus for remaining frames 
                                        for duration in stim_durations])
        
        # these dicts are integer indexable with the number of frames 
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
                       oneOverF_texture_path = self.texture_paths[trial_nr%10]) # one after the other stimulus
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

    def run(self):
        """ Runs experiment. """
        self.start_experiment()
        for trial in self.trials:
            trial.run()            

        self.close()


if __name__ == '__main__':

    settings = op.join(op.dirname(__file__), 'settings.yml')
    session = TestSession('sub-01', n_trials=100, settings_file=settings)
    # session.create_trials(durations=(.25, .25), timing='seconds')
    session.create_trials()

    #session.create_trials(durations=(3, 3), timing='frames')
    session.run()
    session.quit()