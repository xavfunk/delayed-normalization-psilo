"""
Visual detection task

• Gabor stimuli are presented on 50% of all trials, hidden in continously presented dynamic noise.
• Participants are asked to press either of the 4 buttons indicating whether
  they saw a Gabor patch or not and whether they are confident about their
  choice or not.

• Participants will have to be instructed to fairly equally distribute their confidence reports. 
  The task is quite challenging, but they have to refrain from saying low confidence all the time. 

• Because of the continuous noise, the onset of the stimulus interval is cued by a change in color of the
  fixation dot (to green). The dot remains green during the response interval and will turn black after the response.

• If participants did not press a button on time (1.4s after stimulus onset) the fixation dot will 
  momentarily turn white. 

• There is no other trial-to-trial feedback, but participants see the score + average confidence of
  the previous block during blocks breaks. 

• The settings of the original experiment are 4 blocks of 100 trials. The experiment should not last longer than 20 minutes.
"""
import os
import numpy as np
import pandas as pd
import glob, shutil

import psychopy
from psychopy import logging, visual, event
from psychopy.visual import GratingStim, filters, Rect

psychopy.prefs.hardware['audioDevice'] = ['PTB']
logging.console.setLevel(logging.WARNING)

from exptools2.core.trial import Trial
from exptools2.core.eyetracker import PylinkEyetrackerSession
from IPython import embed as shell
import matplotlib.pyplot as plt

settings_file = glob.glob('settings.yml')[0]

# %%
# Experiment settings
np.random.seed(1)   # NOTE: because of this line all experiments will always be idential in the order of stimuli. Feel free to remove if unwanted.
max_trials = 400    # Total number of trials in experiment
block_len = 100     # Number of trials per block

tracker_on = False
signal_parameters = {'contrast': 0.8, 'spatial_freq': 0.035, 'size': 17} # Note: size in degrees
noise_parameters =  {'contrast': 0.35, 'opacity': 0.50, 'size': 19} # Note: size in degrees

class DetectTrial(Trial):
    def __init__(self, parameters = {}, phase_names =[], phase_durations=[], session=None, monitor=None, tracker=None, ID=0, intensity=1): # self, parameters = {}, phase_durations=[], session=None, monitor=None, tracker=None, ID=0,
        self.monitor = monitor
        self.parameters = parameters
        self.ID = ID
        self.phase_durations = phase_durations  
        self.session = session
        self.miniblock = np.floor(self.ID/self.session.block_len)
        self.phase_names = phase_names  
        self.noise_played = False
        self.signal_tone_played = False
        self.intensity = intensity
        
        self.create_stimuli()
        self.parameters.update({'response': -1, 
                                'correct': -1,
                                'miniblock': self.miniblock ,
                                'blinkDuringTrial': 0,
                                'RT': -1,
                                'trial': self.ID,
                                'confidence' : -1})        

        self.stopped = False
        super(DetectTrial, self).__init__(phase_durations = phase_durations,
                                         parameters = parameters,
                                         phase_names = self.phase_names,  
                                         session = self.session, 
                                         trial_nr = self.ID)

    def create_stimuli(self):

        self.center = ( self.session.win.size[0]/2.0, self.session.win.size[1]/2.0 )       
        self.fixation = GratingStim(self.session.win, tex='sin', units='pix',mask = 'circle',size=10, pos=[0,0], sf=0, color ='black')
        self.stim_sizePIX =  np.round(signal_parameters['size'] * self.session.pix_per_deg).astype(int)
        self.noise_sizePIX = np.round(noise_parameters['size'] * self.session.pix_per_deg / 2).astype(int) # <= notice division by 2, it's to increase the noise element size to 2 later

        spatial_freq =  signal_parameters['spatial_freq']
        self.grating = GratingStim(self.session.win,contrast = signal_parameters['contrast'], opacity=self.parameters['signal_opacity'],
            tex = "sin", mask="gauss", units='pix', size=self.stim_sizePIX, sf = signal_parameters['spatial_freq'], color=[1,1,1]) 

        # Define initial values for noise stimulus. Binary noise.
        noiseTexture = np.random.randint(0,2, size=[self.noise_sizePIX,self.noise_sizePIX]) *2 -1

        # Duplicate values, to increase noise element size to 2
        noiseTexture = np.repeat(np.repeat(noiseTexture, 2, axis=1), 2, axis=0)

        # Create noise stimulus
        self.noise = GratingStim(self.session.win, contrast= noise_parameters['contrast'], name='noise', units='pix', 
            mask='raisedCos', size=(self.noise_sizePIX*2, self.noise_sizePIX*2), opacity= noise_parameters['opacity'], 
            blendmode='avg', tex=noiseTexture)

        # set orientation of target Gabor        
        self.grating.ori = -45 + self.parameters['signal_orientation'] * 90

        # Some introductory text
        if self.ID == 0:
            intro_text = """In this experiment you are supposed to detect a visual pattern in noise.\n
            Use the 4 buttons to indicate whether you saw the visual pattern or not and how confident you are about your decision. \n
            Use one of the two right keys if you saw the pattern or one of the two left keys if you did not see the pattern, but only noise. \n
            If you are a little more sure of your decision, press the outer key and if you are a little less sure of your decision, press the inner key.\n
            LEFT-OUTER(A)      LEFT-INNER(S)      RIGHT-INNER(K)      RIGHT-OUTER(L)\n     absent-certain   absent-uncertain   present-uncertain   present-certain\n
            Press the spacebar to continue.\n
            An example of such a visual pattern:"""

        elif (self.ID % self.session.block_len == 0) and self.ID>0:
            conf = int(np.array(self.session.confidence)[-self.session.block_len:][np.array(self.session.confidence)[-self.session.block_len:] >= 0].sum() / float(self.session.block_len) * 100.0)
            perf = int(np.array(self.session.corrects)[-self.session.block_len:][np.array(self.session.corrects)[-self.session.block_len:] >= 0].sum() / float(self.session.block_len) * 100.0)

            intro_text = f"""You had {perf}% of correct responses and {conf}% trials with high confidence.
            Press the spacebar to continue."""

        else:
            intro_text = ''

        self.message = visual.TextStim(self.session.win, pos=[0,0], text= intro_text, color = (1.0, 1.0, 1.0), height=0.5, font='Arial', wrapWidth=850)
        self.example_grating = GratingStim(self.session.win,  tex = "sin", pos=(0,-350), mask="gauss", units='pix',  size=200, sf = 0.05, contrast = 1,ori = -45, color=[1,1,1])

    def draw(self):

        # draw additional stimuli:
        # Continuously update noise pattern
        noiseTexture = np.random.random([self.noise_sizePIX,self.noise_sizePIX])*2.-1.
        noiseTexture = np.repeat(np.repeat(noiseTexture, 2, axis=1), 2, axis=0)
        self.noise.tex = noiseTexture

        if self.phase == 0: # Block start + instructions
            
            if self.ID % self.session.block_len == 0:
                self.message.draw()
                
                if self.ID == 0:
                    self.example_grating.draw()

        elif self.phase == 1: # baseline
            # draw:
            self.noise.draw()
            self.fixation.draw()

        elif self.phase == 2: # stimulus interval
            self.fixation.color = 'green'
            # draw:
            self.noise.draw()                 

            if self.parameters['signal_present']:
                self.grating.draw()           

            self.fixation.draw()

        elif self.phase == 3: # Response interval
            self.noise.draw()
            self.fixation.draw()

        elif self.phase == 4:  # Feedback <- only if someone did not press a button

            if self.parameters['response'] == -1:
                self.fixation.color = 'white'

            else:
                self.fixation.color = 'black'

            # draw:
            self.noise.draw()
            self.fixation.draw() 

        elif self.phase == 5:  # ITI
            self.fixation.color = 'black'
            # draw:
            self.noise.draw()
            self.fixation.draw() 

    def event(self):
        # trigger = None
        for ev, t in event.getKeys(timeStamped=self.session.clock):
           
            if len(ev) > 0:
                
                if ev in ['esc', 'escape']:
                    self.stopped = True
                    self.session.stopped = True

                    print('run canceled by user')
                    self.stop_trial()
                    if self.phase == 0:
                        self.first_trial_hold = False
                        self.stop_phase()

                elif ev == 'space':
                    if self.phase == 0:
                        self.first_trial_hold = False
                        self.stop_phase()

                elif ev == 'a':

                    if self.phase == 3:
                        idx = self.session.global_log.shape[0]
                        stim_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.ID) & \
                                                                 (self.session.global_log['event_type']=='stimulus_window'), 'onset'].to_numpy()

                        RT = t - stim_onset

                        self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                        self.session.global_log.loc[idx, 'onset'] = t
                        self.session.global_log.loc[idx, 'event_type'] = 'response'
                        self.session.global_log.loc[idx, 'phase'] = self.phase
                        self.session.global_log.loc[idx, 'key'] = ev
                        self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                        self.session.global_log.loc[idx, 'response'] = 0
                        self.session.global_log.loc[idx, 'confidence'] = 1
                        self.session.global_log.loc[idx, 'RT'] = RT

                        self.parameters['response'] = 0
                        self.parameters['confidence'] = 1
                        self.parameters['RT'] = RT

                        if self.parameters['signal_present'] ==  self.session.global_log.loc[idx, 'response']:
                            self.session.global_log.loc[idx, 'correct'] = 1
                            self.parameters['correct'] = 1

                        elif self.parameters['signal_present'] !=  self.session.global_log.loc[idx, 'response']:
                            self.session.global_log.loc[idx, 'correct'] = 0
                            self.parameters['correct'] = 0

                        self.stop_phase()

                                            

                elif ev == 's':

                    if self.phase == 3:
                        idx = self.session.global_log.shape[0]
                        stim_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.ID) & \
                                                                 (self.session.global_log['event_type']=='stimulus_window'), 'onset'].to_numpy()[0]

                        RT = t - stim_onset

                        self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                        self.session.global_log.loc[idx, 'onset'] = t
                        self.session.global_log.loc[idx, 'event_type'] = 'response'
                        self.session.global_log.loc[idx, 'phase'] = self.phase
                        self.session.global_log.loc[idx, 'key'] = ev
                        self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                        self.session.global_log.loc[idx, 'response'] = 0
                        self.session.global_log.loc[idx, 'confidence'] = 0
                        self.session.global_log.loc[idx, 'RT'] = RT

                        self.parameters['response'] = 0
                        self.parameters['confidence'] = 0
                        self.parameters['RT'] = RT

                        if self.parameters['signal_present'] ==  self.session.global_log.loc[idx, 'response']:
                            self.session.global_log.loc[idx, 'correct'] = 1
                            self.parameters['correct'] = 1

                        elif self.parameters['signal_present'] !=  self.session.global_log.loc[idx, 'response']:
                            self.session.global_log.loc[idx, 'correct'] = 0
                            self.parameters['correct'] = 0 

                        self.stop_phase()

                elif ev == 'k':

                    if self.phase == 3:
                        idx = self.session.global_log.shape[0]
                        stim_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.ID) & \
                                                                 (self.session.global_log['event_type']=='stimulus_window'), 'onset'].to_numpy()[0]

                        RT = t - stim_onset

                        self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                        self.session.global_log.loc[idx, 'onset'] = t
                        self.session.global_log.loc[idx, 'event_type'] = 'response'
                        self.session.global_log.loc[idx, 'phase'] = self.phase
                        self.session.global_log.loc[idx, 'key'] = ev
                        self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                        self.session.global_log.loc[idx, 'response'] = 1
                        self.session.global_log.loc[idx, 'confidence'] = 0
                        self.session.global_log.loc[idx, 'RT'] = RT

                        self.parameters['response'] = 1
                        self.parameters['confidence'] = 0
                        self.parameters['RT'] = RT

                        if self.parameters['signal_present'] ==  self.session.global_log.loc[idx, 'response']:
                            self.session.global_log.loc[idx, 'correct'] = 1
                            self.parameters['correct'] = 1

                        elif self.parameters['signal_present'] !=  self.session.global_log.loc[idx, 'response']:
                            self.session.global_log.loc[idx, 'correct'] = 0
                            self.parameters['correct'] = 0

                        self.stop_phase()

                elif ev == 'l':

                    if self.phase == 3:
                        idx = self.session.global_log.shape[0]
                        stim_onset = self.session.global_log.loc[(self.session.global_log['trial_nr']==self.ID) & \
                                                                 (self.session.global_log['event_type']=='stimulus_window'), 'onset'].to_numpy()[0]
                        
                        RT = t - stim_onset

                        self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
                        self.session.global_log.loc[idx, 'onset'] = t
                        self.session.global_log.loc[idx, 'event_type'] = 'response'
                        self.session.global_log.loc[idx, 'phase'] = self.phase
                        self.session.global_log.loc[idx, 'key'] = ev
                        self.session.global_log.loc[idx, 'nr_frames'] = self.session.nr_frames 
                        self.session.global_log.loc[idx, 'response'] = 1
                        self.session.global_log.loc[idx, 'confidence'] = 1                
                        self.session.global_log.loc[idx, 'RT'] = RT

                        self.parameters['response'] = 1
                        self.parameters['confidence'] = 1
                        self.parameters['RT'] = RT

                        if self.parameters['signal_present'] ==  self.session.global_log.loc[idx, 'response']:

                            self.session.global_log.loc[idx, 'correct'] = 1
                            self.parameters['correct'] = 1

                        elif self.parameters['signal_present'] !=  self.session.global_log.loc[idx, 'response']:

                            self.session.global_log.loc[idx, 'correct'] = 0
                            self.parameters['correct'] = 0

                        self.stop_phase()

    def pupil_check(self):

        # Track if participants blinked during critical phases of the trial (baseline, stimulus, response)
        pupil = self.session.tracker.getNewestSample().getLeftEye().getPupilSize()

        if self.phase == 1:
            if pupil == 0:  # if blink, mark the trial as bad 
                self.blinkDuringTrial = True
                self.parameters['blinkDuringTrial'] = 1

        elif self.phase == 2: 
            if pupil == 0: # if blink, mark the trial as bad 
                self.blinkDuringTrial = True
                self.parameters['blinkDuringTrial'] = 1

        elif self.phase == 3:
            if pupil == 0: # if blink, mark the trial as bad 
                self.blinkDuringTrial = True
                self.parameters['blinkDuringTrial'] = 1

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
                self.session.timer.addTime(-phase_dur)
                self.first_trial_hold = True
                if (self.ID==0 or self.ID % self.session.block_len == 0) and self.phase==0 :
                    while self.first_trial_hold:

                        self.draw()
                        if self.draw_each_frame:
                            self.session.win.flip()
                            self.session.nr_frames += 1

                        if self.eyetracker_on:
                            self.pupil_check()

                        self.event()

                while self.session.timer.getTime() < 0 and not self.exit_phase and not self.exit_trial:
                    self.draw()
                    if self.draw_each_frame:
                        self.session.win.flip()
                        self.session.nr_frames += 1

                    if self.eyetracker_on:
                        self.pupil_check()

                    self.event()

            if self.exit_phase:  # broke out of phase loop
                self.session.timer.reset()  # reset timer!
                self.exit_phase = False  # reset exit_phase

            if self.exit_trial:
                self.session.timer.reset()
                break

            self.phase += 1  # advance phase

#%%

class DetectSession(PylinkEyetrackerSession):

    def __init__(self, subject_initials, block_len, index=1,  block=0, eyetracker_on=True):
        output_str = os.path.join('detection_data',subject_initials)
        # super(DetectSession, self).__init__(subject_initials, output_str, settings_file)
        super(DetectSession, self).__init__(subject_initials, output_str, settings_file, eyetracker_on=eyetracker_on) # initialize parent class

        self.subject_initials = subject_initials
        self.block_len = block_len
        self.index = index
        self.block=block
        self.create_yes_no_trials()
        self.settings_file = 'settings.yml'

    def create_yes_no_trials(self):
        """creates trials for yes/no runs"""
        self.signal_present = np.array([0,1])
        self.signal_orientation = np.array([0,1])

        unique_trials = len(self.signal_present) * len(self.signal_orientation)
        self.standard_parameters = {'subject': self.subject_initials, 'block': self.block}

        # Amount of signal present trials
        present_trials = max_trials/2
        # Lower and upper contrast limit
        lim_low        = .005   #.008
        lim_up         = .05    #.080
        unique_signals = 10     

        signal_repetitions = present_trials/unique_signals

        if not signal_repetitions == int(signal_repetitions):
            raise ValueError('Signal strengths not balanced')

        signal_repetitions = int(signal_repetitions)

        print('Average signal strength: {:.3f}'.format(np.mean([lim_up,lim_low]), 3))
        signals = np.linspace(lim_low, lim_up, unique_signals)
        print('Unique signals: {}'.format(signals))

        # Now draw some random signals from uniform distribution, with upper and lower limit. We will use 
        signals = np.concatenate(signal_repetitions*[signals])
        signal_order = np.argsort(np.random.rand(len(signals)))
        signals = signals[signal_order]

        # create yes-no trials in nested for-loop:  
        self.trial_parameters_and_durs = []    
        trial_counter = 0
        self.total_duration = 0

        for pres in range(self.signal_present.shape[0]):            # Loop over signal present/absent
            pres_counter = 0
            for ori in range(self.signal_orientation.shape[0]):     # Loop over orientation CW/CCW
                for i in range(int(max_trials/unique_trials)):      # loop over trials within each sub-condition
                    # phase durations, and iti's:
                    # phases: 0=pretrial, 1=baseline, 2=stim, 3=response, 4=feedback 5=ITI
                    phase_durs = [-0.01, 0.5, 0.2, 1.2, 0.3, np.random.uniform(0.6, 1.0)] 
                    params = self.standard_parameters.copy()
                    params.update({'signal_present':self.signal_present[pres], 'signal_orientation': self.signal_orientation[ori]})

                    if pres == 1:
                        params.update({'signal_opacity': signals[pres_counter]})
                        pres_counter += 1 

                    else:
                        params.update({'signal_opacity': 0})

                    self.trial_parameters_and_durs.append([params.copy(), np.array(phase_durs)])
                    self.total_duration += np.array(phase_durs).sum()
                    trial_counter += 1

        parameters = pd.DataFrame([self.trial_parameters_and_durs[t][0] for t in range(len(self.trial_parameters_and_durs))])
        self.run_order = np.argsort(np.random.rand(len(self.trial_parameters_and_durs)))
        self.phase_names = ['blockStart','baseline_window','stimulus_window','response_window','feedback_window', 'ITI_window']

        self.corrects = []
        self.confidence = []

        # Get index of first trials of each block
        ftib = np.arange(0, max_trials, block_len)

        # Now, loop over all trials (in random order) to initialize the Trial classes. Later we can then just run the pre-initialised 
        # Trial objects, so that the transition from trial N to trial N+1 goes more swiftly. This should minimize the delay in noise presentation.

        self.Trials2Run = []
        for t_no, t in enumerate(self.run_order):

            if np.isin(t_no, ftib): # For first trial in block, add some time before target stimulus onset
                phase_durs_ftib = phase_durs
                phase_durs_ftib[1] += 1 
                self.trial_parameters_and_durs[t][1] = phase_durs_ftib

            self.Trials2Run.append(DetectTrial(parameters=self.trial_parameters_and_durs[t][0], phase_durations=self.trial_parameters_and_durs[t][1],\
                                      session=self,monitor=self.monitor,  ID=t_no, phase_names = self.phase_names))

        print("total duration: %.2f min." % (self.total_duration / 60.0))

    def run(self):
        """run the session"""
        # Cycle through trials
        self.start_experiment()
        self.stopped = False
        self.session_parameters = []

        if self.eyetracker_on:
            self.calibrate_eyetracker()
            self.start_recording_eyetracker()

        fixlostcount = 0

        # loop over trials
        for t in range(max_trials):
            self.stopped = False
            print(t)
            # Run trial
            # shell()
            self.Trials2Run[t].run()    

            # Print message to terminal, if participant blinked
            if self.Trials2Run[t].parameters['blinkDuringTrial']:
                fixlostcount +=1
                print(str(fixlostcount) + " trials with lost fixation")

            self.corrects.append(self.Trials2Run[t].parameters['correct'])
            self.confidence.append(self.Trials2Run[t].parameters['confidence'])
            self.session_parameters.append(self.Trials2Run[t].parameters)

            if self.stopped == True:
                break

        self.close()

def main(initials, index, block=0):
    ts = DetectSession(subject_initials=initials, block_len = block_len, index=index, block=block, eyetracker_on=tracker_on)
    ts.run()

    # define directory to store behavioral data
    dataDir = os.path.join(os.getcwd(), 'detection_data', initials)

    if not os.path.isdir(dataDir):
        os.makedirs(dataDir)

    # and save
    pd.DataFrame(ts.session_parameters).to_csv(os.path.join(dataDir, '{}_parameters.csv'.format(initials)))

if __name__ == '__main__':

    # Store info about the experiment session
    initials = "{:02d}".format(int(input('\n\nParticipant ID: ')))
    index = 0

    main(initials=initials, index=index, block = 0)

