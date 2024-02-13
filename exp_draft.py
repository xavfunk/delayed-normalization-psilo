import os.path as op
from exptools2.core import Session
from exptools2.core import PylinkEyetrackerSession
from exptools2.core import Trial
from psychopy.visual import Rect, TextStim, ImageStim
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


        # get the stimulus array with self.session.var_isi/dur_dict and save it into self.frames
        if self.parameters['trial_type'] == 'dur':
            # self.frames = self.session.var_dur_dict[self.parameters['stim_dur']]
            self.frames = self.session.var_dur_dict_flip[self.parameters['stim_dur']]
            
            # debugging
            # manual_array = np.zeros(96)
            # manual_array[0] = 1 # on
            # manual_array[95] = -1 # off after 800 ms
            # # flicker a bit
            # manual_array[65] = 1
            # manual_array[71] = -1
            # manual_array[77] = 1
            # manual_array[83] = -1
            # manual_array[89] = 1
            # manual_array[95] = -1

            # self.frames = manual_array


        else:
            # self.frames = self.session.var_isi_dict[self.parameters['stim_dur']]
            self.frames = self.session.var_isi_dict_flip[self.parameters['stim_dur']]

            # debugging
            # manual_array = np.zeros(96)
            # manual_array[0] = 1 # on
            # manual_array[95] = -1 # off after almost 500 ms
            # # flicker a bit
            # manual_array[65] = 1
            # manual_array[71] = -1
            # manual_array[77] = 1
            # manual_array[83] = -1
            # manual_array[89] = 1
            # manual_array[95] = -1

            # self.frames = manual_array
        
        # debugging counters etc
        #self.fix_color = "red"
        # self.change_back_counter = 0 # counting frames to change back fixation color
        # self.overt_counter_trial = TextStim(self.session.win, text='{}'.format(self.session.nr_frames), color = 'green') # for debugging
        # self.overt_counter_iti = TextStim(self.session.win, text='{}'.format(self.session.nr_frames), color = 'red') # for debugging
        # self.overt_counter_prep = TextStim(self.session.win, text='{}'.format(self.session.nr_frames), color = 'blue') # for debugging
        # self.overt_message_flip_counter_0 = TextStim(self.session.win, text='flip counter at 0, draw img and fix', color = 'blue') # for debugging
        # self.overt_message_flip_counter_1 = TextStim(self.session.win, text='flip counter at 1, draw fix', color = 'green') # for debugging
        # self.overt_message_flip_counter_2 = TextStim(self.session.win, text='flip counter at 2, draw img and fix', color = 'blue') # for debugging
        # self.overt_message_flip_counter_3 = TextStim(self.session.win, text='flip counter at 3, draw fix', color = 'red') # for debugging

        # Square for Photodiode
        self.white_square = Rect(self.session.win, 1, 1, pos = (-10,-8))
        self.black_square = Rect(self.session.win, 1, 1, pos = (-10,-8), fillColor = 'black')


    def wait_print_t(self):
        # wait for t
        events = event.getKeys(keyList=['t'])
        if events: 
            self.overt_counter_prep.setText('saw t')
            self.overt_counter_prep.draw()   

    def draw_flip(self, current_frame):
        """ Draws stimuli and flips the window 
        """
        #print(self.session.nr_frames, current_frame)

        # fixation dot color change
        if int((self.session.clock.getTime()*120)) in self.parameters['dot_color_timings']:
            # change color
            self.session.fix_dot_color_idx += 1
            self.session.default_fix.setColor(self.session.fix_dot_colors[self.session.fix_dot_color_idx % len(self.session.fix_dot_colors)])

        if self.phase == 0: # prep timeself.session.total_nr_frames 
            #self.overt_counter_prep.setText(self.session.nr_frames) 
            #self.overt_counter_prep.draw() 
            #self.wait_print_t()

            # draw texture
            #self.img.draw()
            # draw fixation TODO confirm we want this
            self.session.default_fix.draw()
            self.white_square.draw()
            self.session.win.flip()
            # self.img.draw()
            #print("doing nothing in prep at {}".format(self.session.nr_frames))
            
            # events = event.getKeys(keyList=['t'])
            # if events:  
            #     self.exit_phase = True

        elif self.phase == 1: # we are in stimulus presentation
            #print(self.frames)
            ## if the self.frame array at this frame index is one, flip it
            
            # if self.frames[self.session.nr_frames%self.session.total_duration] == 1:
            #print(self.session.nr_frames)
            # try:
                # if self.frames[self.session.nr_frames] == 1: 
            # print(self.frames)
            if self.frames[current_frame] == 1:
                ## debugging printouts
                # print(current_frame)
                # print("trial type is: {}".format(np.where(self.frames == 1)[0]))
                # print("flip counter is: {}".format(self.flip_counter))
                # print("self.frames is: {}".format(self.session.win.frames))
                
                self.black_square.draw()
                self.img.draw()
                self.session.default_fix.draw()

                ## debugging printouts
                #print("flippin at {}".format(self.session.nr_frames))
                self.session.win.flip(clearBuffer = False)
            
            elif self.frames[current_frame] == -1:
                                
                self.white_square.draw()
                self.session.default_fix.draw()

                #print("flippin back at {}".format(self.session.nr_frames))
                self.session.win.flip()
                self.white_square.draw()
                self.session.default_fix.draw()

            else:
                #self.white_square.draw()
                # self.session.default_fix.draw()
                self.session.win.flip(clearBuffer = False)
            
        
        else: # we are in iti
            self.white_square.draw()
            # draw fixation
            self.session.default_fix.draw()
            self.session.win.flip()

            ## debugging printouts
            # self.overt_counter_iti.setText(self.session.nr_frames) 
            # self.overt_counter_iti.draw()
            # self.wait_print_t()




    def draw(self):
        """ Draws stimuli 
        This is to be used when flipping on every frame
        """
        # fixation dot color change
        if int((self.session.clock.getTime()*120)) in self.parameters['dot_color_timings']:
            # change color
            self.session.fix_dot_color_idx += 1
            self.session.default_fix.setColor(self.session.fix_dot_colors[self.session.fix_dot_color_idx % len(self.session.fix_dot_colors)])

        if self.phase == 0: # prep time
            ## debugging
            # self.overt_counter_prep.setText(self.session.nr_frames) 
            # self.overt_counter_prep.draw() 

            # self.wait_print_t()

            self.white_square.draw()
            self.session.default_fix.draw()

        elif self.phase == 1: # we are in stimulus presentation
            
            ## if the self.frame array at this frame index is one, show the texture, otherwise blank or fix
#            if self.frames[self.session.nr_frames%self.session.total_duration] == 1:
            if self.frames[self.session.nr_frames] == 1:

#            if self.frames[self.session.nr_frames] == 1:  

                # draw texture
                self.img.draw()
                # draw fixation TODO confirm we want this
                self.session.default_fix.draw()
                self.black_square.draw()


                # debug
                # self.overt_counter_trial.setText(self.session.nr_frames) 
                # self.overt_counter_trial.draw() 
                # self.wait_print_t()

            else:
                # draw blank TODO confirm if this is needded
                self.blank.draw()     
    
                self.white_square.draw()

                # draw fixation 
                self.session.default_fix.draw()
                
                # debug
                # self.overt_counter_trial.setText(self.session.nr_frames) 
                # self.overt_counter_trial.draw() 
                # self.wait_print_t()



        else: # we are in iti
            
            self.white_square.draw()
            # draw fixation
            self.session.default_fix.draw()

            # debug
            # self.overt_counter_iti.setText(self.session.nr_frames) 
            # self.overt_counter_iti.draw()
            # self.wait_print_t()

        # overt_counter = TextStim(self.session.win, text='{}'.format(self.session.nr_frames)) # for debugging
        # overt_counter.draw()

    def draw_return(self):
        """ 
        DPERECATED
        Draws stimuli and return flipping flag
        """
        if self.phase == 0: # prep time
            # self.overt_counter_prep.setText(self.session.nr_frames) 
            # self.overt_counter_prep.draw() 
            self.session.default_fix.draw()

            # self.wait_print_t()
            return 1

        elif self.phase == 1: # we are in stimulus presentation
            
            if self.frames[self.session.nr_frames] == 1:

                # draw texture
                self.img.draw()
                # draw fixation TODO confirm we want this
                self.session.default_fix.draw()
                
                return 1 

            elif self.frames[self.session.nr_frames] == -1:
                self.session.default_fix.draw()
                return 1

            else:
                return 0


        else: # we are in iti
            # draw fixation
            self.session.default_fix.draw()
            return 1
            



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

                    if self.exit_phase or self.exit_trial:
                        break
                    # here, either draw_flip() or (draw() and session.win.flip()) should be used
                    self.draw_flip(current_frame = frame)
                    # self.draw()
                    # self.session.win.flip()
                    
                    # print("it is clock time {}".format(self.session.clock.getTime()))
                    # print("it is clock frame time {}".format(int(self.session.clock.getTime()*120)))
                    
                    # flip_flag = self.draw_return()
                    # #print(flip_flag)
                    # if flip_flag == 1:
                    #     if self.phase ==1:
                    #         print("flipping at {}".format(frame))
                    #     self.session.win.flip()
                    #print("it is timer time {}".format(self.session.timer.getTime()))
                    if self.phase == 1:
                        # TODO instead of printing, put these values into an array
                        # that tells us exactly on which frames this happened
                        print(frame, session.win.nDroppedFrames, self.frames[frame])
                        # TODO how to exaclty handle the checked frames
                        # save them to df, while keeping trial parameters
                        self.check_frames[frame][0] = frame
                        self.check_frames[frame][1] = session.win.nDroppedFrames
                        self.check_frames[frame][2] = self.frames[frame]

                    
                    self.get_events()
                    self.session.nr_frames += 1

            if self.exit_phase:  # broke out of phase loop
                self.session.timer.reset()  # reset timer!
                self.exit_phase = False  # reset exit_phase
            if self.exit_trial:
                self.session.timer.reset()
                break

            self.phase += 1  # advance phase        


class EyetrackerSession(PylinkEyetrackerSession):
    """ Simple session with x trials. """
    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10, eyetracker_on=True):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        self.total_nr_frames = 0
        self.fix_dot_color_idx = 0
        self.fix_dot_colors = ['green', 'red']

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
                TestTrial(session=self,
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
            # TODO wait for scanner t
            trial.run()        
            # print(session.win._frameTimes)


        self.close()


if __name__ == '__main__':

    settings = op.join(op.dirname(__file__), 'settings.yml')
    session = EyetrackerSession('sub-01', n_trials=100, settings_file=settings,
                                 eyetracker_on = False) # False if testing without eyetracker!
    # session.create_trials(durations=(.25, .25), timing='seconds')
    session.create_trials()
    print(session.var_dur_dict_flip)
    print(session.var_isi_dict_flip)



    #session.create_trials(durations=(3, 3), timing='frames')
    session.run()
    session.quit()