import os.path as op
from exptools2.core import Session
from exptools2.core import Trial
from psychopy.visual import TextStim, ImageStim
from exptools2 import utils

class TestTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, txt=None, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        self.txt = TextStim(self.session.win, txt)
        x_offset = 0 # positive is right, negative is left
        y_offset = 1 # positive is up, negative is down

        self.session.default_fix.setPos((0+x_offset, 0+y_offset))

        # get the 1/f texture
        self.img = ImageStim(self.session.win, "../textures/minmax/oneOverF_texture_1_1024_0.bmp", size = 10, pos = (0+x_offset,0+y_offset),
                             mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred

    def draw(self):
        """ Draws stimuli """
        self.img.draw()
        self.session.default_fix.draw()



class TestSession(Session):
    """ Simple session with x trials. """
    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        super().__init__(output_str, output_dir=None, settings_file=settings_file)
        self.default_fix.setSize(self.settings['task']['fix_dot_size'])

    def create_trials(self, durations=(.5, .5), timing='seconds'):
        self.trials = []
        for trial_nr in range(self.n_trials):
            self.trials.append(
                TestTrial(session=self,
                          trial_nr=trial_nr,
                          phase_durations=durations,
                          txt='Trial %i' % trial_nr,
                          parameters=dict(trial_type='even' if trial_nr % 2 == 0 else 'odd'),
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
    session = TestSession('size_check', n_trials=1, settings_file=settings)
    session.create_trials(durations=(100, 10), timing='seconds')
    #session.create_trials(durations=(3, 3), timing='frames')
    session.run()
    session.quit()
