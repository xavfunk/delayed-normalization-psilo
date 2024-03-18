import yaml

from psychopy import core
from psychopy.sound import Sound
from psychopy.visual import Rect, TextStim, ImageStim

from psychopy.hardware.emulator import SyncGenerator
from psychopy.visual import Window, TextStim
from psychopy.event import waitKeys, Mouse
from psychopy.monitors import Monitor
from psychopy import logging
from psychopy import prefs as psychopy_prefs
from exptools2.stimuli import create_circle_fixation


with open("settings.yml", "r", encoding="utf8") as f_in:
    settings = yaml.safe_load(f_in)


monitor = Monitor(**settings["monitor"])

win = Window(monitor=monitor.name, **settings["window"])

# get the 1/f texture
img = ImageStim(win, './textures/rms/oneOverF_texture_1_1024_52.bmp', size = 1,
                        mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred

img.draw()
win.flip()
core.wait(5)
