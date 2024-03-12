import psychopy
print(psychopy.__version__)

import psychopy.core as core
from psychopy.sound import Microphone
import matplotlib.pyplot as plt
import numpy as np
from psychopy import visual

win = visual.Window([1200,1200], fullscr = True, screen = 1)
white_square = visual.Rect(win, 1, 1, pos = (0,0))
black_square = visual.Rect(win, 1, 1, pos = (0,0), fillColor = 'black')

black_square.draw()
win.flip()
timer = core.Clock()

core.wait(10.0) 

mic = Microphone(streamBufferSecs=6.0)  # open the microphone
# print("time is: {}".format(timer.getTime()))
print('Starting recording')

# core.wait(1.0)  # wait 1 second

# for i in range(1,11):

#     white_square.draw()
#     win.flip()

#     core.wait(.1) 
#     black_square.draw()
#     win.flip()
#     core.wait(.1)

flicker_frequency = 12
current_frame = 0
target_time = 1.0
start_time = timer.getTime()
# Input latency 4.988662 ms
input_latency = 0.004988662

mic.start()  # start recording
while (timer.getTime() - start_time) < target_time:
    #  if current_frame == 0:
    #     mic.start()
     
     # When to draw stimuli
     black_square.draw()

     if current_frame % flicker_frequency < flicker_frequency/2:
        white_square.draw()
     # Show whatever has been drawn. Sometimes the stimuli, other times a blank screen. 
     # flip() waits for the next monitor update so the loop is time-locked to the screen here.
     win.flip()
     current_frame += 1 

# white_square.draw()
# win.flip()
# core.wait(.01)  
# black_square.draw()
# win.flip()

# core.wait(5-3)
mic.stop()  # stop recording
end_time = timer.getTime()
print("eclipsed time: {}".format(end_time-start_time))


audioClip = mic.getRecording()

print(audioClip.duration) 
print(audioClip)  
print(type(audioClip)) 
print(audioClip.__dict__) 
print(audioClip.samples)  
print(audioClip.sampleRateHz * audioClip.duration)  

t = np.linspace(0, audioClip.duration, int(audioClip.sampleRateHz * audioClip.duration))
control = np.arange(target_time * 120) % flicker_frequency < flicker_frequency/2
t2 = np.linspace(0, target_time, len(control))
# plt.plot(t, audioClip.samples)

fig, axs = plt.subplots(nrows = 2)
twin0 = axs[0].twinx()
twin1 = axs[1].twinx()


axs[0].plot(t, audioClip.samples[:,0])
# axs[0].plot(t[int(np.round(input_latency* audioClip.sampleRateHz)):], audioClip.samples[:,0][int(np.round(input_latency*audioClip.sampleRateHz)):])
twin0.plot(t2, control, color ='red')

axs[1].plot(t, audioClip.samples[:,1])
twin1.plot(t2, control, color = 'red')

fig.suptitle('frames: {}, frequency: {}, target time: {},\nrec. duration: {}'.format(current_frame, flicker_frequency, target_time, audioClip.duration))
fig.savefig('audio_samples.png')

############3

# import psychopy.core as core
# import sounddevice as sd
# import matplotlib.pyplot as plt

# duration = 5.5  # seconds
# fs = 48000

# myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
# core.wait(5.0)  # wait 10 seconds

# print(myrecording)  # should be ~10 seconds
# print(myrecording.shape[0]/fs)  # should be ~10 seconds
# print(type(myrecording))  # should be ~10 seconds

# plt.plot([1, 2, 4], [1, 2, 3])

# plt.plot(myrecording)

# audioClip.save('test.wav')  # save the recorded audio as a 'wav' file
