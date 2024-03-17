import psychopy
print(psychopy.__version__)

import psychopy.core as core
from psychopy.sound import Microphone
import matplotlib.pyplot as plt
import numpy as np
from psychopy import visual
import time
from scipy.signal import find_peaks
import pandas as pd

# How often each condition should be repeated
n_repeats = 50

## make experimental conditions
stim_conds = [0, 2, 4, 8, 16, 32, 64] # frames in 120 FPS, either duration or isi times
total_duration = 96 # (<800 ms in total) in exp design; 800 ms = .8*120 = 96 frames
var_duration = np.vstack([np.hstack((np.ones(stim_cond), # showing stimulus
                                    np.zeros(total_duration - stim_cond))) # no stimulus for the remaining frames
                                    for stim_cond in stim_conds])


# these dicts are integer indexable with the current number of trial frames 
var_dur_dict = {dur:frames for dur, frames in zip(stim_conds, var_duration)}

var_duration_flip = np.zeros((len(stim_conds), total_duration))
for i in range(len(stim_conds)):
   #print(i)
   var_duration_flip[i, 0] = 1 # on
   var_duration_flip[i, stim_conds[i]] = -1 # off
   
   if i == 0:
         var_duration_flip[i, 0] = 0

# init window
win = visual.Window([1200,1200], fullscr = True, screen = 1)

# make squares in the bottom right corner for the diode
white_square = visual.Rect(win, 1, 1, pos = (1,-1))
black_square = visual.Rect(win, 1, 1, pos = (1,-1), fillColor = 'black')

# start with a black square
black_square.draw()
win.flip()

# initialize timer
timer = core.Clock()
core.wait(2.0) 

mic = Microphone(streamBufferSecs=6.0)  # open the microphone

## code for making flicker stimulus
# flicker_frequency = 12
# target_time = 96 * 1/120
# sequence = np.arange(target_time * 120) % flicker_frequency < flicker_frequency/2

# prepare lists to store results 
conditions = []
frames = []
recording_durations = []
delta_peaks = []
n_peaks_found = []

## loop over experimental conditions
# for each condition (starting at index 1 leaves out the 0-frame condition)
for k in range(1, len(var_duration_flip)):
   # get the sequence for that condition
   sequence = var_duration_flip[k]

   # for each repeat
   for j in range(n_repeats):
      
      current_frame = 0
      
      # prepare first flip, we start with white square
      white_square.draw()
      # start recording asap after first flip
      win.callOnFlip(mic.start)
      # get start time
      start_time = timer.getTime()
      # do first flip
      win.flip()

      current_frame += 1

      # for each frame in the sequence
      for i in range(1, len(sequence)):
         # draw a white square if sequence[current_frame] is not 0
         if sequence[current_frame]:
            white_square.draw()
         # otherwise, draw black square
         else:
            black_square.draw()
         win.flip()

         current_frame += 1

      mic.stop()  # stop recording
      end_time = timer.getTime()
      # print("eclipsed time: {}".format(end_time-start_time))

      # get the recording of this iteration 
      audioClip = mic.getRecording()

      ## find the peaks in the second channel
      # height and distance are heuristics that might need adjustments 
      peaks, _ = find_peaks(audioClip.samples[:,1], height = .3, distance = audioClip.sampleRateHz*1/120) 
      # print((peaks[1] - peaks[0])/audioClip.sampleRateHz)

      # store results into lists
      conditions.append(stim_conds[k])
      frames.append(current_frame)
      recording_durations.append(audioClip.duration)
      delta_peaks.append((peaks[1] - peaks[0])/audioClip.sampleRateHz)
      n_peaks_found.append(len(peaks))

# convert lists into df
results = pd.DataFrame({'conditions':conditions,
                        'frames': frames,
                        'duration': recording_durations,
                        'delta_peaks': delta_peaks,
                        'n_peaks_found': n_peaks_found})

# save results
results.to_csv('/photodiode_test_results/timing_test_results_{}_dur_psychophys__.csv'.format(n_repeats), index = False)

# print(audioClip.duration) 
# print(audioClip)  
# print(type(audioClip)) 
# print(audioClip.__dict__) 
# print(audioClip.samples)  
# print(audioClip.sampleRateHz)
# print(audioClip.sampleRateHz * audioClip.duration)  

# # save the recording 
# timestamp = time.time()
# with open('audio_recordings/audio_recording__{}_{}_{}.npy'.format(flicker_frequency, target_time, timestamp), 'wb') as f:
#     np.save(f, audioClip.samples)

# # find peaks
# peaks, _ = find_peaks(audioClip.samples[:,0], height = .5, distance = audioClip.sampleRateHz*1/120) 

# # plotting
# t = np.linspace(0, audioClip.duration, int(np.round(audioClip.sampleRateHz * audioClip.duration)))
# control = np.arange(target_time * 120) % flicker_frequency < flicker_frequency/2
# t2 = np.linspace(0, target_time, len(control))
# # plt.plot(t, audioClip.samples)

# fig, axs = plt.subplots(nrows = 2)
# twin0 = axs[0].twinx()
# twin1 = axs[1].twinx()

# # input latency 4.988662 ms
# input_latency = 0.004988662

# axs[0].plot(t, audioClip.samples[:,0])
# # axs[0].plot(t[int(np.round(input_latency* audioClip.sampleRateHz)):], audioClip.samples[:,0][int(np.round(input_latency*audioClip.sampleRateHz)):])
# # twin0.plot(t2, control, color ='red')
# # twin0.vlines(input_latency+1/120,0,1)
# axs[0].plot(t[peaks], audioClip.samples[:,0][peaks])
# print(len(peaks))
# print((peaks[1] - peaks[0])/audioClip.sampleRateHz)

# axs[1].plot(t, audioClip.samples[:,1])
# twin1.plot(t2, control, color = 'red')

# # for ax ticks
# stepsize = 1/120 * flicker_frequency
# max_tick = target_time + stepsize
# # twin1.vlines(input_latency+1/120,0,1)


# for ax in axs:
#    ax.set_xticks(np.arange(0,max_tick,stepsize))

# fig.suptitle('frames: {}, frequency: {}, target time: {},\nrec. duration: {:.2f}'.format(current_frame, flicker_frequency, target_time, audioClip.duration))
# fig.savefig('audio_recordings/audio_plot_{}_{}_{}.png'.format(flicker_frequency, target_time, timestamp))