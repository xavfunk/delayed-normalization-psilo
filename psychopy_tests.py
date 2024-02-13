from psychopy import visual, core
import numpy as np

win = visual.Window([1200,1200])
img = visual.ImageStim(win, 'textures/oneOverF_texture_1_1024_2.bmp',
                             mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred

manual_array = np.zeros(96*10)
manual_array[0] = 1 # on
manual_array[-1] = -1 # off after 800 ms
message1 = visual.TextStim(win, text='flip 1')
message2 = visual.TextStim(win, text='flip 2')
message = visual.TextStim(win, text='hello')
rect = visual.Rect(win, .1, .1, pos = (1,1), fillColor ="black")

print(manual_array)
img.draw()
rect.draw()
core.wait(2.0)

message1.draw()
win.flip()

core.wait(2.0)

message2.draw()
win.flip(clearBuffer = False)
core.wait(2.0)


for frame in range(len(manual_array)):
    if manual_array[frame] == 1:
        img.draw()
        win.flip(clearBuffer = False)
        print(win.frames)
    elif manual_array[frame] == -1:
        message.draw()
        win.flip()
        print(win.frames)
    else:
        win.flip(clearBuffer = False)
        
core.wait(2.0)

manual_array_full = np.ones(96*10)

for frame in range(len(manual_array)):
    if manual_array_full[frame] == 1:
        img.draw()
        win.flip()
        print(win.frames)

    elif manual_array_full[frame] == -1:
        message.draw()
        win.flip()
        print(win.frames)

    else:
        continue

core.wait(2.0)


# def draw_return():
#     """ Draws stimuli and return flipping flag"""
#     if self.phase == 0: # prep time
#         # self.overt_counter_prep.setText(self.session.nr_frames) 
#         # self.overt_counter_prep.draw() 
#         self.session.default_fix.draw()

#         # self.wait_print_t()
#         return 1

#     elif self.phase == 1: # we are in stimulus presentation
        
#         if self.frames[self.session.nr_frames] == 1:

#             # draw texture
#             self.img.draw()
#             # draw fixation TODO confirm we want this
#             self.session.default_fix.draw()
            
#             return 1 

#         elif self.frames[self.session.nr_frames] == -1:
#             self.session.default_fix.draw()
#             return 1

#         else:
#             return 0


#     else: # we are in iti
#         # draw fixation
#         self.session.default_fix.draw()
#         return 1


# message = visual.TextStim(win, text='hello')
# #message.autoDraw = True  # Automatically draw every frame
# # message.draw()
# img.draw()
# win.flip()
# core.wait(2.0)
# message.text = 'world'  # Change properties of existing stim
# message.draw()
# win.flip(clearBuffer=False)
# core.wait(2.0)
# win.flip()

# # win.flip(clearBuffer=False)
# core.wait(2.0)
# # win.flip(clearBuffer=False)
