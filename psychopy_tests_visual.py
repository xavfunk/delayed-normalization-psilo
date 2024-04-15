from psychopy import visual, core
import numpy as np
from psychopy import event
# from psychopy.hardware import keyboard
import pyglet

win = visual.Window([1200,1200])
# getting a texture
img = visual.ImageStim(win, 'textures/minmax/oneOverF_texture_1_1024_2.bmp',
                             mask = 'raisedCos', maskParams = {'fringeWidth':0.2}) # proportion that will be blurred

button = visual.ImageStim(win, 'button.png')
# makes green, but is actually the reverse of green (pink) probs because the ori is black
button.setColor((255, 128, 255), 'rgb255')
eye = visual.ImageStim(win, 'witness.png')

# found pyglet solution at https://discourse.psychopy.org/t/tracking-key-release/1099
key = pyglet.window.key
keyboard = key.KeyStateHandler()
win.winHandle.push_handlers(keyboard)

# eye indicates TiEst trial about to start
eye.draw()
win.flip()
core.wait(3)


timer = core.Clock()
frame_count = 0

for i in range(120 * 5): # running this for 5 seconds a 120 frames

    # if space gets pressed, we count the time
    if 'space' in event.getKeys(keyList = 'space'):

        # start timer
        timer.reset()
        print(f"key pressed, started timer, {timer.getTime()}")

        # start while loop that will run as long as the key satys pressed
        key_pressed = True
        while key_pressed:
            # if key is pressed, we draw the stimulus
            if keyboard[key.SPACE]: # this evaluates to true as long as the key is pressed
                # draw stimulus
                img.draw()
                # counting frames, helpful for debugging
                frame_count += 1 
                # flip
                win.flip()

            else:
                # if key is released, key_pressed will be set to false, exiting the while loop
                key_pressed = False
                print(f"key released, ending timer {timer.getTime()}")

        # getting the passed time
        time_passed = timer.getTime()

        # printout comparing passed time with counted frames
        print(f"this time has passed with pressed key: {time_passed} and frames: {frame_count} so fc/120 {frame_count/120}")

        # exit trial
        break

    # as long as nothing has been pressed, a green button icon ecourages pressing the button
    
    else:
        button.draw()

    print(f'flipped {i}') # debug message    
    win.flip()

print(f'time was: {timer.getTime()}')
win.flip()


# if 'quit' in keys:
#     core.quit()
# for key in keys:
#     print(key.name, key.rt, key.duration)



# manual_array = np.zeros(96*10)
# manual_array[0] = 1 # on
# manual_array[-1] = -1 # off after 800 ms
# message1 = visual.TextStim(win, text='flip 1')
# message2 = visual.TextStim(win, text='flip 2')
# message = visual.TextStim(win, text='hello')
# rect = visual.Rect(win, .1, .1, pos = (1,1), fillColor ="black")

# print(manual_array)
# img.draw()
# rect.draw()
# core.wait(2.0)

# message1.draw()
# win.flip()

# event.waitKeys(keyList = 't')


# #core.wait(2.0)

# message2.draw()
# win.flip(clearBuffer = False)
# core.wait(2.0)


# for frame in range(len(manual_array)):
#     if manual_array[frame] == 1:
#         img.draw()
#         win.flip(clearBuffer = False)
#         print(win.frames)
#     elif manual_array[frame] == -1:
#         message.draw()
#         win.flip()
#         print(win.frames)
#     else:
#         win.flip(clearBuffer = False)
        
# core.wait(2.0)

# manual_array_full = np.ones(96*10)

# for frame in range(len(manual_array)):
#     if manual_array_full[frame] == 1:
#         img.draw()
#         win.flip()
#         print(win.frames)

#     elif manual_array_full[frame] == -1:
#         message.draw()
#         win.flip()
#         print(win.frames)

#     else:
#         continue

# core.wait(2.0)


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
