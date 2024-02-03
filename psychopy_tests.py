from psychopy import visual, core
win = visual.Window([400,400])
message = visual.TextStim(win, text='hello')
#message.autoDraw = True  # Automatically draw every frame
message.draw()
win.flip()
core.wait(2.0)
message.text = 'world'  # Change properties of existing stim
message.draw()
win.flip(clearBuffer=False)
core.wait(2.0)
win.flip()

# win.flip(clearBuffer=False)
core.wait(2.0)
# win.flip(clearBuffer=False)
