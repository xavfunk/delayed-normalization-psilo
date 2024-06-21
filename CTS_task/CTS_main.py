import sys
import os

from CTS_sess import DelayedNormSession

def main():
    """
    Simple short script to take CLI arguments, mapping them to the respective
    output folder and string
    """
    
    subject = sys.argv[1] # which subject
    sess =  sys.argv[2] # which session
    task = 'CTS' 
    run = sys.argv[3] # which run    
    # output_str = 'sub-{}_sess-{}_task-{}_run-{}'.format(subject.zfill(2), sess.zfill(2), task, run.zfill(2))
    output_str = f'sub-{subject.zfill(2)}_sess-{sess.zfill(2)}_task-{task}_run-{run.zfill(2)}'
    output_dir = f'{task}_pilot/sub-{subject}/ses-{sess}'
    
    # print(output_dir) 

    # Check if the directory already exists
    if not os.path.exists(output_dir):
        # Create the directory
        os.makedirs(output_dir)
        print("output_dir created successfully!")
    else:
        print("output_dir already exists!")

    settings = os.path.join(os.path.dirname(__file__), 'settings.yml')

    session = DelayedNormSession(output_str, output_dir=output_dir, settings_file=settings, n_trials = None,
                                 eyetracker_on = False, photodiode_check = False, debug = True) # debug drops frames
    session.create_trials()
    session.run()

    session.quit()

if __name__ == '__main__':
    main()