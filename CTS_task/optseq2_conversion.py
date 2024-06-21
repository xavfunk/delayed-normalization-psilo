import numpy as np
import pandas as pd
import glob
import os
import random

## grab files
directory = 'trial_sequences/optseq2_out'
target_directory = 'trial_sequences/tr-200_min-3_tsearch-12'
pattern = 'tr-200_min-3_tsearch-12*.par'
texture_id = [i for i in range(160)] # there are 160 density-4 snakes

# define mapping frames to ms
frames_ms_mapping = {0 : 0, 2 : 17, 4 : 33, 8 : 67, 16 : 134, 32 : 267, 64 : 533}

files = glob.glob(os.path.join(directory, pattern))
print(files)
for file in files:

    with open(file, 'r') as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        split_line = line.split(' ')
        vals = [val.strip() for val in split_line if (val != '') & (val != '\n')]
        for i in range(len(vals)-1):
            vals[i] = np.round(float(vals[i]), 2)
        
        # adding 0s to itis
        if vals[-1] == 'NULL':
            rows[-1][2] = np.round(rows[-1][2] + vals[2], 2)
            # uncomment to debug
    #         rows.append(vals)
        
        else:
            rows.append(vals)
        
    iti_df = pd.DataFrame(rows, columns = ['time', 'id', 'iti_s', 'unknown', 'type'])
    iti_df['time'] = iti_df['time'].astype(float)
    iti_df['iti_s'] = iti_df['iti_s'].astype(float)
    iti_df['iti_TR'] = np.round(iti_df['iti_s']/1.6).astype(int)
    iti_df['cond_frames'] = [int(val.split('_')[1]) for val in iti_df['type']]
    iti_df['type'] = [val.split('_')[0] for val in iti_df['type']]
    iti_df = iti_df.drop(['id', 'unknown'], axis = 1)
    iti_df['cond_ms'] = [frames_ms_mapping[val] for val in iti_df['cond_frames']]
    
    random.shuffle(texture_id)
    iti_df['texture_id'] = texture_id[:39]

    # check
    print(iti_df)

    # save
    try:
        os.mkdir(target_directory)
    except FileExistsError:
        pass
    path = os.path.join(target_directory, f'tr-200_min-3_tsearch-12_{file.split("-")[-1].split(".")[0]}.csv')
    iti_df.to_csv(path, index = False)
