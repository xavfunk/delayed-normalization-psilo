import pandas as pd

full_seq = pd.read_csv('trial_sequence_s000.csv')
short = full_seq[:3]
short.to_csv('trial_sequence_s000_short.csv', index = False)
