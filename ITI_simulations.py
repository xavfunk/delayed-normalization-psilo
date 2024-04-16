import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.special import gamma
import statsmodels.api as sm
import heapq
import pandas as pd
import json
from datetime import datetime
import math
import os

# trials is a mapping between trial types and the corresponding amplitude scaling factor 
trials = {"dur_0" : 0,
         "dur_17" : 1,
         "dur_33" : 2,
         "dur_67" : 3,
         "dur_134" : 4,
        # "dur_267-isi_0" : 5,
         "dur_267" : 5,
         "dur_533" : 6,
         "isi_17" : 5,
         "isi_33" : 5.1,
         "isi_67" : 5.3,
         "isi_134" : 5.5,
         "isi_267" : 5.8,
         "isi_533" : 6}

## predicted from DN model
trials = {"dur_0" : 0,
         "dur_17" : 2.08,
         "dur_33" : 2.57,
         "dur_67" : 3.14,
         "dur_134" : 3.74,
        # "dur_267-isi_0" : 4.47,
         "dur_267" : 4.47, 
         "dur_533" : 5.54,
         "isi_17" : 4.50,
         "isi_33" : 4.53,
         "isi_67" : 4.61,
         "isi_134" : 4.80,
         "isi_267" : 5.25,
         "isi_533" : 6.21}


def make_hist(n_trials = 39):
    """
    takes the basic histogram and potentially shifts things around a bit
    
    TODO implement the actual shifting around
    """
    
    basic_hist = (np.array([20, 10, 5, 2, 1, 1, 0, 0, 0]), np.array([4, 5, 6, 7, 8, 9, 10, 11, 12]))
    
    hist = basic_hist
    
    return hist

def get_geometric_distribution(p, length):
    
    geometric_dist = np.zeros(length)
    geometric_dist[0] = p
    
    for i in range(1, length):
        geometric_dist[i] = geometric_dist[i-1] - (geometric_dist[i-1] * p) 
        
    return geometric_dist

def get_geometric_hist(p, n_bins, n_trials, bins_start = 4, TR=None):
    
    dist = get_geometric_distribution(p, n_bins)
    hist = np.round(dist * n_trials)
    
    n_add = 2
    while np.sum(hist) < n_trials:
        # add one to the first bin with only one
        hist[np.argmax(hist < n_add)] += 1
        # then increase counter
        n_add += 1
    
    bins = np.arange(bins_start, n_bins + bins_start)
    
    # histogram is tuple of (values, bins)
    if TR is None:
        return (hist.astype(int), bins)
    else:
        return (hist.astype(int), bins*TR)

def randomize_trials(hist_tup):
    """
    takes an ITI distribution tuple of (n_ITIs, bins [seconds]/[TR])
    returns a randomized trial sequence
    """
    # first, make a sequence list
    sequence_list = [[tup[1]]*tup[0] for tup in list(zip(hist_tup[0], hist_tup[1]))]
    sequence_list = [val for sublist in sequence_list for val in sublist]

    random.shuffle(sequence_list) # shuffle inplace
    sequence_array = np.array(sequence_list, dtype = int)
    # print('rand_trials_shape: ', sequence_array.shape)
    # print(sequence_array)
    
    return sequence_array

def unfold_upsample(sequence_list, upsample_factor = 1):
    """
    takes a sequence list and then unfolds and upsamples it by some factor
    """

    # unfold
    seq_array = np.zeros(np.sum(sequence_list,  dtype = int))
    # seq_array = np.zeros(int(np.sum(sequence_list,  dtype = int)*upsample_factor*TR)) #
    seq_array = np.zeros(int(np.sum(sequence_list,  dtype = int)*upsample_factor)) #

    # cumsum gives basically the end indeces of the trials, we drop the last and add a 0 in front 
    idxs = np.cumsum(np.hstack((0,sequence_list)))[:-1] 
    # idxs *= upsample_factor*TR#
    idxs *= upsample_factor

    seq_array[idxs.astype(int)] = 1
    
    # # upsample
    # seq_array = np.repeat(seq_array, upsample_factor)
    # for i, element in enumerate(seq_array):
    #     if element == 1: # detect a one

    #         for k in range(1,upsample_factor): # set the following k to 0
    #             #print(k)
    #             seq_array[i+k] = 0
    
    return seq_array

    
def scale_amplitudes(seq_array, trials = trials, n_repeats = 3, n_blanks = None):
    """
    takes a trial sequence and scales the signal amplitudes according to some experimental design 
    trials is a dict with the different trial types and their amplitudes
    
    returns appropriately scaled trials and dict specifying when each trial event happened
    """
    event_types = {}
    
    if n_blanks is None:
        design = list(trials.keys()) * n_repeats 
    else:
        # make design with blanks
        design = list(trials.keys()) * n_repeats + ['blank_0'] * n_blanks
        # add blank condition to trials dict
        trials['blank_0'] = 0

    random.shuffle(design)
    
    j = 0 # index for design
    out_array = np.zeros_like(seq_array)
#     print(seq_array, trials)
    # match repeats with 1's
    for i, event in enumerate(seq_array):
        if event == 1:  # check if event occured
#             print(f"at index {i}, found value {event}, putting value {event} * {trials[design[j]]}, since this is {design[j]}")
            # multiply with trial amplitude
            out_array[i] = event * trials[design[j]]
            # save i:event
            event_types[i] = design[j]
            # increase counter
            j += 1

    # print("scale_ampl_out: ", out_array.shape)
    return out_array, event_types

def plot_trial_sequence(scaled_events, events_dict, ax = None, upsample_factor = 1):
    # if an ax was passed, don't return it 
    return_fig_ax = True if ax is None else False
    
    # Create a list of indices where the value is 1
    events_indices = list(events_dict.keys())
    # downsample
    events_indices = [i//upsample_factor for i in events_indices]

    # Plotting the vertical lines at the indices where the value is 1
    if ax is None:
        fig, ax = plt.subplots()
    
    for i in events_indices:
        ax.vlines(i, ymin=0, ymax=scaled_events[::upsample_factor][i], colors='r', linestyles='solid')
    ax.set_xticks(events_indices)
    ax.set_xticklabels(list(events_dict.values()), rotation = 90, size = 7)
    
    if return_fig_ax:
        return fig, ax
    else:
        return


def canHRF(t, a1=6, a2=1, a3=16, a4=1, alpha=1/6):
    """
    makes a canonical two-gamma HRF according to 
    
    $$
    h(t) = \frac{t^{a_1−1}e^{-a_2t}}{\Gamma(a_1)} - \alpha \frac{t^{a_3−1}e^{-a_4t}}{\Gamma(a_3)},
    $$
    
    t is the input time
    a1, a2, a3, a4 are shape params
    alpha controls the ratio of response to undershoot
    
    some plausible parameters are: alpha = 1/6, a1 = 6, 
    a3 = 16 and a2 = a4 = 1, see defaults, 
    which give a nice hrf returning to baseline after 25s
    """
    
    hrf = (t**(a1-1) * np.exp(-a2*t))/gamma(a1) - alpha * (t**(a3-1) * np.exp(-a4*t))/gamma(a3)
    return hrf


def convolve_HRF(trials_scaled, upsample_factor, length=30, **kwargs):
    """
    takes a (scaled) trial sequence and convolves it with a twogamma hrf 
    """
    
    t = np.linspace(0, length, length * upsample_factor)
    hrf = canHRF(t, **kwargs)
    trials_scaled_convolved = np.convolve(hrf, trials_scaled, mode='full')

    return trials_scaled_convolved


def make_design_matrix(events_dict, max_t = 300, upsample_factor = 1, conds = ["dur_0", "dur_17", "dur_33", "dur_67", "dur_134",
                                                          "dur_267", "dur_533", "isi_17", "isi_33",
                                                          "isi_67", "isi_134", "isi_267", "isi_533"]):
    
    # keep track of conds : tps
    regr_dict = {}
    
    # setup design matrix, shape max_t x len(conds + 1) to keep space for intercept
    design_matrix = np.zeros((max_t, len(conds) + 1))
    
    # for each cond
    for i, cond in enumerate(conds):
        # find timepoints of stimulation and downsample
        tps = [key//upsample_factor for key, val in events_dict.items() if val == cond]
        regr_dict[cond] = tps
        
        # setup regressor
        regressor = np.zeros(max_t)
        regressor[tps] = 1
        
        # convolve with HRF, cut to max_t, put into matrix
        conv_regressor = convolve_HRF(regressor, 1)[:max_t]
        design_matrix[:,i] = conv_regressor.T 
        
    # column of ones for intercept
    design_matrix[:,-1] = 1
    
    return design_matrix, regr_dict

def add_noise(Y, SNR = 1):
    return Y + 1/SNR * np.random.normal(0, 1, size = Y.shape)

def n_largest_indices(lst, n):
    indexed_lst = [(-value, index) for index, value in enumerate(lst)]
    heapq.heapify(indexed_lst)
    largest_indices = []
    for _ in range(n):
        largest_indices.append(heapq.heappop(indexed_lst)[1])
    return largest_indices

def trial_df_from_simulation(events_dict, rand_seq, r2, dist_amp, design_type, trial_id = None, TR=1.32, n_conds = 13, SNR=4, hist_p = None, n_blanks = None, blank_pre=None, blank_post=None,
                             bins_start = None, n_searched = None, n_best = None, save = False, seed=None, out_dir = 'trial_sequences'):
    """
    computes a trial df alongside metadata from simulation results, from
    events_dict, rand_seq and corresponding r2
    """
    # define ms to frame mapping
    ms_to_frame_120 = {0:0, 17:2, 33:4, 67:8, 134:16, 267:32, 533:64}
    
    event_type = [type_cond.split('_')[0] for type_cond in events_dict.values()] 

    # Zhou took a different texture for each trial
    texture_id = [i for i in range(len(events_dict))]
    random.shuffle(texture_id)
    iti_TR = rand_seq[1:-1] # getting sliced as it still contains the blanks
    iti_s = [iti_TR_i * TR for iti_TR_i in iti_TR] 
    cond_ms = [float(type_cond.split('_')[1]) for type_cond in events_dict.values()]
    cond_frames = [ms_to_frame_120[int(ms)] for ms in cond_ms]


    # print(len(event_type))
    # print(len(iti_s))
    # print(len(iti_TR))
    # print(len(cond_frames))
    # print(len(cond_ms))
    # print(len(texture_id))
    
    df = pd.DataFrame({
        'type':event_type, 
        'iti_s':iti_s,
        'iti_TR': iti_TR,
        'cond_frames':cond_frames,
        'cond_ms':cond_ms,
        'texture_id':texture_id})

    # metadata
    n_trials = df.shape[0]
    length_seconds = np.sum(df.iti_s)
    length_TR = np.sum(df.iti_TR)

    # Get the current date
    current_date = datetime.now().date()

    metadata = {
        "title": "",
        "description": "",
        "created_at": current_date.strftime('%Y-%m-%d'),
        "TR": TR,
        "len_seconds":length_seconds,
        "len_TR":length_TR,
        "n_trials":n_trials,
        "design_type" : design_type,
        # simulation parameters
        "r^2": r2,
        "dist_amp" : dist_amp,
        "n_conditions": n_conds,
        "full_searchspace":math.factorial(n_conds),
        "searched_searchspace":n_searched,
        "histogram_p": hist_p,
        "n_blank_trials_in_sequence" : n_blanks,
        "n_blanks_pre" : blank_pre,
        "n_blanks_post" : blank_post,
        "histogram_start_bin": bins_start,
        "simulation_SNR": SNR,
        "n_best": n_best,
        "seed": seed
        }
    
    if save:
        filename = "trial_sequence_c{}".format(str(trial_id).zfill(3))

        # save df
        out_dir = out_dir + f'/d_{design_type}_p_{hist_p}_s_{seed}' if design_type == 'var' else out_dir + f'/d_{design_type}_b_{n_blanks}_s_{seed}'
        # root = root + '/c_{}_s_{}'.format(4, seed)
        os.makedirs(out_dir, exist_ok=True)
        df_path = os.path.join(out_dir, filename + ".csv")
        print(df_path)
        df.to_csv(df_path, index=False)

        # save metadata
        json_path = os.path.join(out_dir, filename + ".json")
        with open(json_path, "w") as json_file:
            json.dump(metadata, json_file, indent=4)

    return df, metadata