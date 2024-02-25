import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.special import gamma
import statsmodels.api as sm


# trials is a mapping between trial types and the corresponding amplitude scaling factor 
trials = {"dur_0" : 0,
         "dur_17" : 1,
         "dur_33" : 2,
         "dur_67" : 3,
         "dur_134" : 4,
#         "dur_267-isi_0" : 5,
         "dur_267" : 5, # for demonstration purposes 
         "dur_533" : 6,
         "isi_17" : 5,
         "isi_33" : 5.1,
         "isi_67" : 5.3,
         "isi_134" : 5.5,
         "isi_267" : 5.8,
         "isi_533" : 6}


def make_hist(n_trials = 39):
    """
    takes the basic histogram and potentially shifts things around a bit
    
    TODO implement the actual shifting around
    """
    
    basic_hist = (np.array([20, 10, 5, 2, 1, 1, 0, 0, 0]), np.array([4, 5, 6, 7, 8, 9, 10, 11, 12]))
    
    hist = basic_hist
    
    return hist

def randomize_trials(hist_tup):
    """
    takes an ITI distribution tuple of (n_ITIs, bins [seconds])
    returns a randomized trial sequence
    """
    # first, make a sequence list
    sequence_list = [[tup[1]]*tup[0] for tup in list(zip(hist_tup[0], hist_tup[1]))]
    sequence_list = [val for sublist in sequence_list for val in sublist]

    random.shuffle(sequence_list) # shuffle inplace
        
    return np.array(sequence_list, dtype = int)

def unfold_upsample(sequence_list, upsample_factor = 10):
    """
    takes a sequence list and then unfolds and upsamples it by some factor
    """
    # unfold
    seq_array = np.zeros(np.sum(sequence_list,  dtype = int))
    
    # cumsum gives basically the end indeces of the trials, we drop the last and add a 0 in front 
    idxs = np.cumsum(np.hstack((0,sequence_list)))[:-1] # cumsum gives basically the end times of the trials, we drop the last and add a 0 in front 
    seq_array[idxs.astype(int)] = 1
    
    # upsample
    seq_array = np.repeat(seq_array, upsample_factor)
    for i, element in enumerate(seq_array):
        if element == 1: # detect a one

            for k in range(1,upsample_factor): # set the following k to 0
                #print(k)
                seq_array[i+k] = 0
    
    return seq_array

    
def scale_amplitudes(seq_array, trials = trials, n_repeats = 3):
    """
    takes a trial sequence and scales the signal amplitudes according to some experimental design 
    trials is a dict with the different trial types and their amplitudes
    
    returns appropriately scaled trials and dict specifying when each trial event happened
    """
    event_types = {}
    design = list(trials.keys()) * n_repeats # 3 repeats
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
            
    return out_array, event_types

def plot_trial_sequence(scaled_events, events_dict, ax = None):
    # if an ax was passed, don't return it 
    return_fig_ax = True if ax is None else False
    
    # Create a list of indices where the value is 1
    events_indices = list(events_dict.keys())

    # Plotting the vertical lines at the indices where the value is 1
    if ax is None:
        fig, ax = plt.subplots()
    
    for i in events_indices:
        ax.vlines(i, ymin=0, ymax=scaled_events[i], colors='r', linestyles='solid', label='Vertical Lines (1s)')
    ax.set_xticks(events_indices)
    ax.set_xticklabels(list(events_dict.values()), rotation = 90, size = 8)
    
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
    takes a (scaled) trial sequence and convolved it with a twogamma hrf 
    """
    
    t = np.linspace(0, length, length * upsample_factor)
    hrf = canHRF(t)
    trials_scaled_convolved = np.convolve(hrf, trials_scaled, mode='full')

    return trials_scaled_convolved


def make_design_matrix(events_dict, max_t = 300, conds = ["dur_0", "dur_17", "dur_33", "dur_67", "dur_134",
                                                          "dur_267-isi_0", "dur_533", "isi_17", "isi_33",
                                                          "isi_67", "isi_134", "isi_267", "isi_533"]):
    
    # keep track of conds : tps
    regr_dict = {}
    
    # setup design matrix, shape max_t x len(conds + 1) to keep space for intercept
    design_matrix = np.zeros((max_t, len(conds) + 1))
    
    # for each cond
    for i, cond in enumerate(conds):
        # find timepoints of stimulation
        tps = [key for key, val in events_dict.items() if val == cond]
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



if __name__ == '__main__':
    ## TODO refactor to seperate plotting from calculations, possibly into either function or a class
    # TODO generate random histograms
    # for now fixed
    upsample_factor = 1 
    n_trials = 39
    hist = make_hist(n_trials = 39)
    convolution_length = 30

    # fill an array of size n with randomized seqs
    n_seqs = 100
    rand_seqs = np.zeros((n_seqs, n_trials))
    scaled_events = np.zeros((n_seqs, np.dot(*hist) * upsample_factor))
    all_events_dict = {}
    convolved_timeseries = np.zeros((n_seqs, (np.dot(*hist) + convolution_length-1) * upsample_factor))

    # generate random distributions over basic histogram
    # searchspace is 13! = 6227020800 possibilities, optseq simply does a subset of those
    for i in range(len(rand_seqs)):
        rand_seqs[i] = randomize_trials(hist)
        upsampled = unfold_upsample(rand_seqs[i], upsample_factor)
        scaled_events[i], all_events_dict[i] = scale_amplitudes(upsampled, trials = trials)
        convolved_timeseries[i] = convolve_HRF(scaled_events[i], upsample_factor, length=convolution_length)
    
    i = 0
    n_plots = 5
    fig, axs = plt.subplots(n_plots, 4, figsize = (10, 20))
    
    for i in range(n_plots):
        if i == 0:
            axs[i, 0].set_title("Ground truth: randomized events of\n different amplitudes convolved with HRF") 

        plot_trial_sequence(scaled_events[i], all_events_dict[i], axs[i, 0])
        axs[i, 0].set_xlim(-5, np.dot(*hist) + convolution_length)
        axs[i, 0].plot(convolved_timeseries[i])
    # fig.suptitle("Ground truth: randomized events of\n different amplitudes convolved with HRF")

    # add random noises for each
    SNR = 4
    noisy_timeseries = add_noise(convolved_timeseries, SNR = SNR)

    for i in range(n_plots):
        if i == 0:
            axs[0, 1].set_title(f"adding noise, SNR = {SNR}") 

        plot_trial_sequence(scaled_events[i], all_events_dict[i], axs[i, 1])
        axs[i, 1].set_xlim(-5, np.dot(*hist) + convolution_length)
        axs[i, 1].plot(noisy_timeseries[i])

    # model them
        
    # design matrix
    n_conds = 13
    design_matrices = np.zeros((n_seqs, (np.dot(*hist) + convolution_length)-1,14))
    
    for i in range(len(rand_seqs)):
        design_matrices[i] = make_design_matrix(all_events_dict[i], (np.dot(*hist) + convolution_length)-1)[0]
    for i in range(n_plots):
        axs[i, 2].imshow(design_matrices[i][:, :-1], aspect = .05)
    
    # statistics
    glm_results = {}
    predicted_timeseries = np.zeros_like(noisy_timeseries)
    for i in range(len(rand_seqs)):
        glm = sm.GLM(noisy_timeseries[i], design_matrices[i], family=sm.families.Gaussian())
        glm_results[i] = glm.fit()
        predicted_timeseries[i] = glm.predict(glm_results[i].params)


    # plot prediction
    for i in range(n_plots):
        axs[i, 3].plot(noisy_timeseries[i])
        axs[i, 3].plot(predicted_timeseries[i])
        axs[i, 3].text(130,1.2, f'$R^2$ : {glm_results[i].pseudo_rsquared():.2f}', bbox=dict(facecolor='white', alpha=0.5))

    fig.tight_layout()
    plt.show()

    # for each histogram, find the 12 best 
    # look into results to find best scores and keep idx, then do the plotting with the best ones
    
