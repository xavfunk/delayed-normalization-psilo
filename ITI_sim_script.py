from ITI_simulations import *
import statsmodels.api as sm

def main():
    pass

if __name__ == '__main__':
    # TODO refactor into functions
    # TODO make nice printouts
    # TODO seperate the random steps
    # fix randomness
    seed = 107
    np.random.seed(seed)
    random.seed(seed)

    # whether the simulated design is constant itis or variable ones
    design_type= 'const' # 'var'

    # simulation variables
    TR = 1.5
    hist_p = .27 if design_type== 'var' else None # geometric rate of ITI histogram # .23
    bins_start = 3 # shortest n_TR per trial
    upsample_factor = int(TR * 100)

    n_seqs = 1000 # how many seqences to simulate (searchspace is 13! = 6227020800)
    n_conds = 13 # number of conditions
    n_noise_reps = 100

    convolution_length = 30
    n_plots = 5 # number of plots ot create
    n_best = 10 # select the n best sequences
    blank_pre = 13 # blanks before starting trials
    blank_post = 18 # blanks after trials are over

    # amount of noise
    SNR = 2.5

    if design_type== 'const':
        n_repeats = 5
        n_trials = 79
        n_blank_trials = n_trials - n_conds * n_repeats
        hist = (np.array([n_conds*n_repeats + n_blank_trials]), np.array([3])) # const hist

    else:
        n_repeats = 3
        n_trials = n_conds * n_repeats
        n_blank_trials = None
        hist = get_geometric_hist(p = hist_p, n_bins = 12, n_trials=n_trials, bins_start=bins_start)

    # print(f"length of trials: {np.dot(*hist)} TRs, {np.dot(*hist) * TR} seconds")
    # print(n_trials)

    n_TRs = np.dot(*hist) + blank_pre + blank_post
    # print(n_TRs)

    # fill an array of size n with randomized seqs
    rand_seqs = np.zeros((n_seqs, n_trials))

    # print(int(np.dot(hist[0], hist[1]*TR) * upsample_factor))
    upsampled_size = int(n_TRs*upsample_factor)
    scaled_events = np.zeros((n_seqs, upsampled_size))  # will contain upsampled and scaled events, just before convolution


    all_events_dict = {}
    convolved_timeseries = np.zeros((n_seqs, upsampled_size))
    # print(convolved_timeseries.shape)
    # print(f'convolved_timeseries down shape: {convolved_timeseries[:,::upsample_factor].shape}')

    # generate random distributions over basic histogram
    # searchspace is 13! = 6227020800 possibilities, optseq simply does a subset of those
    for i in range(len(rand_seqs)):
        rand_seqs[i] = randomize_trials(hist)

    # pad in front and back with blanks
    rand_seqs = np.hstack((np.array([13]*rand_seqs.shape[0]).reshape(-1, 1),
                rand_seqs,
                np.array([18]*rand_seqs.shape[0]).reshape(-1, 1)))

    # print(rand_seqcs)

    for i in range(len(rand_seqs)):

        upsampled = unfold_upsample(rand_seqs[0], upsample_factor)

        # remove "blank onsets" (first and last 1 in upsampled)
        # Find the index of the first occurrence of 1
        first_1_index = np.argmax(upsampled != 0)
        # Find the index of the last occurrence of 1
        last_1_index = upsampled.size - np.argmax(upsampled[::-1] != 0) - 1
        # remove them
        upsampled[first_1_index] = 0
        upsampled[last_1_index] = 0

        scaled_events[i], all_events_dict[i] = scale_amplitudes(upsampled, n_repeats = n_repeats, n_blanks=n_blank_trials)
        convolved_timeseries[i] = convolve_HRF(scaled_events[i], upsample_factor, length=convolution_length)[:upsampled_size]

    noisy_timeseries = add_noise(convolved_timeseries, SNR = SNR)
    conv_full = convolved_timeseries

    # back down to TR
    convolved_timeseries = convolved_timeseries[:, ::upsample_factor]
    noisy_timeseries = noisy_timeseries[:, ::upsample_factor]

    design_matrices = np.zeros((n_seqs, n_TRs, 14))
    # print(design_matrices.shape)
    # print(all_events_dict[0])

    for i in range(len(rand_seqs)):
        design_matrices[i] = make_design_matrix(all_events_dict[i], max_t = n_TRs, upsample_factor=upsample_factor)[0]

    # statistics
    stats_results = {}
    predicted_timeseries = np.zeros_like(noisy_timeseries)

    # result amplitudes ground truth
    # print(np.array(list(trials.values()))[:-1])

    amp_gt = np.array(list(trials.values()))[:-1] if design_type== 'const' else np.array(list(trials.values()))
    


    ## setup score array with shape (n_seqs, n_noise_reps)
    r2_reps = np.zeros((n_seqs, n_noise_reps))
    dist_amp_reps = np.zeros((n_seqs, n_noise_reps))

    
    # add outer loop over j
    for j in range(n_noise_reps):
        # make time series noisy here
        noisy_timeseries = add_noise(conv_full, SNR = SNR)[:, ::upsample_factor]
        # print(noisy_timeseries.shape)

        for i in range(len(rand_seqs)):

            ## GLM
            glm = sm.GLM(noisy_timeseries[i], design_matrices[i], family=sm.families.Gaussian())
            stats_results[i] = glm.fit()
            predicted_timeseries[i] = glm.predict(stats_results[i].params)

            ## ridge
            # ols = sm.OLS(noisy_timeseries[i], design_matrices[i])
            # stats_results[i] = ols.fit_regularized(alpha = .5, L1_wt = 0)
            # predicted_timeseries[i] = ols.predict(stats_results[i].params)

            amp_pred = stats_results[i].params[:-1] if design_type == 'var' else stats_results[i].params[:-2]
            
            distance = np.linalg.norm(amp_gt - amp_pred)

            r2_reps[i, j] = stats_results[i].pseudo_rsquared()
            dist_amp_reps[i, j] = -np.linalg.norm(amp_gt - amp_pred)

            
    ## median-compress the scores to 1D
    r2_reps_med = np.median(r2_reps, axis=1)
    dist_amp_reps_med = np.median(dist_amp_reps, axis=1)
    # print(r2_reps)

    ## visualize
    # plt.hist(r2_reps[0])
    # plt.show()

    # print(r2_reps_med.shape)
    # print(dist_amp_reps_med.shape)


    sort = True
    if sort is True:
        all_r2 = [stats_results[i].pseudo_rsquared() for i in range(len(stats_results))]
        amp_pred = stats_results[i].params[:-1] if design_type == 'var' else stats_results[i].params[:-2]

        dist_amp = [-np.linalg.norm(amp_gt - amp_pred) for i in range(len(stats_results))]

        idxs = n_largest_indices(dist_amp_reps_med, n_best)
    else:
        idxs = range(n_plots)

    fig, axs = plt.subplots(n_plots, 4, figsize = (20, 10))

    for i, j in zip(idxs, range(n_plots)):
        # GT
        plot_trial_sequence(scaled_events[i], all_events_dict[i], axs[j, 0], upsample_factor= upsample_factor)
        axs[j, 0].set_xlim(0, n_TRs)
        axs[j, 0].plot(convolved_timeseries[i])

        # noisy
        plot_trial_sequence(scaled_events[i], all_events_dict[i], axs[j, 1], upsample_factor= upsample_factor)
        axs[j, 1].set_xlim(0, n_TRs)

        axs[j, 1].plot(noisy_timeseries[i])

        # design
        # axs[j, 2].imshow(design_matrices[i][:, :-1], aspect = .05)

        # prediction
        axs[j, 3].plot(noisy_timeseries[i], label = 'observed')
        axs[j, 3].plot(convolved_timeseries[i], label = 'ground truth')
        axs[j, 3].plot(predicted_timeseries[i], label = 'predicted')

        # pseudo rsquared
        axs[j, 3].text(len(predicted_timeseries[1])*.8,1.2, f'$R^2$ : {stats_results[i].pseudo_rsquared():.2f}', bbox=dict(facecolor='white', alpha=0.5))
        axs[j, 3].text(len(predicted_timeseries[1])*.8,1.0, f'$-||amp_g-amp_p||$ : {dist_amp_reps_med[i]:.2f}', bbox=dict(facecolor='white', alpha=0.5))

        axs[j, 3].set_xlim(0, n_TRs)
        amp_pred = stats_results[i].params[:-1] if design_type == 'var' else stats_results[i].params[:-2]

        # betas
        axs[j, 2].scatter(amp_gt, amp_pred)
        axs[j, 2].set_xlabel('amp_gt')
        axs[j, 2].set_ylabel('amp_pred')


        # betas
        # betas = {key : value for key, value in zip(trials.keys(), stats_results[i].params)}
        # plot_trial_sequence(scaled_events[i], all_events_dict[i], trials = betas, ax = axs[j, 3])

        # save event dict
        # print(rand_seqs[i])

        # print(all_events_dict[i])
        # print(all_events_dict[i].values())
        # print(list(zip(rand_seqs[i],all_events_dict[i].values())))



    axs[0, 0].set_title("Ground truth: randomized events of\n different amplitudes convolved with HRF")
    axs[0, 1].set_title(f"adding noise, SNR = {SNR}")
    # axs[0, 2].set_title(f"design matrices")
    axs[0, 2].set_title(f"gt vs pred amp")

    axs[0, 3].set_title(f"predicted timeseries")
    axs[-1, 3].legend(ncols = 3)

    fig.tight_layout()
    plt.show()


    # Saving
    for trial_id, i in enumerate(idxs):
        # only saving one
        if trial_id ==0:
            trial_df_from_simulation(all_events_dict[i], rand_seqs[i], all_r2[i], dist_amp = dist_amp[i], design_type = design_type, trial_id = trial_id, TR=TR, n_conds = n_conds,
                                 SNR=SNR, hist_p = hist_p, bins_start = bins_start, n_searched = n_seqs, n_blanks=n_blank_trials, blank_pre=blank_pre, blank_post=blank_post,
                                 n_best = n_best, save = True, seed = seed, out_dir = '/home/xavfunk/repos/delayed-normalization-psilo/CTS_task/trial_sequences/const_fix')
