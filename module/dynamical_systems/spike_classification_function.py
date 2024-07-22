import numpy as np


def spike_classification(timeseries):
    # timeseries = s_results[0]

    result_array = np.zeros_like(np.array(timeseries))
    result_array[:-1][(timeseries[:-1] <= 0.75) & (timeseries[1:] > 0.75)] = 2
    result_array[:-1][(timeseries[:-1] <= 0.1) & (timeseries[1:] > 0.1)] = 1

    # plt.plot(result_array)
    # plt.show()


    # Cuts of partial plots
    total_spikes = np.sum(result_array == 1)
    activity_indices = np.where(result_array == 1)[0]
    spike_diff = np.diff(activity_indices)
    mean_diff = spike_diff.mean()
    supra_indices = activity_indices[np.where(spike_diff > (1.3 * mean_diff))[0]+1]
    if len(supra_indices)>2:
        result_array = result_array[supra_indices[1]-1:supra_indices[-1]] # changed from -10 to -1
        # plt.plot(result_array)
        # plt.show()

        total_spikes = np.sum(result_array ==1)
        supra_threshold= 1 if np.sum(result_array==2) > 1 else 0 # changed from 4 to 1

        burst_small= np.where(np.diff(np.where(result_array == 1)) < (1.3 * mean_diff))[1]
        burst_large = np.where(np.diff(np.where(result_array == 2)) < (1.3 * mean_diff))[1]
        sub_threshold_bursting = 1 if np.sum(np.diff(burst_small)<= 2) > 0.5*len(burst_small) else 0
        supra_threshold_bursting = 1 if np.sum(np.diff(burst_large)<= 2) > 0.5*len(burst_large) else 0
    elif len(activity_indices)>2:
        result_array = result_array[activity_indices[1]-1:activity_indices[-1]]
        total_spikes = np.sum(result_array == 1)
        supra_threshold = 1 if np.sum(result_array == 2) > 1 else 0
        sub_threshold_bursting = 0
        supra_threshold_bursting = 0
    else:
        type = 0 #return 0

    if total_spikes < 2:
        type = 0  # No activity
    elif supra_threshold == 1 and supra_threshold_bursting == 1:
        type = 4  # Super threshold bursting
    elif supra_threshold == 1 and supra_threshold_bursting == 0:
        type = 3  # Super threshold spiking
    elif supra_threshold == 0 and sub_threshold_bursting == 1:
        type = 2  # Sub threshold bursting
    elif supra_threshold == 0 and sub_threshold_bursting == 0:
        type = 1  # Sub threshold spiking

    return type