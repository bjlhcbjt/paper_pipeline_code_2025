#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mne
import numpy as np

def compute_grand_average(epochs_list, channel):
    """
    Return grand-average array for a single channel across subjects.
    """
    return np.mean([ep.copy().pick_channels([channel]).average().data for ep in epochs_list], axis=0)

def compute_extremes(evoked, windows):
    """
    Find extremum (max for P*, min for N*) within given windows per component.
    Returns: dict[component] = (extreme_value, extreme_time)
    """
    extreme_values = {}
    for component, window in windows.items():
        tmin, tmax = window
        start, stop = evoked.time_as_index([tmin, tmax])
        data = evoked.data[0, start:stop]
        is_positive = component.endswith('P200') or component.endswith('P300')
        extreme_value = np.max(data) if is_positive else np.min(data)
        extreme_index = np.argmax(data) if is_positive else np.argmin(data)
        extreme_time = evoked.times[start + extreme_index]
        extreme_values[component] = (extreme_value, extreme_time)
    return extreme_values

def update_time_windows(evoked, windows):
    """
    Update time windows around detected peaks ONLY for P300 components.
    Returns: dict with updated windows for P300 keys.
    """
    updated_time_windows = {}
    
    for ch_comp, (tmin, tmax) in windows.items():
        ch, comp = ch_comp.split('_')
        
        if comp == 'P300':
            mask = np.logical_and(evoked.times >= tmin, evoked.times <= tmax)
            data = evoked.copy().pick_channels([ch]).data[0]

            peak_index = np.argmax(data[mask])
            peak_time = evoked.times[mask][peak_index]

            updated_time_windows[ch_comp] = (peak_time - 0.025, peak_time + 0.025)
            
    return updated_time_windows

def bootstrap_sampling(channel_data, n_bootstraps, func):
    """
    bootstrap: resample trials (axis=0), apply 'func' to a bootstrap sample,
    """
    bootstrap_results = []
    N = channel_data.shape[0]

    for _ in range(n_bootstraps):
        sample_indices = np.random.choice(N, N, replace=True)
        bootstrap_sample = channel_data[sample_indices]
        bootstrap_results.append(func(bootstrap_sample))

    return np.std(bootstrap_results, ddof=1)

def mean_amplitude_sme(epochs, time_windows, n_bootstraps):
    """
    SEM of mean amplitude within group-level time_windows
    """
    sem_values_mean = {}
    results = []

    for ch_comp, (tmin_peak, tmax_peak) in time_windows.items():
        epochs_cropped = epochs.copy().crop(tmin=tmin_peak, tmax=tmax_peak)
        channel_data = epochs_cropped.copy().pick_channels([ch_comp.split('_')[0]]).get_data()[:, 0, :]

        sem = bootstrap_sampling(channel_data, n_bootstraps, np.mean)
        sem_values_mean[ch_comp] = {'sem': sem}
        results.append(f"{ch_comp} Mean Amplitude SEM: {sem:.4f}")

    return sem_values_mean, results

def adaptive_sme(epochs, updated_time_windows, n_bootstraps):
    """
    SEM using individualised updated_time_windows (for P300).
    """
    results = []

    for ch_comp, (tmin_peak, tmax_peak) in updated_time_windows.items():
        _, comp = ch_comp.split('_')

        if comp == 'P300':
            epochs_cropped = epochs.copy().crop(tmin=tmin_peak, tmax=tmax_peak)
            channel_data = epochs_cropped.copy().pick_channels([ch_comp.split('_')[0]]).get_data()[:, 0, :]

            sem = bootstrap_sampling(channel_data, n_bootstraps, np.mean)
            results.append(f"{ch_comp} Adaptive Amplitude SEM: {sem:.4f}")

    return results

def peak_amplitude_sme(epochs, windows, n_bootstraps):
    """
    SEM of peak amplitude within windows (for P300).
    """
    results = []

    for ch_comp, (tmin_peak, tmax_peak) in windows.items():
        _, comp = ch_comp.split('_')

        if comp == 'P300':
            epochs_cropped = epochs.copy().crop(tmin=tmin_peak, tmax=tmax_peak)
            channel_data = epochs_cropped.pick_channels([ch_comp.split('_')[0]]).get_data()[:, 0, :]

            func = lambda x: np.max(x, axis=1)
            sem = bootstrap_sampling(channel_data, n_bootstraps, func)
            results.append(f"{ch_comp} Peak Amplitude SEM: {sem:.4f}")
                       
    return results

def peak_latency_func(sample, ch_comp, times):
    """
    Return per-trial peak latency array from a sample (n_trials, n_times).
    P*: argmax; N*: argmin.
    """
    _, comp = ch_comp.split('_')
    peak_indices = np.argmax(sample, axis=1) if 'P' in comp else np.argmin(sample, axis=1)
    return times[peak_indices]

def centroid_latency_func(sample, ch_comp, times):
    """
    Return per-trial centroid latency array (center of mass around the extremum).
    """
    _, comp = ch_comp.split('_')

    extremum_value = np.max(sample, axis=1)  # Use maximum voltage for P3 component
    numerator = np.sum(times[np.newaxis, :] * (sample - extremum_value[:, np.newaxis]), axis=1)
    denominator = np.sum(sample - extremum_value[:, np.newaxis], axis=1)
    denominator[denominator == 0] = 1e-12                  
    centroid_latency = numerator / denominator

    return centroid_latency

def bootstrap_diff_scores(hit_epochs, stim_epochs, n_bootstraps=10000):
    """
    Bootstrap mean difference (hit - stim) 
    """
    diff_scores = np.zeros(n_bootstraps)
    
    hit_data = hit_epochs.get_data()  # shape: (n_trials, n_channels, n_timepoints)
    stim_data = stim_epochs.get_data()
    
    hit_avgs = np.mean(hit_data, axis=(1, 2))   
    stim_avgs = np.mean(stim_data, axis=(1, 2))
    
    n_hit = len(hit_avgs)
    
    for i in range(n_bootstraps):
        sampled_hit_avg = np.mean(np.random.choice(hit_avgs, size=n_hit, replace=True))
        sampled_stim_avg = np.mean(np.random.choice(stim_avgs, size=n_hit, replace=True))
        diff_scores[i] = sampled_hit_avg - sampled_stim_avg
    
    mean_diff = np.mean(diff_scores)
    bootstrap_sem = np.std(diff_scores, ddof=1)
    
    return mean_diff, bootstrap_sem

# Parameters & windows
windows = {
    'Fz_N200': (0.200, 0.300),
    'Fz_P300': (0.300, 0.500),
    'Pz_P200': (0.150, 0.250),
    'Pz_P300': (0.300, 0.500)
}
n_bootstraps = 10000

channels = ['Fz', 'Pz']
# event_id 1:HIT 2:MISS 3:FA 4:STIM


# epochs_list: list of mne.Epochs objects, same tmin/sfreq expected
epochs_list = [mne.read_epochs(os.path.join(directory, file), preload=True).pick_channels(channels)[event_id]
               for file in os.listdir(directory) if file.endswith('-epo.fif')]

# Grand-average per channel
grand_average_Fz = compute_grand_average(epochs_list, 'Fz')
grand_average_Pz = compute_grand_average(epochs_list, 'Pz')

# Build EvokedArray for each channel
sfreq = epochs_list[0].info['sfreq']
tmin = epochs_list[0].tmin
info_Fz = mne.create_info(ch_names=['Fz'], sfreq=epochs_list[0].info['sfreq'], ch_types='eeg')
info_Pz = mne.create_info(ch_names=['Pz'], sfreq=epochs_list[0].info['sfreq'], ch_types='eeg')

evoked_Fz = mne.EvokedArray(grand_average_Fz, info_Fz, tmin=epochs_list[0].tmin)
evoked_Pz = mne.EvokedArray(grand_average_Pz, info_Pz, tmin=epochs_list[0].tmin)

# Extremes & time-window building
extreme_values_Fz = compute_extremes(evoked_Fz, {k: v for k, v in windows.items() if k.startswith('Fz')})
extreme_values_Pz = compute_extremes(evoked_Pz, {k: v for k, v in windows.items() if k.startswith('Pz')})
extreme_values = {**extreme_values_Fz, **extreme_values_Pz}

time_windows = {}
for component, (value, time) in extreme_values.items():
    if 'P300' in component:
        time_windows[component] = (time - 0.025, time + 0.025)  # +/-25 ms for P300
    else:
        time_windows[component] = (time - 0.015, time + 0.015)  # +/-15 ms for others

