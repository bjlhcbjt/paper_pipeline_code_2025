#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mne
import os.path as op
import numpy as np
from autoreject import compute_thresholds
from mne_icalabel.iclabel import iclabel_label_components 

# Load raw data
# Bad channels are already included in raw data
raw = mne.io.read_raw_fieldtrip(data_file, info=info, data_name='data')

# Set electrode montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# Filtering (band-pass) + notch filter
# filtering option: l_freq (high-pass), h_freq (low-pass)
raw_channel.filter(l_freq=l_freq, h_freq=h_freq, method='fir', picks='eeg')  # change the filtering parameter
raw.notch_filter(freqs=[48, 52], picks='eeg')

# Events and epoching
events = mne.find_events(raw, stim_channel='Marker',consecutive=True, 
                         min_duration=0, output='onset', shortest_event = 1)
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.5, tmax=1.5, 
                    baseline=None, preload=True, picks='eeg')

# Reject bad epochs using threshold per channel type
threshes = compute_thresholds(epochs, method='bayesian_optimization', 
                              random_state=None, picks=None, augment=False, 
                              verbose=True, n_jobs=1)

channel_types = {ch_name: channel_type for ch_name, channel_type in zip(epochs.ch_names, epochs.get_channel_types())}

reject = {}
for ch_name, threshold in threshes.items():
    ch_type = channel_types[ch_name]
    reject[ch_type] = max(reject[ch_type], threshold)

epochs_threshold_clean = epochs.copy().drop_bad(reject=reject)

# Get indices of rejected epochs from the drop_log
rejected_epochs = [i for i, drop in enumerate(epochs_threshold_clean.drop_log) if drop]

rejected_epochs_custom = {}
# Iterate over the drop_log to check if each epoch should be rejected
for i_epoch, drop_log_entry in enumerate(epochs_threshold_clean.drop_log):
    if drop_log_entry:
        # Determine if there are rejected channels other than specific frontal channels (e.g., Fp1/Fp2)
        non_fp_channels_rejected = [ch for ch in drop_log_entry if ch not in ['Fp1', 'Fp2']]
        # If there are rejected channels other than 'Fp1' and 'Fp2', record the epoch and its rejected channels
        if non_fp_channels_rejected:
            rejected_epochs_custom[i_epoch] = non_fp_channels_rejected

# Create new Epochs object containing the epochs that were not rejected by the custom logic
good_epochs_indices = list(set(range(len(epochs))) - set(rejected_epochs_custom.keys()))
epochs_clean = epochs[good_epochs_indices]

# Interpolate bad channels
epochs_clean = epochs_clean.interpolate_bads(reset_bads=False) 

# ICA
# if using 1Hz_ICA option denote this row
#filt_raw = epochs_clean.copy().filter(l_freq=1.0, h_freq=None)

filt_raw = epochs_clean.copy().set_eeg_reference("average")

picks_ica = mne.pick_types(epochs_clean.info, eeg=True, exclude='bads')

# create ICA Epochs
ica = mne.preprocessing.ICA(max_iter="auto", method="infomax", 
                            random_state=97, fit_params=dict(extended=True) )
ica.fit(filt_raw, picks=picks_ica)     # [~reject_log.bad_epochs]

# ICLabel classification and artifact rejection
proba = iclabel_label_components(filt_raw, ica)  # (n_components, 7)
brain_p = proba[:, 0]             
pred_k  = proba.argmax(axis=1)    
max_p   = proba.max(axis=1)       

THRESH_OTHER = 0.95   # stricter for "other"
THRESH_OTHERWISE = 0.90
BRAIN_CAP = 0.30      

exclude_idx = []
for i, (k, pmax, pbrain) in enumerate(zip(pred_k, max_p, brain_p)):
    if k == 0:
        continue
    if k == 6:  # "other"
        if pmax >= THRESH_OTHER and pbrain < BRAIN_CAP:
            exclude_idx.append(i)
    else:       # other artifact classes
        if pmax >= THRESH_OTHERWISE and pbrain < BRAIN_CAP:
            exclude_idx.append(i)

ica.exclude = np.array(exclude_idx, dtype=int)
epochs_ica = ica.apply(epochs_clean.copy(), exclude=ica.exclude)

# Reference: REST or CAR (choose one option)
# REST re-reference
from mne.datasets import fetch_fsaverage
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
trans = op.join(fs_dir, "bem", "fsaverage-trans.fif")  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

fwd = mne.make_forward_solution(
    raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None)
epochs_rest_ref = epochs_ica.copy().set_eeg_reference("REST", forward=fwd)

# CAR re-reference
epochs_CAR_ref = epochs_ica.copy().set_eeg_reference('average', projection=False)

# Baseline correction
epochs_bs = epochs_rest_ref.apply_baseline((-0.2, 0))  # baseline time window from -0.2 to 0
# epochs_bs = epochs_CAR_ref.apply_baseline((-0.2, 0))  # baseline time window from -0.2 to 0

