# paper_pipeline_code_2025
This repository contains the research code used in our study examining the impact of different EEG preprocessing pipelines 
on N-back working memory task analyses through a comprehensive multiverse approach.

# Paper Information
Title: No Single Best Pipeline: Multiverse Analysis of EEG Preprocessing for N-back Working Memory Tasks

Authors: Huang, et al.

Institution: University of New South Wales, Black Dog Institute

Status: Under Review

# Code Description
- pipeline_code.py: Main pipeline for EEG preprocessing multiverse analysis
- compute_eeg_metrics_code.py: Functions for computing various EEG metrics across different preprocessing approaches

# Requirements
Python Environment: Python 3.7 or higher

Required package

- Numpy
- OS
- MNE (for EEG processing)
- AUTOREJECT

# Data Requirements
EEG data should be in standard formats supported by MNE-Python (e.g., .mat .fif, .edf, .bdf)

# Data Availability
The original EEG dataset used in this study is not included in this repository due to Privacy and ethical considerations

# Methodology
This code implements a multiverse analysis approach by:

Applying each pipeline to same datasets

Computing relevant EEG metrics for each pipeline

# License
This project is licensed under the BSD 2-Clause License - see the LICENSE file for details.
