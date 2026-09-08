	
# Multimodal Neurotechnology Platform

**Version:** 1.0.0  
**Author:** Diego Casasús Benítez

Portable Python research platform for synchronized EEG, EMG and IMU acquisition, PsychoPy-based silent- and imagined-speech paradigms, Lab Streaming Layer integration, XDF recording, visual-ERP calibration, and trial-level online decoding using exported PyTorch decoder bundles.

The platform was developed as part of the Master's Thesis:

> **Development of an online Python-based platform for synchronized electroencephalography, electromyography and inertial measurement unit in silent speech paradigms**

## Scope

The platform was developed and evaluated for fixed-vocabulary multimodal silent-speech research.

It integrates:

- graphical session preparation and configuration;
- PsychoPy experimental control;
- EEG, EMG and IMU stream discovery through LSL;
- synchronized experimental markers;
- LabRecorder/XDF recording;
- visual-ERP calibration and brainprint comparison;
- unimodal and multimodal decoder deployment;
- trial-level online inference and feedback;
- structured runtime and result logging.

The supplied decoder resources are specific to the acquisition modalities, preprocessing procedures, channel definitions, experimental timing and vocabulary used during this project. The platform should not be interpreted as an unrestricted or continuous speech-decoding system.

## Privacy

This repository intentionally contains no participant recordings, XDF files, processed participant datasets, online validation results, participant brainprints, or personal metadata.

Runtime-generated and participant-derived resources are excluded through `.gitignore`.

## Installation

### Windows

Install Miniforge3 and open **Miniforge Prompt** in the repository root.

Create and activate the project environment:

```bat
conda env create -f environment_windows.yml
conda activate tfm_unified