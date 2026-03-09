# Deep Learning Based Real-Time Phase-Amplitude Correction for Phased Array Transducers in Transcranial Focused Ultrasound

## Overview

This repository provides the implementation of a deep learning model for predicting phase and amplitude corrections in transcranial focused ultrasound (tFUS). 
Complex human skull structure causes strong acoustic distortions that leads to phase aberration and amplitude reduction, which can significantly degrade focusing accuracy.
The proposed framework learns the mapping between skull geometry and acoustic propagation using neural networks trained on simulation data. 
By directly predicting phase and amplitude for each transducer element, the model enables fast and scalable inference compared to conventional time-reversal or wave simulation methods.

## Features

- **Element-wise prediction for phased-array transducers**

  : Independent neural networks are trained to estimate the optimal phase and amplitude for each transducer element in a large phased-array system.

- **Fast inference compared to simulation-based approaches**

  : The model replaces computationally expensive acoustic simulations with neural network inference, enabling significantly faster prediction of focusing parameters.

- **Simulation-driven training pipeline**

  : Training data are generated using acoustic simulations, allowing the model to learn the relationship between skull geometry and acoustic wave propagation.


## Repository Structure

- `Phase_base_training.py` : training script for phase base models  
- `Phase_fine_tuning.py` : fine-tuning script for phase models  
- `Amp_base_training.py` : training script for amplitude base models  
- `Amp_fine_tuning.py` : fine-tuning script for amplitude models  
- `Final_Parallel_Inference_Batch.py` : parallel inference for all transducer elements  
- `defining_fcns.py` : utility functions used across the pipeline

## Example Dataset

Due to the large size of the dataset, we provide a representative example case.
The example corresponds to:

- **Area:** Area1
- **Skull:** Skull11

This example dataset is sufficient to reproduce the inference pipeline.

## Requirements

- Python 3.9+
- torch 2.5.1+cu121
- NumPy 2.2.6
- NVIDIA GPU with CUDA support (recommended for efficient inference)

## Download Models and Data

Pretrained models and example inference data can be downloaded from:

**Google Drive**

https://drive.google.com/drive/folders/1Hw5GOtM4PqualxSg2lXmMF2VakDnGVZf?usp=sharing

After downloading, place the files into the repository as follows:

- `repo_example_data.pt` → `data/`
  
- The 1024 pretrained amplitude models in `Amp_model/Base/` → `checkpoints/Amp_model/Base/`
  
- The fine-tuned amplitude models in `Amp_model/Fine_tuned/` → `checkpoints/Amp_model/Fine_tuned/`
  
- The 1024 pretrained phase models in `Phase_model/Base/` → `checkpoints/Phase_model/Base/`
  
- The fine-tuned phase models in `Phase_model/Fine_tuned/` → `checkpoints/Phase_model/Fine_tuned/`

## Running Inference

1. Download the pretrained models and example data from the Google Drive link provided above.

2. Place the downloaded files in the repository following the structure described in **Download Models and Data**.

3. Run the inference script `Final_Parallel_Inference_Batch.py` to get the final predicted phase and amplitude for target point.

## Training (Optional)

The repository also provides scripts for training the models from scratch.

1. Train the base models using:
`Phase_base_training.py` and `Amp_base_training.py`. 

2. Fine-tune the model for target skull using:
`Phase_fine_tuning.py` and `Amp_fine_tuning.py`

Note that pretrained and fine-tuned models used in the paper are already provided in the Google Drive link.

## Loss Function

For amplitude model training, the model uses a **Huber loss** as a loss function, while the model uses a customized loss function incorporating **KL loss** and **Cosine-based Circular loss** for phase model training.


## Evaluation

- **Phase Prediction Performance**
  : evaluated using Circular Mean Absolute Error (CMAE), Circular Huber Loss, and Cosine-based Circular Loss.

- **Amplitude Prediction Performance**
  : evaluated using Relative Energy Error (REE), Mean Absolute Error (MAE), and Huber Loss.

- **Ultrasound Focusing Accuracy**
  : evaluated using Peak Location Error (PLE), Mean Surface Distance (MSD), Relative Peak Pressure (RPP).

