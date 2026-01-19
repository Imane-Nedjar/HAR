# BEMD–Attention–BiLSTM for Robust Human Activity Recognition

This repository provides the official implementation of a hybrid deep learning framework for **Human Activity Recognition (HAR)** based on **Bivariate Empirical Mode Decomposition (BEMD)** and an **Attention-enhanced Bidirectional Long Short-Term Memory (BiLSTM)** network.

The proposed approach aims to improve recognition performance on non-stationary and noisy inertial sensor signals by extracting pertinent multiscale features prior to temporal modeling.

---


## Repository Structure
├── Proposed_Model_(BEMD+BiLSTM+Attention).py   # Main framework 
   
├── Baseline_Configuration_II.py                # BEMD + BiLSTM 

├── Baseline_Configuration_I.py                 # Raw signal BiLSTM

├── requirements.txt                            # Dependency list

└── README.md                                 # Documentation

---

## Prerequisites

- Python **3.8** or higher  
- Access to the **UCI Human Activity Recognition (UCI-HAR) Dataset**

The dataset can be downloaded from:
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

---

## Installation

Install the required Python dependencies using:

```bash
pip install -r requirements.txt

