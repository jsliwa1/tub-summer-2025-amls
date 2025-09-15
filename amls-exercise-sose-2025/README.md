# ECG Time Series Classifier

## Authors
- Jakub Władysław Śliwa
- Filip Piotr Matysik

## Project Overview
This project was developed as part of the **Architecture of Machine Learning Systems** module (Summer Semester 2025) as an alternative exercise. The primary objective was to build a classifier for **univariate electrocardiogram (ECG) time series data**.  

Each team member independently implemented a classifier, enabling a **comparative analysis of two distinct model architectures**. My responsibility was to build a model inspired by Kachuee et al, 2018.

## Dataset
The dataset is related to the **PhysioNet/Cin Challenge 2017** dataset (Clifford et al., 2017).  

- **Training set:** 6,179 samples (with labels)  
- **Test set:** 2,649 samples (labels not provided for final evaluation)  

Each observation is a **time series of ECG signal measurements** sampled at **300 Hz** and labeled into one of four categories:  

0. Normal  
1. Atrial fibrillation (AF)  
2. Other (neither normal nor AF)  
3. Too noisy to process  

For further details, please refer to the report (PDF).


