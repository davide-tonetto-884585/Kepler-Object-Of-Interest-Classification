# Kepler Objects of Interest (KOI) Analysis: Classification Project

## Description

This project analyzes data on Kepler Objects of Interest (KOIs) from the NASA Exoplanet Archive. The primary goal is to develop a statistical model to classify KOIs as either potential exoplanet **CANDIDATEs** or **FALSE POSITIVEs** based on their observed properties. The analysis utilizes the Q1-Q17 Data Release 25 (DR25) KOI table, chosen for its homogeneity and automated vetting process suitable for statistical analysis.

The project involves several stages:
1.  Data loading and preparation.
2.  Exploratory Data Analysis (EDA) through visualization.
3.  Dimensionality reduction using Principal Component Analysis (PCA).
4.  Predictive modeling using Generalized Linear Models (Logistic Regression).

## Data

The primary dataset used is:

* `data/q1_q17_dr25_koi.csv`: Contains the Kepler Objects of Interest cumulative table from Data Release 25 (Q1-Q17). This dataset was downloaded from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html).

## Project Structure
progetto_SIL/
├── data/
│   ├── q1_q17_dr25_koi.csv       # Primary DR25 KOI dataset
├── data_preparation.Rmd          # R Markdown script for data loading, cleaning, and initial summary
├── data_visualization.Rmd        # R Markdown script for exploratory data analysis visualizations and PCA analysis
├── model.Rmd                     # R Markdown script for models testing
└── model_pca.Rmd                 # R Markdown script for models testing using PCA

