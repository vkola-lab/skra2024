# Subchondral bone length versus bone marrow lesions: Survival models for total knee replacement

This work has been submitted to  _Arthritis Care & Research_ for peer review.

## Prerequisites

The tool was developed based on the following dependencies:

1. NumPy
5. Scipy
6. Pandas
7. Lifelines
8. Statistics
9. Matplotlib
10. Sksurv
11. Sklearn
12. Jupyter Notebooks


#### Data files

All data files for this project are stored within the folder 'OAI_SBL_Analysis_Data'. Within this folder all information for patients regarding demographics, TKR, BML size, and SBL can be found. 

Notice: You must update the path to this folder within each script before conducting Cox Proportional Hazards or Kaplan-Meier Analysis. 

#### Scripts for Kaplan Meier Analysis

These scripts can be found under the 'Kaplan_Meier_Analysis' folder. By running these you will be able to generate Figures 2A and 2B from the paper. 

#### Scripts for Cox Proportional Hazards Analysis

These scripts can be found under the 'CPH_Analysis' folder. By running these you will be able to generate Figures 3A through 6 from the paper. 


## Stastistics
#### Time-Dependent Area Under the Curve (AUC)
#### Concordance Index
#### Brier Score
#### Log-Rank Test

#### Results
Each CPH Analysis script will produce the time-dependent AUC, Concordance Index, and Brier Score. Kaplan-Meier Analysis scripts will generate Kaplan Meier curves and the Log-Rank test associated with each curve. Generated figures are located in ~/AUC_Curves/ and ~/KMF_Curves/.
