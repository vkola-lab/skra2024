# Survival analysis on subchondral bone length for total knee replacement

This work is published in *Skeletal Radiology*.

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

```python

loc_data = "~/OAI_SBL_Analysis_Data/" --> path to data
loc_module = "~/OAI_Github_scripts/" --> path to analysis scripts
```

#### Scripts for Kaplan Meier Analysis

These scripts can be found under the 'Kaplan_Meier_Analysis' folder. By running these you will be able to generate Figures 2A and 2B from the paper. 

Notice: When saving figures from Kaplan-Meier analysis, be sure to change the path to your suited location. 

```python
plt.savefig('~/KMF_Curves/kmf_BML.pdf', format='pdf', bbox_inches = 'tight')
```
#### Scripts for Cox Proportional Hazards Analysis

These scripts can be found under the 'CPH_Analysis' folder. By running these you will be able to generate the other figures from the paper. 

Notice: When saving figures from CPH analysis, be sure to change the path to your suited location. 

```python
plt.savefig('~AUC_Curves/AUC_ALL_Knees_SBL_BML_figure_all_SBL.pdf', format='pdf', bbox_inches = 'tight')
```

## Stastistics
#### Time-Dependent Area Under the Curve (AUC)
#### Concordance Index
#### Brier Score
#### Log-Rank Test

#### Results
Each CPH Analysis script will produce a time-dependent AUC, Concordance Index, and Brier Score. Kaplan-Meier Analysis scripts will generate Kaplan Meier curves and the Log-Rank test associated with each curve. Generated figures are located in ~/AUC_Curves/ and ~/KMF_Curves/.
