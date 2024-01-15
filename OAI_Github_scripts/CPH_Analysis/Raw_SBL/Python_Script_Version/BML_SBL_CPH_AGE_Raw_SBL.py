#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

sys.path.insert(0, "/home/tsurendr/OAI_Skeletal_Radiology_Code_Working")
sys.path
# get_ipython().run_line_magic('pip', 'install -U pandas==1.5.3')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import lifelines
import math
import sys
from statistics import stdev
from lifelines.statistics import logrank_test, multivariate_logrank_test,pairwise_logrank_test

# Set the path of SBL and merge1 files and read the files
loc_data = "/data_1/OAI_SBL_Analysis_Data/"
loc_module = "/home/tsurendr/OAI_Github_scripts"
loc_data_SBL = loc_data + "SBL_0904.csv"
loc_merge1 = loc_data + "merge1_KL.csv"
raw_data_SBL = pd.read_csv(loc_data_SBL)
merge1 = pd.read_csv(loc_merge1)
# load custom module 
sys.path.append(loc_module)


# In[ ]:


sbl_col_names = ["F" + str(i) for i in range(200)] + [
    "T" + str(i) for i in range(200)
]  # femur: F0~ F199 , tibia: T0 ~ T199
sbl_col_names_femur = ["F" + str(i) for i in range(200)]
sbl_col_names_tibia = ["T" + str(i) for i in range(200)]


# In[ ]:


##################################
# In this model we are using RAW SBL VALUES#
##################################
# raw_sbl_values = raw_data_SBL.loc[:, sbl_col_names].values
# # print(raw_data_SBL)
# sbl_values = np.empty_like(raw_sbl_values)  # for saving normalized SBL
# for row in range(raw_sbl_values.shape[0]):
#     sbl_values[row, :] = (
#         raw_sbl_values[row, :] / raw_sbl_values[row, :].mean()
#     )  # normalize by the averaged val. of SBL
# df_normalized_SBL = pd.DataFrame(sbl_values, columns=sbl_col_names)
# # get mean of Kellgren Lawrence (KL) grade 0
# sbl_KL_0_mean = df_normalized_SBL.loc[
#     (merge1["KL_Grade"] == 0) , sbl_col_names
# ].values.mean(0)

# print(f"shape of sbl_KL_0_mean: {sbl_KL_0_mean.shape}")
# baseline = sbl_KL_0_mean
# sbl_difference = df_normalized_SBL.loc[:, sbl_col_names].sub(baseline, axis=1)
# # absolute value of sbl difference.
# sbl_difference_absolute = sbl_difference.abs()
# sbl_difference_absolute.name = "normalized_sbl"
df_normalized_SBL_both = pd.DataFrame(raw_data_SBL)
df_normalized_SBL_both = df_normalized_SBL_both.add_suffix('_merged')
# df_normalized_SBL_both['SBL_difference_merged'] = df_normalized_SBL_both.sum(axis=1)

df_normalized_SBL_both.drop(columns=df_normalized_SBL_both.columns[0], axis=1,  inplace=True)
df_normalized_SBL_both.drop(columns=df_normalized_SBL_both.columns[-1], axis=1,  inplace=True)
df_normalized_SBL_both.drop(columns=df_normalized_SBL_both.columns[-1], axis=1,  inplace=True)
print(df_normalized_SBL_both)


# In[ ]:


# Combining 3 Variables: df_normalized_SBL_both, df_normalized_SBL_femur, df_normalized_SBL_tibia
data_SBL = pd.merge(
    raw_data_SBL, df_normalized_SBL_both, right_index=True, left_index=True
)  # merge df_normalized_SBL_both


# In[ ]:


# Splitting data by knee side(right/left) and formatting columns appropriately 

print("total number of baseline knees", len(data_SBL))
data_SBL["id"] = data_SBL["id"].astype(str)
data_BioMarkers = pd.read_csv(loc_data + "Biomarker_data.csv")
data_SBL = data_SBL.drop(["Unnamed: 0"], axis=1)
data_BioMarkers = data_BioMarkers.drop(["Unnamed: 0"], axis=1)
side_SBL_temp = data_SBL.groupby("SIDE")
side_1_SBL_Right = side_SBL_temp.get_group(1)
side_2_SBL_Left = side_SBL_temp.get_group(2)


# In[ ]:


print("total number of right knees", len(side_1_SBL_Right))
print("total number of left knees", len(side_2_SBL_Left))


# In[ ]:


# settings
NUM_YEARS = 11.0  
encoding = "utf-8"
# read and preprocessing
raw_df = pd.read_sas(loc_data + "outcomes99.sas7bdat")
# must censor data per knee
print("Before data drop mri data", len(raw_df))
df = raw_df.dropna(axis=0, subset=["V99RNTCNT"])
print("complete mri data", len(df))
print(f"number of drop: {len(raw_df)-len(df)}")
df = df.copy()
df.loc[:, "id"] = df["id"].apply(lambda x: str(x, encoding))
merge1 = merge1.dropna(axis=0, subset=["P02SEX"])
merge1 = merge1.dropna(axis=0, subset=["V00AGE"])
merge1 = merge1.dropna(axis=0, subset=["V00XRJSM"])
merge1 = merge1.dropna(axis=0, subset=["V00XRJSL"])
merge1["id"] = merge1["id"].astype(str)
merge1_temp = merge1.groupby("SIDE")
merge1_right = merge1_temp.get_group(1)
merge1_left = merge1_temp.get_group(2)
df_8_years = df[df["V99RNTCNT"] <= NUM_YEARS].copy()  
print("oai Data: ", len(df_8_years))


# In[ ]:


# KL Grade information preprocessing for right knees
data_KL_grade_right = pd.read_csv(loc_data + "rightFilteredklMEAS.csv")
data_KL_grade_right = data_KL_grade_right.drop(["Unnamed: 0"], axis=1)
data_KL_grade_right = data_KL_grade_right.dropna(axis=0, subset=["V00XRKLR"])
data_KL_grade_right["id"] = data_KL_grade_right["id"].astype(str)


# In[ ]:


# KL Grade information preprocessing for left knees
data_KL_grade_left = pd.read_csv(loc_data + "leftFilteredklMEAS.csv")
data_KL_grade_left = data_KL_grade_left.drop(["Unnamed: 0"], axis=1)
data_KL_grade_left = data_KL_grade_left.dropna(axis=0, subset=["V00XRKLL"])
data_KL_grade_left["id"] = data_KL_grade_left["id"].astype(str)


# In[ ]:


# BML information preprocessing for right knees. 
# right side
data_BML_right = pd.read_csv(loc_data + "rightFilteredbmlMoaks.csv")
data_BML_right["id"] = data_BML_right["id"].astype(str)
data_BML_right = data_BML_right.drop(["Unnamed: 0"], axis=1)
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMSFMA',
'V00MBMSFLA',
'V00MBMSFMC',
'V00MBMSFLC',
'V00MBMSFMP',
'V00MBMSFLP',
'V00MBMSSS',
'V00MBMSTMA',
'V00MBMSTLA',
'V00MBMSTMC',
'V00MBMSTLC',
'V00MBMSTMP',
'V00MBMSTLP'])

# For verification after data processing
print("bml right Data: ", len(data_BML_right))


# In[ ]:


# Race Data, dropping patients with missing data
race_data = pd.read_csv('/data_1/OAI_Backup/MeasInventory.csv')
race_data = race_data.dropna(axis=0, subset=['P02RACE'])
race_data = race_data.drop(columns=['V00AGE'])
race_data = race_data.drop(columns=['P02SEX'])
race_data = race_data.drop(columns=['V00XRKLR'])
race_data = race_data.drop(columns=['V00XRKLL'])
race_data['id'] = race_data['id'].astype(str)


# In[ ]:


# Merging all right knee info, including demographics, TKR, KL Grade, BML, etc...
oai_bml_merge_right = pd.merge(df_8_years, data_BML_right, how="inner", on=["id"])
oai_bml_SBL_KL_merge_right_pre = pd.merge(
    oai_bml_merge_right, side_1_SBL_Right, how="inner", on=["id"]
)
print(len(oai_bml_SBL_KL_merge_right_pre))

oai_bml_SBL_KL_merge_right_age_pre_1 = pd.merge(
    oai_bml_SBL_KL_merge_right_pre, data_KL_grade_right, how="inner", on=["id"]
)
oai_bml_SBL_KL_merge_right_pre = pd.merge(
    oai_bml_SBL_KL_merge_right_age_pre_1, merge1_right, how="inner", on=["id"]
)


oai_bml_SBL_KL_merge_right = pd.merge(
    oai_bml_SBL_KL_merge_right_pre, race_data, how="inner", on=["id"]
)
print(len(oai_bml_SBL_KL_merge_right))

oai_bml_SBL_KL_merge_right.drop_duplicates(subset=["id"], inplace=True, keep="last")
oai_bml_SBL_KL_merge_right.reset_index(drop=True, inplace=True)
print(len(oai_bml_SBL_KL_merge_right))


# In[ ]:


if 'V00XRJSM' in oai_bml_SBL_KL_merge_right:
    print(True)
else:
    print(False)


# In[ ]:


if 'V00XRJSL' in oai_bml_SBL_KL_merge_right:
    print(True)
else:
    print(False)


# In[ ]:


# BML information preprocessing for left knees. 
# left side
data_BML_left = pd.read_csv(loc_data + "leftFilteredbmlMoaks.csv")
data_BML_left["id"] = data_BML_left["id"].astype(str)

data_BML_left = data_BML_left.drop(["Unnamed: 0"], axis=1)
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMSFMA',
'V00MBMSFLA',
'V00MBMSFMC',
'V00MBMSFLC',
'V00MBMSFMP',
'V00MBMSFLP',
'V00MBMSSS',
'V00MBMSTMA',
'V00MBMSTLA',
'V00MBMSTMC',
'V00MBMSTLC',
'V00MBMSTMP',
'V00MBMSTLP'])

# For verification after data processing
print("bml left Data: ", len(data_BML_left))


# In[ ]:


# Merging all left knee info, including demographics, TKR, KL Grade, BML, etc...
oai_bml_merge_left = pd.merge(df_8_years, data_BML_left, how="inner", on=["id"])
oai_bml_SBL_KL_merge_left_pre = pd.merge(
    oai_bml_merge_left, side_2_SBL_Left, how="inner", on=["id"]
)
oai_bml_SBL_KL_merge_left_age_pre_1 = pd.merge(
    oai_bml_SBL_KL_merge_left_pre, data_KL_grade_left, how="inner", on=["id"]
) 

oai_bml_SBL_KL_merge_left_pre = pd.merge(
    oai_bml_SBL_KL_merge_left_age_pre_1, merge1_left, how="inner", on=["id"]
)

oai_bml_SBL_KL_merge_left = pd.merge(
    oai_bml_SBL_KL_merge_left_pre, race_data, how="inner", on=["id"]
)
oai_bml_SBL_KL_merge_left.drop_duplicates(subset=["id"], inplace=True, keep="last")
oai_bml_SBL_KL_merge_left.reset_index(drop=True, inplace=True)
print(len(oai_bml_SBL_KL_merge_left))


# In[ ]:


if 'V00XRJSM' in oai_bml_SBL_KL_merge_left:
    print(True)
else:
    print(False)


# In[ ]:


if 'V00XRJSL' in oai_bml_SBL_KL_merge_left:
    print(True)
else:
    print(False)


# In[ ]:


print(oai_bml_SBL_KL_merge_right['V00XRJSM'].isnull().sum())
print(oai_bml_SBL_KL_merge_right['V00XRJSL'].isnull().sum())


# In[ ]:


print(oai_bml_SBL_KL_merge_left['V00XRJSM'].isnull().sum())
print(oai_bml_SBL_KL_merge_left['V00XRJSL'].isnull().sum())


# In[ ]:


# need 3 groups representing merged femur and tibia. 

femur_column_list = ['V00MBMSFMA',
'V00MBMSFLA',
'V00MBMSFMC',
'V00MBMSFLC',
'V00MBMSFMP',
'V00MBMSFLP']

tibia_column_list = ['V00MBMSSS',
'V00MBMSTMA',
'V00MBMSTLA',
'V00MBMSTMC',
'V00MBMSTLC',
'V00MBMSTMP',
'V00MBMSTLP']

merged_column_list = ['V00MBMSFMA',
'V00MBMSFLA',
'V00MBMSFMC',
'V00MBMSFLC',
'V00MBMSFMP',
'V00MBMSFLP',
'V00MBMSSS',
'V00MBMSTMA',
'V00MBMSTLA',
'V00MBMSTMC',
'V00MBMSTLC',
'V00MBMSTMP',
'V00MBMSTLP']


# In[ ]:


# Determining the largest BML in each knee based on if it is a Merged, Femur, or Tibia model
# For both left and right knees

# right side
right_knee_tkr = oai_bml_SBL_KL_merge_right[
    (oai_bml_SBL_KL_merge_right["V99RNTCNT"] <= NUM_YEARS)
    & (oai_bml_SBL_KL_merge_right["V99ERKDAYS"].isnull() == False)
]
print("total knees on right side: ", len(oai_bml_SBL_KL_merge_right))
print("censored right knees: ", len(oai_bml_SBL_KL_merge_right) - len(right_knee_tkr))
print("proper right knee tkr: ", len(right_knee_tkr))

oai_bml_SBL_KL_merge_right["right_tkr"] = np.where(
    oai_bml_SBL_KL_merge_right["id"].isin(right_knee_tkr["id"]) == True, 1, 0
)
oai_bml_SBL_KL_merge_right["bml_total_merged"] = oai_bml_SBL_KL_merge_right[merged_column_list].max(axis=1)
oai_bml_SBL_KL_merge_right["bml_total_femur"] = oai_bml_SBL_KL_merge_right[femur_column_list].max(axis=1)
oai_bml_SBL_KL_merge_right["bml_total_tibia"] = oai_bml_SBL_KL_merge_right[tibia_column_list].max(axis=1)
print(oai_bml_SBL_KL_merge_right["bml_total_merged"].unique())

# left side
left_knee_tkr = oai_bml_SBL_KL_merge_left[
    (oai_bml_SBL_KL_merge_left["V99RNTCNT"] <= NUM_YEARS)
    & (oai_bml_SBL_KL_merge_left["V99ELKDAYS"].isnull() == False)
]

print("total knees on left side: ", len(oai_bml_SBL_KL_merge_left))
print("censored left knees: ", len(oai_bml_SBL_KL_merge_left) - len(left_knee_tkr))
print("proper left knee tkr: ", len(left_knee_tkr))


oai_bml_SBL_KL_merge_left["right_tkr"] = np.where(
    oai_bml_SBL_KL_merge_left["id"].isin(left_knee_tkr["id"]) == True, 1, 0
)  

oai_bml_SBL_KL_merge_left["bml_total_merged"] = oai_bml_SBL_KL_merge_left[merged_column_list].max(axis = 1)
oai_bml_SBL_KL_merge_left["bml_total_femur"] = oai_bml_SBL_KL_merge_left[femur_column_list].max(axis = 1)
oai_bml_SBL_KL_merge_left["bml_total_tibia"] = oai_bml_SBL_KL_merge_left[tibia_column_list].max(axis = 1)
print(oai_bml_SBL_KL_merge_left["bml_total_merged"].unique())

print(oai_bml_SBL_KL_merge_right)


# In[ ]:


# Renaming right knee columns for ease of use
oai_bml_SBL_KL_merge_right = oai_bml_SBL_KL_merge_right.rename(columns = {"V00AGE":'AGE', "V00XRKLR":'KL_grade',"P01BMI":'BMI',"P02RACE":'RACE'})

print('RACE' in oai_bml_SBL_KL_merge_right)
print("KL_grade" in oai_bml_SBL_KL_merge_right.columns.tolist() )


# In[ ]:


# Renaming left knee columns for ease of use
oai_bml_SBL_KL_merge_left = oai_bml_SBL_KL_merge_left.rename(columns = {"V00AGE":'AGE',"V00XRKLL":'KL_grade',"P01BMI":'BMI',"P02RACE":'RACE'})
# oai_bml_SBL_KL_merge_left = oai_bml_SBL_KL_merge_left.rename(columns = {})

print("KL_grade" in oai_bml_SBL_KL_merge_left.columns.tolist() )


# In[ ]:


# Determining which patiets had a TKR and if not, what their most recent time of follow-up was.

from time_adder import add_time

oai_SBL_KL_BML_right = add_time(oai_bml_SBL_KL_merge_right, "right")
oai_SBL_KL_BML_left = add_time(oai_bml_SBL_KL_merge_left, "left")


# In[ ]:


# Selecting columns from left and right knee info, to merge into 1 table for each SBL and BML model
# right side
oai_right_temp_SBL_Merged_right = pd.concat( [oai_SBL_KL_BML_right.loc[:, 'F0_merged':'T199_merged'], oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX",'AGE','KL_grade','BMI', 'RACE','bml_total_merged','bml_total_femur','bml_total_tibia', "V00XRJSM",'V00XRJSL']]], axis = 1)

oai_right_temp_SBL_Femur_right = pd.concat( [oai_SBL_KL_BML_right.loc[:, 'F0_merged':'F199_merged'], oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX",'AGE','bml_total_femur','KL_Grade']]], axis = 1)

oai_right_temp_SBL_Tibia_right = pd.concat( [oai_SBL_KL_BML_right.loc[:, 'T0_merged':'T199_merged'], oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX",'AGE','bml_total_tibia','KL_Grade']]], axis = 1)


oai_right_temp_BML_Merged_right = oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX", "bml_total_merged",'AGE','KL_Grade']
]
oai_right_temp_BML_Femur_right = oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX", "bml_total_femur",'AGE','KL_Grade']
]
oai_right_temp_BML_Tibia_right = oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX", "bml_total_tibia",'AGE','KL_Grade']
]

oai_right_temp_JSN_Merged_right = oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX", "V00XRJSM",'V00XRJSL', 'KL_Grade']
]


# left side
oai_right_temp_SBL_Merged_left = pd.concat( [oai_SBL_KL_BML_left.loc[:, 'F0_merged':'T199_merged'], oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX",'AGE','KL_grade','BMI', 'RACE','bml_total_merged','bml_total_femur','bml_total_tibia', "V00XRJSM",'V00XRJSL']]], axis = 1)

oai_right_temp_SBL_Femur_left = pd.concat( [oai_SBL_KL_BML_left.loc[:, 'F0_merged':'F199_merged'], oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX",'AGE','bml_total_femur','KL_Grade']]], axis = 1)

oai_right_temp_SBL_Tibia_left = pd.concat( [oai_SBL_KL_BML_left.loc[:, 'T0_merged':'T199_merged'], oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX",'AGE','bml_total_tibia','KL_Grade']]], axis = 1)


oai_right_temp_BML_Merged_left = oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX", "bml_total_merged",'AGE','KL_Grade']
]
oai_right_temp_BML_Femur_left = oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX", "bml_total_femur",'AGE','KL_Grade']
]
oai_right_temp_BML_Tibia_left = oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX", "bml_total_tibia",'AGE','KL_Grade']
]

oai_right_temp_JSN_Merged_left = oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX", "V00XRJSM",'V00XRJSL', 'KL_Grade']
]






# In[ ]:


# merging information for various SBL and BML models based on left and right knee information that was selected previously
oai_right_temp_SBL_Merged_all = pd.concat(
    [oai_right_temp_SBL_Merged_right, oai_right_temp_SBL_Merged_left],
    ignore_index=True,
)
oai_right_temp_SBL_Femur_all = pd.concat(
    [oai_right_temp_SBL_Femur_right, oai_right_temp_SBL_Femur_left],
    ignore_index=True,
)
oai_right_temp_SBL_Tibia_all = pd.concat(
    [oai_right_temp_SBL_Tibia_right, oai_right_temp_SBL_Tibia_left],
    ignore_index=True,
)
oai_right_temp_BML_Merged_all = pd.concat(
    [oai_right_temp_BML_Merged_right, oai_right_temp_BML_Merged_left],
    ignore_index=True,
)

oai_right_temp_BML_Merged_all = pd.concat(
    [oai_right_temp_BML_Merged_right, oai_right_temp_BML_Merged_left],
    ignore_index=True,
)
oai_right_temp_BML_Femur_all = pd.concat(
    [oai_right_temp_BML_Femur_right, oai_right_temp_BML_Femur_left],
    ignore_index=True,
)
oai_right_temp_BML_Tibia_all = pd.concat(
    [oai_right_temp_BML_Tibia_right, oai_right_temp_BML_Tibia_left],
    ignore_index=True,
)

oai_right_temp_JSN_Merged_all = pd.concat(
    [oai_right_temp_JSN_Merged_right, oai_right_temp_JSN_Merged_left],
    ignore_index=True,
)

# Dropping unknown information from Race column
oai_right_temp_SBL_Merged_all.drop(oai_right_temp_SBL_Merged_all[(oai_right_temp_SBL_Merged_all['RACE'] == '.D: Don t Know/Unknown/Uncertain') ].index, inplace=True)


# In[ ]:


# dropping the knee with the smaller bml from patients who have 2 knees in the table in order to avoid confounding variables. 
oai_right_temp_SBL_Merged_all = oai_right_temp_SBL_Merged_all.sort_values('KL_grade').drop_duplicates('id', keep='last').sort_index()
oai_right_temp_SBL_Femur_all = oai_right_temp_SBL_Femur_all.sort_values('KL_Grade').drop_duplicates('id', keep='last').sort_index()
oai_right_temp_SBL_Tibia_all = oai_right_temp_SBL_Tibia_all.sort_values('KL_Grade').drop_duplicates('id', keep='last').sort_index()
oai_right_temp_BML_Merged_all = oai_right_temp_BML_Merged_all.sort_values('KL_Grade').drop_duplicates('id', keep='last').sort_index()
oai_right_temp_BML_Femur_all = oai_right_temp_BML_Femur_all.sort_values('KL_Grade').drop_duplicates('id', keep='last').sort_index()
oai_right_temp_BML_Tibia_all = oai_right_temp_BML_Tibia_all.sort_values('KL_Grade').drop_duplicates('id', keep='last').sort_index()
oai_right_temp_JSN_Merged_all = oai_right_temp_JSN_Merged_all.sort_values('KL_Grade').drop_duplicates('id', keep='last').sort_index()



#Checking to make sure each patient has no more than 1 knee in the table
print(len(list(set([x for i,x in enumerate(oai_right_temp_SBL_Merged_all['id'].tolist()) if oai_right_temp_SBL_Merged_all['id'].tolist().count(x) > 1]))))



# In[ ]:


groups_merged = oai_right_temp_SBL_Merged_all.groupby("P02SEX")
males_merged = groups_merged.get_group(1)
females_merged = groups_merged.get_group(2)
print("total males", len(males_merged))
# check the gender population; male:1, female:2
print('total females', len(females_merged))


# In[ ]:


# grouping by sex
groups_merged = oai_right_temp_SBL_Merged_all.groupby("P02SEX")
males_merged = groups_merged.get_group(1)
females_merged = groups_merged.get_group(2)
males_merged = males_merged.drop(columns=["P02SEX"])
females_merged = females_merged.drop(columns=["P02SEX"])

groups_femur = oai_right_temp_SBL_Femur_all.groupby("P02SEX")
males_femur = groups_femur.get_group(1)
females_femur = groups_femur.get_group(2)
males_femur = males_femur.drop(columns=["P02SEX"])
females_femur = females_femur.drop(columns=["P02SEX"])

groups_tibia = oai_right_temp_SBL_Tibia_all.groupby("P02SEX")
males_tibia = groups_tibia.get_group(1)
females_tibia = groups_tibia.get_group(2)
males_tibia = males_tibia.drop(columns=["P02SEX"])
females_tibia = females_tibia.drop(columns=["P02SEX"])

groups_BML_merged = oai_right_temp_BML_Merged_all.groupby("P02SEX")
males_BML_merged = groups_BML_merged.get_group(1)
females_BML_merged = groups_BML_merged.get_group(2)


groups_BML_femur = oai_right_temp_BML_Femur_all.groupby("P02SEX")
males_BML_femur = groups_BML_femur.get_group(1)
females_BML_femur = groups_BML_femur.get_group(2)



groups_BML_tibia = oai_right_temp_BML_Tibia_all.groupby("P02SEX")
males_BML_tibia = groups_BML_tibia.get_group(1)
females_BML_tibia = groups_BML_tibia.get_group(2)


males_BML_merged = males_BML_merged.drop(columns=["P02SEX"])
females_BML_merged = females_BML_merged.drop(columns=["P02SEX"])

males_BML_femur = males_BML_femur.drop(columns=["P02SEX"])
females_BML_femur = females_BML_femur.drop(columns=["P02SEX"])

males_BML_tibia = males_BML_tibia.drop(columns=["P02SEX"])
females_BML_tibia = females_BML_tibia.drop(columns=["P02SEX"])


# Dropping irrelevent information from tables before input to Cox Models
oai_right_temp_SBL_Merged_all = oai_right_temp_SBL_Merged_all.drop(columns=["P02SEX",'id','KL_grade'])
oai_right_temp_SBL_Femur_all = oai_right_temp_SBL_Femur_all.drop(columns=["P02SEX",'id','KL_Grade'])
oai_right_temp_SBL_Tibia_all = oai_right_temp_SBL_Tibia_all.drop(columns=["P02SEX",'id','KL_Grade'])
oai_right_temp_BML_Merged_all = oai_right_temp_BML_Merged_all.drop(columns=["P02SEX",'id','KL_Grade'])
oai_right_temp_BML_Femur_all = oai_right_temp_BML_Femur_all.drop(columns=["P02SEX",'id','KL_Grade'])
oai_right_temp_BML_Tibia_all = oai_right_temp_BML_Tibia_all.drop(columns=["P02SEX",'id','KL_Grade'])
oai_right_temp_JSN_Merged_all = oai_right_temp_JSN_Merged_all.drop(columns=["P02SEX",'id','KL_Grade'])


# In[ ]:


#Splitting populations based on age

less_sixty_merged = oai_right_temp_SBL_Merged_all[(oai_right_temp_SBL_Merged_all['AGE'] < 60)]
sixty_seventy_merged = oai_right_temp_SBL_Merged_all[(oai_right_temp_SBL_Merged_all['AGE'] <=  70) & (oai_right_temp_SBL_Merged_all['AGE'] >=  60)]
greater_seventy_merged = oai_right_temp_SBL_Merged_all[(oai_right_temp_SBL_Merged_all['AGE'] > 70)]


less_sixty_femur = oai_right_temp_SBL_Femur_all[(oai_right_temp_SBL_Femur_all['AGE'] < 60)]
sixty_seventy_femur = oai_right_temp_SBL_Femur_all[(oai_right_temp_SBL_Femur_all['AGE'] <=  70) & (oai_right_temp_SBL_Femur_all['AGE'] >=  60)]
greater_seventy_femur = oai_right_temp_SBL_Femur_all[(oai_right_temp_SBL_Femur_all['AGE'] > 70)]

less_sixty_tibia = oai_right_temp_SBL_Tibia_all[(oai_right_temp_SBL_Tibia_all['AGE'] < 60)]
sixty_seventy_tibia = oai_right_temp_SBL_Tibia_all[(oai_right_temp_SBL_Tibia_all['AGE'] <=  70) & (oai_right_temp_SBL_Tibia_all['AGE'] >=  60)]
greater_seventy_tibia = oai_right_temp_SBL_Tibia_all[(oai_right_temp_SBL_Tibia_all['AGE'] > 70)]



oai_right_temp_SBL_Merged_all.loc[(oai_right_temp_SBL_Merged_all['AGE'] < 60), 'AGE_Group'] = 0
oai_right_temp_SBL_Merged_all.loc[(oai_right_temp_SBL_Merged_all['AGE'] <=  70) & (oai_right_temp_SBL_Merged_all['AGE'] >=  60), 'AGE_Group'] = 1
oai_right_temp_SBL_Merged_all.loc[(oai_right_temp_SBL_Merged_all['AGE'] > 70), 'AGE_Group'] = 2



# In[ ]:


# Determining average age of patients with and without a TKR

with_TKR = oai_right_temp_SBL_Merged_all[(oai_right_temp_SBL_Merged_all['right_tkr'] == 1)]
without_TKR = oai_right_temp_SBL_Merged_all[(oai_right_temp_SBL_Merged_all['right_tkr'] == 0)]

print((without_TKR.AGE.mean()))
print((with_TKR.AGE.mean()))


# In[ ]:


# Selecting SBL, BML, AGE, and TKR from each table before information is input into the CPH models



less_sixty_sbl_merged = pd.concat( [less_sixty_merged.loc[:, 'F0_merged':'T199_merged'], less_sixty_merged[
    ["time", "right_tkr"]]], axis = 1)

less_sixty_sbl_femur = pd.concat( [less_sixty_femur.loc[:, 'F0_merged':'F199_merged'], less_sixty_femur[
    ["time", "right_tkr"]]], axis = 1)

less_sixty_sbl_tibia = pd.concat( [less_sixty_tibia.loc[:, 'T0_merged':'T199_merged'], less_sixty_tibia[
    ["time", "right_tkr"]]], axis = 1)
less_sixty_bml_merged = less_sixty_merged[["time", "right_tkr", "bml_total_merged"]]
less_sixty_bml_femur = less_sixty_merged[["time", "right_tkr", "bml_total_femur"]]
less_sixty_bml_tibia = less_sixty_merged[["time", "right_tkr", "bml_total_tibia"]]

sixty_seventy_sbl_merged = pd.concat( [sixty_seventy_merged.loc[:, 'F0_merged':'T199_merged'], sixty_seventy_merged[
    ["time", "right_tkr"]]], axis = 1)

sixty_seventy_sbl_femur = pd.concat( [sixty_seventy_femur.loc[:, 'F0_merged':'F199_merged'], sixty_seventy_femur[
    ["time", "right_tkr"]]], axis = 1)

sixty_seventy_sbl_tibia = pd.concat( [sixty_seventy_tibia.loc[:, 'T0_merged':'T199_merged'], sixty_seventy_tibia[
    ["time", "right_tkr"]]], axis = 1)


sixty_seventy_bml_merged = sixty_seventy_merged[["time", "right_tkr", "bml_total_merged"]]
sixty_seventy_bml_femur = sixty_seventy_merged[["time", "right_tkr", "bml_total_femur"]]
sixty_seventy_bml_tibia = sixty_seventy_merged[["time", "right_tkr", "bml_total_tibia"]]

greater_seventy_sbl_merged = pd.concat( [greater_seventy_merged.loc[:, 'F0_merged':'T199_merged'], greater_seventy_merged[
    ["time", "right_tkr"]]], axis = 1)

greater_seventy_sbl_femur = pd.concat( [greater_seventy_femur.loc[:, 'F0_merged':'F199_merged'], greater_seventy_femur[
    ["time", "right_tkr"]]], axis = 1)

greater_seventy_sbl_tibia = pd.concat( [greater_seventy_tibia.loc[:, 'T0_merged':'T199_merged'], greater_seventy_tibia[
    ["time", "right_tkr"]]], axis = 1)


greater_seventy_bml_merged = greater_seventy_merged[["time", "right_tkr", "bml_total_merged"]]
greater_seventy_bml_femur = greater_seventy_merged[["time", "right_tkr", "bml_total_femur"]]
greater_seventy_bml_tibia = greater_seventy_merged[["time", "right_tkr", "bml_total_tibia"]]


less_sixty_JSN_merged = less_sixty_merged[["time", "right_tkr", "V00XRJSM",'V00XRJSL']]
sixty_seventy_JSN_merged = sixty_seventy_merged[["time", "right_tkr", "V00XRJSM",'V00XRJSL']]
greater_seventy_JSN_merged = greater_seventy_merged[["time", "right_tkr", "V00XRJSM",'V00XRJSL']]


# In[ ]:


# Determing number of patients in each age range
print(len(less_sixty_sbl_merged))
print(len(sixty_seventy_sbl_merged))
print(len(greater_seventy_sbl_merged))


# In[ ]:


# from sksurv.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
# from sksurv.metrics import (
#     concordance_index_censored,
#     concordance_index_ipcw,
#     cumulative_dynamic_auc,
#     integrated_brier_score,
# )

# from sklearn.preprocessing import StandardScaler
# merged_tkr_time = less_sixty_sbl_merged[["right_tkr", "time"]].copy()
# merged_tkr_time["right_tkr"] = merged_tkr_time["right_tkr"].astype(bool)
# merged_tkr_time = merged_tkr_time.to_numpy()
# aux_test = [(e1, e2) for e1, e2 in merged_tkr_time]
# merged_structured = np.array(
# aux_test, dtype=[("Status", "?"), ("Survival_in_days", "<f8")]
# )
# males_merged_sbl = less_sixty_sbl_merged.copy().drop("time", axis=1)  # keep as a dataframe
# biomarker_vals = males_merged_sbl.copy().drop("right_tkr", axis=1)  # keep as a dataframe
# scaler = StandardScaler()
# biomarker_vals = scaler.fit_transform(biomarker_vals)
# biomarker_vals = pd.DataFrame(biomarker_vals)
# print(biomarker_vals)
# train, test = train_test_split(less_sixty_sbl_merged, test_size=0.2, random_state=120)
# x_all_T = OneHotEncoder().fit_transform(biomarker_vals)

# # Selecting Timeframe in which to observe event occurence
# t_max = min(test["time"].max(), train["time"].max())
# t_min = max(test["time"].min(), train["time"].min())
# # Sci-kit survival CPH model
# estimator = CoxPHSurvivalAnalysis(alpha = 10).fit(biomarker_vals, merged_structured)
# survs = estimator.predict_survival_function(x_all_T)
# survs_predict = estimator.predict(x_all_T)
# times = np.arange(t_min, t_max)
# preds = np.asarray([[fn(t) for t in times] for fn in survs])
# # Integrated brier score in order to account for change to patients over the course of the study
# score = integrated_brier_score(merged_structured, merged_structured, preds, times)
# # Concordance index mdoel output
# c_harrell = concordance_index_censored(merged_structured["Status"], merged_structured["Survival_in_days"], survs_predict)

# print('concordance index censored scikit:{:.3f}'.format(c_harrell[0]))
# print("Brier Score {:.3f}".format(score))


# In[ ]:


from all_knee_sbl_bootstrap import cph_creator_SBL



# In[ ]:


# CPH Model for SBL Merged data for patients AGE < 60
cindex_1, brier_1, auc_1  = cph_creator_SBL(less_sixty_sbl_merged, True)


# In[ ]:


# CPH Model for SBL Femur data for patients AGE < 60
cindex_2, brier_2, auc_2 = cph_creator_SBL(less_sixty_sbl_femur, True)


# In[ ]:


# CPH Model for SBL Tibia data for patients AGE < 60
cindex_3, brier_3, auc_3 = cph_creator_SBL(less_sixty_sbl_tibia, True)


# In[ ]:


# CPH Model for BML Merged data for patients AGE < 60
cindex_4, brier_4, auc_4 = cph_creator_SBL(less_sixty_bml_merged, False)


# In[ ]:


# CPH Model for BML Femur data for patients AGE < 60
cindex_5, brier_5, auc_5 = cph_creator_SBL(less_sixty_bml_femur, False)


# In[ ]:


# CPH Model for BML Tibia data for patients AGE < 60
cindex_6, brier_6, auc_6  = cph_creator_SBL(less_sixty_bml_tibia, True)


# In[ ]:


# CPH Model for JSN data for patients AGE < 60
cindex_7, brier_7, auc_7 = cph_creator_SBL(less_sixty_JSN_merged, False)


# In[ ]:


mean_value_1 = cindex_1[1]
ci1_lower = cindex_1[0][0]
ci1_upper = cindex_1[0][1]
plt.errorbar(x=[1], y=[mean_value_1], yerr=[[mean_value_1 - ci1_lower], [ci1_upper - mean_value_1]], fmt='o', capsize=5,label='SBL Merged')


mean_value_2 = cindex_2[1]
ci2_lower = cindex_2[0][0]
ci2_upper = cindex_2[0][1]
plt.errorbar(x=[2], y=[mean_value_2], yerr=[[mean_value_2 - ci2_lower], [ci2_upper - mean_value_2]], fmt='o', capsize=5,label='SBL Femur')


mean_value_3 = cindex_3[1]
ci3_lower = cindex_3[0][0]
ci3_upper = cindex_3[0][1]
plt.errorbar(x=[3], y=[mean_value_3], yerr=[[mean_value_3 - ci3_lower], [ci3_upper - mean_value_3]], fmt='o', capsize=5,label='SBL Tibia')

mean_value_4 = cindex_4[1]
ci4_lower = cindex_4[0][0]
ci4_upper = cindex_4[0][1]
plt.errorbar(x=[4], y=[mean_value_4], yerr=[[mean_value_4 - ci4_lower], [ci4_upper - mean_value_4]], fmt='o', capsize=5,label='BML Merged')

mean_value_5 = cindex_5[1]
ci5_lower = cindex_5[0][0]
ci5_upper = cindex_5[0][1]
plt.errorbar(x=[5], y=[mean_value_5], yerr=[[mean_value_5 - ci5_lower], [ci5_upper - mean_value_5]], fmt='o', capsize=5,label='BML Femur')

mean_value_6 = cindex_6[1]
ci6_lower = cindex_6[0][0]
ci6_upper = cindex_6[0][1]
plt.errorbar(x=[6], y=[mean_value_6], yerr=[[mean_value_6 - ci6_lower], [ci6_upper - mean_value_6]], fmt='o', capsize=5,label='BML Tibia')

mean_value_7 = cindex_7[1]
ci7_lower = cindex_7[0][0]
ci7_upper = cindex_7[0][1]
plt.errorbar(x=[7], y=[mean_value_7], yerr=[[mean_value_7 - ci7_lower], [ci7_upper - mean_value_7]], fmt='o', capsize=5,label='JSN Merged')

labels = ['SBL Merged', 'SBL Femur', 'SBL Tibia', 'BML Merged', 'BML Femur', 'BML Tibia', 'JSN Merged']
plt.xticks([])
plt.ylabel('Mean Concordance Index')
plt.title('Mean Concordance Index and 95% Confidence Interval')
plt.ylim(-0.1,1.1)
# plt.legend(loc = 'lower right')
# Set x-axis ticks and labels
plt.xticks(range(1, 7 + 1), labels, rotation=45, ha='right')
plt.show()
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/AGE_less_60_SBL_RAW_CINDEX.pdf', format='pdf', bbox_inches = 'tight')
plt.clf()

# In[ ]:


mean_value_1 = brier_1[1]
ci1_lower = brier_1[0][0]
ci1_upper = brier_1[0][1]
print(ci1_lower, ci1_upper, mean_value_1)
plt.errorbar(x=[1], y=[mean_value_1], yerr=[[mean_value_1 - ci1_lower], [ci1_upper - mean_value_1]], fmt='o', capsize=5,label='SBL Merged')


mean_value_2 = brier_2[1]
ci2_lower = brier_2[0][0]
ci2_upper = brier_2[0][1]
plt.errorbar(x=[2], y=[mean_value_2], yerr=[[mean_value_2 - ci2_lower], [ci2_upper - mean_value_2]], fmt='o', capsize=5,label='SBL Femur')


mean_value_3 = brier_3[1]
ci3_lower = brier_3[0][0]
ci3_upper = brier_3[0][1]
plt.errorbar(x=[3], y=[mean_value_3], yerr=[[mean_value_3 - ci3_lower], [ci3_upper - mean_value_3]], fmt='o', capsize=5,label='SBL Tibia')

mean_value_4 = brier_4[1]
ci4_lower = brier_4[0][0]
ci4_upper = brier_4[0][1]
plt.errorbar(x=[4], y=[mean_value_4], yerr=[[mean_value_4 - ci4_lower], [ci4_upper - mean_value_4]], fmt='o', capsize=5,label='BML Merged')

mean_value_5 = brier_5[1]
ci5_lower = brier_5[0][0]
ci5_upper = brier_5[0][1]
plt.errorbar(x=[5], y=[mean_value_5], yerr=[[mean_value_5 - ci5_lower], [ci5_upper - mean_value_5]], fmt='o', capsize=5,label='BML Femur')

mean_value_6 = brier_6[1]
ci6_lower = brier_6[0][0]
ci6_upper = brier_6[0][1]
plt.errorbar(x=[6], y=[mean_value_6], yerr=[[mean_value_6 - ci6_lower], [ci6_upper - mean_value_6]], fmt='o', capsize=5,label='BML Tibia')

mean_value_7 = brier_7[1]
ci7_lower = brier_7[0][0]
ci7_upper = brier_7[0][1]
plt.errorbar(x=[7], y=[mean_value_7], yerr=[[mean_value_7 - ci7_lower], [ci7_upper - mean_value_7]], fmt='o', capsize=5,label='JSN Merged')

labels = ['SBL Merged', 'SBL Femur', 'SBL Tibia', 'BML Merged', 'BML Femur', 'BML Tibia', 'JSN Merged']
plt.xticks([])
plt.ylabel('Mean Brier Score')
plt.title('Mean Brier Score and 95% Confidence Interval')
# plt.ylim(-0.1,1.1)
# plt.legend(loc = 'lower right')
# Set x-axis ticks and labels
plt.xticks(range(1, 7 + 1), labels, rotation=45, ha='right')
plt.show()
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/AGE_less_60_SBL_RAW_BRIER.pdf', format='pdf', bbox_inches = 'tight')
plt.clf()

# In[ ]:


mean_value_1 = auc_1[1]
ci1_lower = auc_1[0][0]
ci1_upper = auc_1[0][1]
print(ci1_lower, ci1_upper, mean_value_1)
plt.errorbar(x=[1], y=[mean_value_1], yerr=[[mean_value_1 - ci1_lower], [ci1_upper - mean_value_1]], fmt='o', capsize=5,label='SBL Merged')


mean_value_2 = auc_2[1]
ci2_lower = auc_2[0][0]
ci2_upper = auc_2[0][1]
plt.errorbar(x=[2], y=[mean_value_2], yerr=[[mean_value_2 - ci2_lower], [ci2_upper - mean_value_2]], fmt='o', capsize=5,label='SBL Femur')


mean_value_3 = auc_3[1]
ci3_lower = auc_3[0][0]
ci3_upper = auc_3[0][1]
plt.errorbar(x=[3], y=[mean_value_3], yerr=[[mean_value_3 - ci3_lower], [ci3_upper - mean_value_3]], fmt='o', capsize=5,label='SBL Tibia')

mean_value_4 = auc_4[1]
ci4_lower = auc_4[0][0]
ci4_upper = auc_4[0][1]
plt.errorbar(x=[4], y=[mean_value_4], yerr=[[mean_value_4 - ci4_lower], [ci4_upper - mean_value_4]], fmt='o', capsize=5,label='BML Merged')

mean_value_5 = auc_5[1]
ci5_lower = auc_5[0][0]
ci5_upper = auc_5[0][1]
plt.errorbar(x=[5], y=[mean_value_5], yerr=[[mean_value_5 - ci5_lower], [ci5_upper - mean_value_5]], fmt='o', capsize=5,label='BML Femur')

mean_value_6 = auc_6[1]
ci6_lower = auc_6[0][0]
ci6_upper = auc_6[0][1]
plt.errorbar(x=[6], y=[mean_value_6], yerr=[[mean_value_6 - ci6_lower], [ci6_upper - mean_value_6]], fmt='o', capsize=5,label='BML Tibia')

mean_value_7 = auc_7[1]
ci7_lower = auc_7[0][0]
ci7_upper = auc_7[0][1]
plt.errorbar(x=[7], y=[mean_value_7], yerr=[[mean_value_7 - ci7_lower], [ci7_upper - mean_value_7]], fmt='o', capsize=5,label='JSN Merged')

labels = ['SBL Merged', 'SBL Femur', 'SBL Tibia', 'BML Merged', 'BML Femur', 'BML Tibia', 'JSN Merged']
plt.xticks([])
plt.ylabel('Average Mean Time-Dependent AUC')
plt.title('Average Mean Time-Dependent AUC and 95% Confidence Interval')
plt.ylim(-0.1,1.1)
# plt.legend(loc = 'lower right')
# Set x-axis ticks and labels
plt.xticks(range(1, 7 + 1), labels, rotation=45, ha='right')
plt.show()
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/AGE_less_60_SBL_RAW_AUC.pdf', format='pdf', bbox_inches = 'tight')
plt.clf()

# In[ ]:


# Column names
columns = ['Cindex', 'Brier', 'AUC']

# Create an empty DataFrame
table_df_1 = pd.DataFrame(columns=columns)


row_1 = {'Cindex': cindex_1, 'Brier': brier_1, 'AUC': auc_1}
row_2 = {'Cindex': cindex_2, 'Brier': brier_2, 'AUC': auc_2}
row_3 = {'Cindex': cindex_3, 'Brier': brier_3, 'AUC': auc_3}
row_4 = {'Cindex': cindex_4, 'Brier': brier_4, 'AUC': auc_4}
row_5 = {'Cindex': cindex_5, 'Brier': brier_5, 'AUC': auc_5}
row_6 = {'Cindex': cindex_6, 'Brier': brier_6, 'AUC': auc_6}
row_7 = {'Cindex': cindex_7, 'Brier': brier_7, 'AUC': auc_7}
table_df_1 = table_df_1.append(row_1, ignore_index=True)
table_df_1 = table_df_1.append(row_2, ignore_index=True)
table_df_1 = table_df_1.append(row_3, ignore_index=True)
table_df_1 = table_df_1.append(row_4, ignore_index=True)
table_df_1 = table_df_1.append(row_5, ignore_index=True)
table_df_1 = table_df_1.append(row_6, ignore_index=True)
table_df_1 = table_df_1.append(row_7, ignore_index=True)
table_df_1.to_csv('/home/tsurendr/TABLES/AGE_less_60_SBL_RAW.csv', index=False)


# In[ ]:


# CPH Model for SBL Merged data for patients AGE 60 - 70
cindex_8, brier_8, auc_8  = cph_creator_SBL(sixty_seventy_sbl_merged, True)


# In[ ]:


# CPH Model for SBL Femur data for patients AGE 60 - 70
cindex_9, brier_9, auc_9  = cph_creator_SBL(sixty_seventy_sbl_femur, True)


# In[ ]:


# CPH Model for SBL Tibia data for patients AGE 60 - 70
cindex_10, brier_10, auc_10  = cph_creator_SBL(sixty_seventy_sbl_tibia, True)


# In[ ]:


# CPH Model for BML Merged data for patients AGE 60 - 70
cindex_11, brier_11, auc_11  = cph_creator_SBL(sixty_seventy_bml_merged, False)


# In[ ]:


# CPH Model for BML Femur data for patients AGE 60 - 70
cindex_12, brier_12, auc_12  = cph_creator_SBL(sixty_seventy_bml_femur, False)


# In[ ]:


# CPH Model for BML Tibia data for patients AGE 60 - 70
cindex_13, brier_13, auc_13 = cph_creator_SBL(sixty_seventy_bml_tibia, False)


# In[ ]:


# CPH Model for JSN data for patients AGE 60 - 70
cindex_14, brier_14, auc_14  = cph_creator_SBL(sixty_seventy_JSN_merged, True)


# In[ ]:


mean_value_1 = cindex_8[1]
ci1_lower = cindex_8[0][0]
ci1_upper = cindex_8[0][1]
plt.errorbar(x=[1], y=[mean_value_1], yerr=[[mean_value_1 - ci1_lower], [ci1_upper - mean_value_1]], fmt='o', capsize=5,label='SBL Merged')


mean_value_2 = cindex_9[1]
ci2_lower = cindex_9[0][0]
ci2_upper = cindex_9[0][1]
plt.errorbar(x=[2], y=[mean_value_2], yerr=[[mean_value_2 - ci2_lower], [ci2_upper - mean_value_2]], fmt='o', capsize=5,label='SBL Femur')


mean_value_3 = cindex_10[1]
ci3_lower = cindex_10[0][0]
ci3_upper = cindex_10[0][1]
plt.errorbar(x=[3], y=[mean_value_3], yerr=[[mean_value_3 - ci3_lower], [ci3_upper - mean_value_3]], fmt='o', capsize=5,label='SBL Tibia')

mean_value_4 = cindex_11[1]
ci4_lower = cindex_11[0][0]
ci4_upper = cindex_11[0][1]
plt.errorbar(x=[4], y=[mean_value_4], yerr=[[mean_value_4 - ci4_lower], [ci4_upper - mean_value_4]], fmt='o', capsize=5,label='BML Merged')

mean_value_5 = cindex_12[1]
ci5_lower = cindex_12[0][0]
ci5_upper = cindex_12[0][1]
plt.errorbar(x=[5], y=[mean_value_5], yerr=[[mean_value_5 - ci5_lower], [ci5_upper - mean_value_5]], fmt='o', capsize=5,label='BML Femur')

mean_value_6 = cindex_13[1]
ci6_lower = cindex_13[0][0]
ci6_upper = cindex_13[0][1]
plt.errorbar(x=[6], y=[mean_value_6], yerr=[[mean_value_6 - ci6_lower], [ci6_upper - mean_value_6]], fmt='o', capsize=5,label='BML Tibia')

mean_value_7 = cindex_14[1]
ci7_lower = cindex_14[0][0]
ci7_upper = cindex_14[0][1]
plt.errorbar(x=[7], y=[mean_value_7], yerr=[[mean_value_7 - ci7_lower], [ci7_upper - mean_value_7]], fmt='o', capsize=5,label='JSN Merged')

labels = ['SBL Merged', 'SBL Femur', 'SBL Tibia', 'BML Merged', 'BML Femur', 'BML Tibia', 'JSN Merged']
plt.xticks([])
plt.ylabel('Mean Concordance Index')
plt.title('Mean Concordance Index and 95% Confidence Interval')
plt.ylim(-0.1,1.1)
# plt.legend(loc = 'lower right')
# Set x-axis ticks and labels
plt.xticks(range(1, 7 + 1), labels, rotation=45, ha='right')
plt.show()
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/AGE_60_70_SBL_RAW_CINDEX.pdf', format='pdf', bbox_inches = 'tight')
plt.clf()

# In[ ]:


mean_value_1 = brier_8[1]
ci1_lower = brier_8[0][0]
ci1_upper = brier_8[0][1]
print(ci1_lower, ci1_upper, mean_value_1)
plt.errorbar(x=[1], y=[mean_value_1], yerr=[[mean_value_1 - ci1_lower], [ci1_upper - mean_value_1]], fmt='o', capsize=5,label='SBL Merged')


mean_value_2 = brier_9[1]
ci2_lower = brier_9[0][0]
ci2_upper = brier_9[0][1]
plt.errorbar(x=[2], y=[mean_value_2], yerr=[[mean_value_2 - ci2_lower], [ci2_upper - mean_value_2]], fmt='o', capsize=5,label='SBL Femur')


mean_value_3 = brier_10[1]
ci3_lower = brier_10[0][0]
ci3_upper = brier_10[0][1]
plt.errorbar(x=[3], y=[mean_value_3], yerr=[[mean_value_3 - ci3_lower], [ci3_upper - mean_value_3]], fmt='o', capsize=5,label='SBL Tibia')

mean_value_4 = brier_11[1]
ci4_lower = brier_11[0][0]
ci4_upper = brier_11[0][1]
plt.errorbar(x=[4], y=[mean_value_4], yerr=[[mean_value_4 - ci4_lower], [ci4_upper - mean_value_4]], fmt='o', capsize=5,label='BML Merged')

mean_value_5 = brier_12[1]
ci5_lower = brier_12[0][0]
ci5_upper = brier_12[0][1]
plt.errorbar(x=[5], y=[mean_value_5], yerr=[[mean_value_5 - ci5_lower], [ci5_upper - mean_value_5]], fmt='o', capsize=5,label='BML Femur')

mean_value_6 = brier_13[1]
ci6_lower = brier_13[0][0]
ci6_upper = brier_13[0][1]
plt.errorbar(x=[6], y=[mean_value_6], yerr=[[mean_value_6 - ci6_lower], [ci6_upper - mean_value_6]], fmt='o', capsize=5,label='BML Tibia')

mean_value_7 = brier_14[1]
ci7_lower = brier_14[0][0]
ci7_upper = brier_14[0][1]
plt.errorbar(x=[7], y=[mean_value_7], yerr=[[mean_value_7 - ci7_lower], [ci7_upper - mean_value_7]], fmt='o', capsize=5,label='JSN Merged')

labels = ['SBL Merged', 'SBL Femur', 'SBL Tibia', 'BML Merged', 'BML Femur', 'BML Tibia', 'JSN Merged']
plt.xticks([])
plt.ylabel('Mean Brier Score')
plt.title('Mean Brier Score and 95% Confidence Interval')
# plt.ylim(-0.1,1.1)
# plt.legend(loc = 'lower right')
# Set x-axis ticks and labels
plt.xticks(range(1, 7 + 1), labels, rotation=45, ha='right')
plt.show()
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/AGE_60_70_SBL_RAW_BRIER.pdf', format='pdf', bbox_inches = 'tight')
plt.clf()

# In[ ]:


mean_value_1 = auc_8[1]
ci1_lower = auc_8[0][0]
ci1_upper = auc_8[0][1]
print(ci1_lower, ci1_upper, mean_value_1)
plt.errorbar(x=[1], y=[mean_value_1], yerr=[[mean_value_1 - ci1_lower], [ci1_upper - mean_value_1]], fmt='o', capsize=5,label='SBL Merged')


mean_value_2 = auc_9[1]
ci2_lower = auc_9[0][0]
ci2_upper = auc_9[0][1]
plt.errorbar(x=[2], y=[mean_value_2], yerr=[[mean_value_2 - ci2_lower], [ci2_upper - mean_value_2]], fmt='o', capsize=5,label='SBL Femur')


mean_value_3 = auc_10[1]
ci3_lower = auc_10[0][0]
ci3_upper = auc_10[0][1]
plt.errorbar(x=[3], y=[mean_value_3], yerr=[[mean_value_3 - ci3_lower], [ci3_upper - mean_value_3]], fmt='o', capsize=5,label='SBL Tibia')

mean_value_4 = auc_11[1]
ci4_lower = auc_11[0][0]
ci4_upper = auc_11[0][1]
plt.errorbar(x=[4], y=[mean_value_4], yerr=[[mean_value_4 - ci4_lower], [ci4_upper - mean_value_4]], fmt='o', capsize=5,label='BML Merged')

mean_value_5 = auc_12[1]
ci5_lower = auc_12[0][0]
ci5_upper = auc_12[0][1]
plt.errorbar(x=[5], y=[mean_value_5], yerr=[[mean_value_5 - ci5_lower], [ci5_upper - mean_value_5]], fmt='o', capsize=5,label='BML Femur')

mean_value_6 = auc_13[1]
ci6_lower = auc_13[0][0]
ci6_upper = auc_13[0][1]
plt.errorbar(x=[6], y=[mean_value_6], yerr=[[mean_value_6 - ci6_lower], [ci6_upper - mean_value_6]], fmt='o', capsize=5,label='BML Tibia')

mean_value_7 = auc_14[1]
ci7_lower = auc_14[0][0]
ci7_upper = auc_14[0][1]
plt.errorbar(x=[7], y=[mean_value_7], yerr=[[mean_value_7 - ci7_lower], [ci7_upper - mean_value_7]], fmt='o', capsize=5,label='JSN Merged')

labels = ['SBL Merged', 'SBL Femur', 'SBL Tibia', 'BML Merged', 'BML Femur', 'BML Tibia', 'JSN Merged']
plt.xticks([])
plt.ylabel('Average Mean Time-Dependent AUC')
plt.title('Average Mean Time-Dependent AUC and 95% Confidence Interval')
plt.ylim(-0.1,1.1)
# plt.legend(loc = 'lower right')
# Set x-axis ticks and labels
plt.xticks(range(1, 7 + 1), labels, rotation=45, ha='right')
plt.show()
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/AGE_60_70_SBL_RAW_AUC.pdf', format='pdf', bbox_inches = 'tight')
plt.clf()

# In[ ]:


# Column names
columns = ['Cindex', 'Brier', 'AUC']

# Create an empty DataFrame
table_df_2 = pd.DataFrame(columns=columns)


row_8 = {'Cindex': cindex_8, 'Brier': brier_8, 'AUC': auc_8}
row_9 = {'Cindex': cindex_9, 'Brier': brier_9, 'AUC': auc_9}
row_10 = {'Cindex': cindex_10, 'Brier': brier_10, 'AUC': auc_10}
row_11 = {'Cindex': cindex_11, 'Brier': brier_11, 'AUC': auc_11}
row_12 = {'Cindex': cindex_12, 'Brier': brier_12, 'AUC': auc_12}
row_13 = {'Cindex': cindex_13, 'Brier': brier_13, 'AUC': auc_13}
row_14 = {'Cindex': cindex_14, 'Brier': brier_14, 'AUC': auc_14}
table_df_2 = table_df_2.append(row_8, ignore_index=True)
table_df_2 = table_df_2.append(row_9, ignore_index=True)
table_df_2 = table_df_2.append(row_10, ignore_index=True)
table_df_2 = table_df_2.append(row_11, ignore_index=True)
table_df_2 = table_df_2.append(row_12, ignore_index=True)
table_df_2 = table_df_2.append(row_13, ignore_index=True)
table_df_2 = table_df_2.append(row_14, ignore_index=True)
table_df_2.to_csv('/home/tsurendr/TABLES/AGE_60_70_SBL_RAW.csv', index=False)


# In[ ]:


# CPH Model for SBL Merged data for patients AGE > 70
cindex_15, brier_15, auc_15  = cph_creator_SBL(greater_seventy_sbl_merged, True)


# In[ ]:


# CPH Model for SBL Femur data for patients AGE > 70
cindex_16, brier_16, auc_16 = cph_creator_SBL(greater_seventy_sbl_femur, True)


# In[ ]:


# CPH Model for SBL Tibia data for patients AGE > 70
cindex_17, brier_17, auc_17 = cph_creator_SBL(greater_seventy_sbl_tibia, True)


# In[ ]:


# CPH Model for BML Merged data for patients AGE > 70
cindex_18, brier_18, auc_18 = cph_creator_SBL(greater_seventy_bml_merged, False)


# In[ ]:


# CPH Model for BML Femur data for patients AGE > 70
cindex_19, brier_19, auc_19 = cph_creator_SBL(greater_seventy_bml_femur, False)


# In[ ]:


# CPH Model for BML Tibia data for patients AGE > 70
cindex_20, brier_20, auc_20 = cph_creator_SBL(greater_seventy_bml_tibia, False)


# In[ ]:


# CPH Model for JSN data for patients AGE > 70
cindex_21, brier_21, auc_21 = cph_creator_SBL(greater_seventy_JSN_merged, True)


# In[ ]:


mean_value_1 = cindex_15[1]
ci1_lower = cindex_15[0][0]
ci1_upper = cindex_15[0][1]
plt.errorbar(x=[1], y=[mean_value_1], yerr=[[mean_value_1 - ci1_lower], [ci1_upper - mean_value_1]], fmt='o', capsize=5,label='SBL Merged')


mean_value_2 = cindex_16[1]
ci2_lower = cindex_16[0][0]
ci2_upper = cindex_16[0][1]
plt.errorbar(x=[2], y=[mean_value_2], yerr=[[mean_value_2 - ci2_lower], [ci2_upper - mean_value_2]], fmt='o', capsize=5,label='SBL Femur')


mean_value_3 = cindex_17[1]
ci3_lower = cindex_17[0][0]
ci3_upper = cindex_17[0][1]
plt.errorbar(x=[3], y=[mean_value_3], yerr=[[mean_value_3 - ci3_lower], [ci3_upper - mean_value_3]], fmt='o', capsize=5,label='SBL Tibia')

mean_value_4 = cindex_18[1]
ci4_lower = cindex_18[0][0]
ci4_upper = cindex_18[0][1]
plt.errorbar(x=[4], y=[mean_value_4], yerr=[[mean_value_4 - ci4_lower], [ci4_upper - mean_value_4]], fmt='o', capsize=5,label='BML Merged')

mean_value_5 = cindex_19[1]
ci5_lower = cindex_19[0][0]
ci5_upper = cindex_19[0][1]
plt.errorbar(x=[5], y=[mean_value_5], yerr=[[mean_value_5 - ci5_lower], [ci5_upper - mean_value_5]], fmt='o', capsize=5,label='BML Femur')

mean_value_6 = cindex_20[1]
ci6_lower = cindex_20[0][0]
ci6_upper = cindex_20[0][1]
plt.errorbar(x=[6], y=[mean_value_6], yerr=[[mean_value_6 - ci6_lower], [ci6_upper - mean_value_6]], fmt='o', capsize=5,label='BML Tibia')

mean_value_7 = cindex_21[1]
ci7_lower = cindex_21[0][0]
ci7_upper = cindex_21[0][1]
plt.errorbar(x=[7], y=[mean_value_7], yerr=[[mean_value_7 - ci7_lower], [ci7_upper - mean_value_7]], fmt='o', capsize=5,label='JSN Merged')

labels = ['SBL Merged', 'SBL Femur', 'SBL Tibia', 'BML Merged', 'BML Femur', 'BML Tibia', 'JSN Merged']
plt.xticks([])
plt.ylabel('Mean Concordance Index')
plt.title('Mean Concordance Index and 95% Confidence Interval')
plt.ylim(-0.1,1.1)
# plt.legend(loc = 'lower right')
# Set x-axis ticks and labels
plt.xticks(range(1, 7 + 1), labels, rotation=45, ha='right')
plt.show()
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/AGE_greater_70_SBL_RAW_CINDEX.pdf', format='pdf', bbox_inches = 'tight')
plt.clf()

# In[ ]:


mean_value_1 = brier_15[1]
ci1_lower = brier_15[0][0]
ci1_upper = brier_15[0][1]
print(ci1_lower, ci1_upper, mean_value_1)
plt.errorbar(x=[1], y=[mean_value_1], yerr=[[mean_value_1 - ci1_lower], [ci1_upper - mean_value_1]], fmt='o', capsize=5,label='SBL Merged')


mean_value_2 = brier_16[1]
ci2_lower = brier_16[0][0]
ci2_upper = brier_16[0][1]
plt.errorbar(x=[2], y=[mean_value_2], yerr=[[mean_value_2 - ci2_lower], [ci2_upper - mean_value_2]], fmt='o', capsize=5,label='SBL Femur')


mean_value_3 = brier_17[1]
ci3_lower = brier_17[0][0]
ci3_upper = brier_17[0][1]
plt.errorbar(x=[3], y=[mean_value_3], yerr=[[mean_value_3 - ci3_lower], [ci3_upper - mean_value_3]], fmt='o', capsize=5,label='SBL Tibia')

mean_value_4 = brier_18[1]
ci4_lower = brier_18[0][0]
ci4_upper = brier_18[0][1]
plt.errorbar(x=[4], y=[mean_value_4], yerr=[[mean_value_4 - ci4_lower], [ci4_upper - mean_value_4]], fmt='o', capsize=5,label='BML Merged')

mean_value_5 = brier_19[1]
ci5_lower = brier_19[0][0]
ci5_upper = brier_19[0][1]
plt.errorbar(x=[5], y=[mean_value_5], yerr=[[mean_value_5 - ci5_lower], [ci5_upper - mean_value_5]], fmt='o', capsize=5,label='BML Femur')

mean_value_6 = brier_20[1]
ci6_lower = brier_20[0][0]
ci6_upper = brier_20[0][1]
plt.errorbar(x=[6], y=[mean_value_6], yerr=[[mean_value_6 - ci6_lower], [ci6_upper - mean_value_6]], fmt='o', capsize=5,label='BML Tibia')

mean_value_7 = brier_21[1]
ci7_lower = brier_21[0][0]
ci7_upper = brier_21[0][1]
plt.errorbar(x=[7], y=[mean_value_7], yerr=[[mean_value_7 - ci7_lower], [ci7_upper - mean_value_7]], fmt='o', capsize=5,label='JSN Merged')

labels = ['SBL Merged', 'SBL Femur', 'SBL Tibia', 'BML Merged', 'BML Femur', 'BML Tibia', 'JSN Merged']
plt.xticks([])
plt.ylabel('Mean Brier Score')
plt.title('Mean Brier Score and 95% Confidence Interval')
# plt.ylim(-0.1,1.1)
# plt.legend(loc = 'lower right')
# Set x-axis ticks and labels
plt.xticks(range(1, 7 + 1), labels, rotation=45, ha='right')
plt.show()
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/AGE_greater_70_SBL_RAW_BRIER.pdf', format='pdf', bbox_inches = 'tight')
plt.clf()

# In[ ]:


mean_value_1 = auc_15[1]
ci1_lower = auc_15[0][0]
ci1_upper = auc_15[0][1]
print(ci1_lower, ci1_upper, mean_value_1)
plt.errorbar(x=[1], y=[mean_value_1], yerr=[[mean_value_1 - ci1_lower], [ci1_upper - mean_value_1]], fmt='o', capsize=5,label='SBL Merged')


mean_value_2 = auc_16[1]
ci2_lower = auc_16[0][0]
ci2_upper = auc_16[0][1]
plt.errorbar(x=[2], y=[mean_value_2], yerr=[[mean_value_2 - ci2_lower], [ci2_upper - mean_value_2]], fmt='o', capsize=5,label='SBL Femur')


mean_value_3 = auc_17[1]
ci3_lower = auc_17[0][0]
ci3_upper = auc_17[0][1]
plt.errorbar(x=[3], y=[mean_value_3], yerr=[[mean_value_3 - ci3_lower], [ci3_upper - mean_value_3]], fmt='o', capsize=5,label='SBL Tibia')

mean_value_4 = auc_18[1]
ci4_lower = auc_18[0][0]
ci4_upper = auc_18[0][1]
plt.errorbar(x=[4], y=[mean_value_4], yerr=[[mean_value_4 - ci4_lower], [ci4_upper - mean_value_4]], fmt='o', capsize=5,label='BML Merged')

mean_value_5 = auc_19[1]
ci5_lower = auc_19[0][0]
ci5_upper = auc_19[0][1]
plt.errorbar(x=[5], y=[mean_value_5], yerr=[[mean_value_5 - ci5_lower], [ci5_upper - mean_value_5]], fmt='o', capsize=5,label='BML Femur')

mean_value_6 = auc_20[1]
ci6_lower = auc_20[0][0]
ci6_upper = auc_20[0][1]
plt.errorbar(x=[6], y=[mean_value_6], yerr=[[mean_value_6 - ci6_lower], [ci6_upper - mean_value_6]], fmt='o', capsize=5,label='BML Tibia')

mean_value_7 = auc_21[1]
ci7_lower = auc_21[0][0]
ci7_upper = auc_21[0][1]
plt.errorbar(x=[7], y=[mean_value_7], yerr=[[mean_value_7 - ci7_lower], [ci7_upper - mean_value_7]], fmt='o', capsize=5,label='JSN Merged')

labels = ['SBL Merged', 'SBL Femur', 'SBL Tibia', 'BML Merged', 'BML Femur', 'BML Tibia', 'JSN Merged']
plt.xticks([])
plt.ylabel('Average Mean Time-Dependent AUC')
plt.title('Average Mean Time-Dependent AUC and 95% Confidence Interval')
plt.ylim(-0.1,1.1)
# plt.legend(loc = 'lower right')
# Set x-axis ticks and labels
plt.xticks(range(1, 7 + 1), labels, rotation=45, ha='right')
plt.show()
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/AGE_greater_70_SBL_RAW_AUC.pdf', format='pdf', bbox_inches = 'tight')
plt.clf()

# In[ ]:


# Column names
columns = ['Cindex', 'Brier', 'AUC']

# Create an empty DataFrame
table_df_3 = pd.DataFrame(columns=columns)


row_15 = {'Cindex': cindex_15, 'Brier': brier_15, 'AUC': auc_15}
row_16 = {'Cindex': cindex_16, 'Brier': brier_16, 'AUC': auc_16}
row_17 = {'Cindex': cindex_17, 'Brier': brier_17, 'AUC': auc_17}
row_18 = {'Cindex': cindex_18, 'Brier': brier_18, 'AUC': auc_18}
row_19 = {'Cindex': cindex_19, 'Brier': brier_19, 'AUC': auc_19}
row_20 = {'Cindex': cindex_20, 'Brier': brier_20, 'AUC': auc_20}
row_21 = {'Cindex': cindex_21, 'Brier': brier_21, 'AUC': auc_21}
table_df_3 = table_df_3.append(row_15, ignore_index=True)
table_df_3 = table_df_3.append(row_16, ignore_index=True)
table_df_3 = table_df_3.append(row_17, ignore_index=True)
table_df_3 = table_df_3.append(row_18, ignore_index=True)
table_df_3 = table_df_3.append(row_19, ignore_index=True)
table_df_3 = table_df_3.append(row_20, ignore_index=True)
table_df_3 = table_df_3.append(row_21, ignore_index=True)
table_df_3.to_csv('/home/tsurendr/TABLES/AGE_greater_70_SBL_RAW.csv', index=False)

