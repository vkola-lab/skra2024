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
    ['id',"time", "right_tkr", "P02SEX",'KL_grade', 'RACE','bml_total_merged','bml_total_femur','bml_total_tibia', "V00XRJSM",'V00XRJSL']]], axis = 1)

oai_right_temp_SBL_Femur_right = pd.concat( [oai_SBL_KL_BML_right.loc[:, 'F0_merged':'F199_merged'], oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX",'bml_total_femur','KL_grade']]], axis = 1)

oai_right_temp_SBL_Tibia_right = pd.concat( [oai_SBL_KL_BML_right.loc[:, 'T0_merged':'T199_merged'], oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX",'bml_total_tibia','KL_grade']]], axis = 1)


oai_right_temp_BML_Merged_right = oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX", "bml_total_merged",'KL_grade']
]
oai_right_temp_BML_Femur_right = oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX", "bml_total_femur",'KL_grade']
]
oai_right_temp_BML_Tibia_right = oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX", "bml_total_tibia",'KL_grade']
]

oai_right_temp_JSN_Merged_right = oai_SBL_KL_BML_right[
    ['id',"time", "right_tkr", "P02SEX", "V00XRJSM",'V00XRJSL', 'KL_Grade']
]


# left side
oai_right_temp_SBL_Merged_left = pd.concat( [oai_SBL_KL_BML_left.loc[:, 'F0_merged':'T199_merged'], oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX",'KL_grade','bml_total_merged','bml_total_femur','bml_total_tibia', "V00XRJSM",'V00XRJSL']]], axis = 1)

oai_right_temp_SBL_Femur_left = pd.concat( [oai_SBL_KL_BML_left.loc[:, 'F0_merged':'F199_merged'], oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX",'bml_total_femur','KL_grade']]], axis = 1)

oai_right_temp_SBL_Tibia_left = pd.concat( [oai_SBL_KL_BML_left.loc[:, 'T0_merged':'T199_merged'], oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX",'bml_total_tibia','KL_grade']]], axis = 1)


oai_right_temp_BML_Merged_left = oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX", "bml_total_merged",'KL_grade']
]
oai_right_temp_BML_Femur_left = oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX", "bml_total_femur",'KL_grade']
]
oai_right_temp_BML_Tibia_left = oai_SBL_KL_BML_left[
    ['id',"time", "right_tkr", "P02SEX", "bml_total_tibia",'KL_grade']
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


oai_right_temp_SBL_Merged_all.drop(oai_right_temp_SBL_Merged_all[(oai_right_temp_SBL_Merged_all['RACE'] == '.D: Don t Know/Unknown/Uncertain') ].index, inplace=True)


# In[ ]:


# dropping knee with the smaller bml from patients who have 2 knees in the table in order to avoid confounding variables. 
oai_right_temp_SBL_Merged_all = oai_right_temp_SBL_Merged_all.sort_values( 'KL_grade').drop_duplicates('id', keep='last').sort_index()
oai_right_temp_SBL_Femur_all = oai_right_temp_SBL_Femur_all.sort_values('KL_grade').drop_duplicates('id', keep='last').sort_index()
oai_right_temp_SBL_Tibia_all = oai_right_temp_SBL_Tibia_all.sort_values('KL_grade').drop_duplicates('id', keep='last').sort_index()
oai_right_temp_BML_Merged_all = oai_right_temp_BML_Merged_all.sort_values('KL_grade').drop_duplicates('id', keep='last').sort_index()
oai_right_temp_BML_Femur_all = oai_right_temp_BML_Femur_all.sort_values('KL_grade').drop_duplicates('id', keep='last').sort_index()
oai_right_temp_BML_Tibia_all = oai_right_temp_BML_Tibia_all.sort_values('KL_grade').drop_duplicates('id', keep='last').sort_index()
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
oai_right_temp_SBL_Merged_all = oai_right_temp_SBL_Merged_all.drop(columns=["P02SEX",'id'])
oai_right_temp_SBL_Femur_all = oai_right_temp_SBL_Femur_all.drop(columns=["P02SEX",'id'])
oai_right_temp_SBL_Tibia_all = oai_right_temp_SBL_Tibia_all.drop(columns=["P02SEX",'id'])
oai_right_temp_BML_Merged_all = oai_right_temp_BML_Merged_all.drop(columns=["P02SEX",'id'])
oai_right_temp_BML_Femur_all = oai_right_temp_BML_Femur_all.drop(columns=["P02SEX",'id'])
oai_right_temp_BML_Tibia_all = oai_right_temp_BML_Tibia_all.drop(columns=["P02SEX",'id'])
oai_right_temp_JSN_Merged_all = oai_right_temp_JSN_Merged_all.drop(columns=["P02SEX",'id','KL_Grade'])


# In[ ]:


#Splitting populations based on KL Grade
KL_groups_merged = oai_right_temp_SBL_Merged_all.groupby("KL_grade")
KL_groups_femur = oai_right_temp_SBL_Femur_all.groupby("KL_grade")
KL_groups_tibia = oai_right_temp_SBL_Tibia_all.groupby("KL_grade")


kl_0_merged = KL_groups_merged.get_group(0.0)
kl_1_merged = KL_groups_merged.get_group(1.0)
kl_2_merged = KL_groups_merged.get_group(2.0)
kl_3_merged = KL_groups_merged.get_group(3.0)
kl_4_merged = KL_groups_merged.get_group(4.0)

kl_0_femur = KL_groups_femur.get_group(0.0)
kl_1_femur  = KL_groups_femur.get_group(1.0)
kl_2_femur  = KL_groups_femur.get_group(2.0)
kl_3_femur  = KL_groups_femur.get_group(3.0)
kl_4_femur  = KL_groups_femur.get_group(4.0)

kl_0_tibia = KL_groups_tibia.get_group(0.0)
kl_1_tibia = KL_groups_tibia.get_group(1.0)
kl_2_tibia = KL_groups_tibia.get_group(2.0)
kl_3_tibia = KL_groups_tibia.get_group(3.0)
kl_4_tibia = KL_groups_tibia.get_group(4.0)


# In[ ]:


# Selecting SBL, BML, KL Grade, and TKR from each table before information is input into the CPH models


kl_0_sbl_merged =  pd.concat( [kl_0_merged.loc[:, 'F0_merged':'T199_merged'], kl_0_merged[
    ["time", "right_tkr"]]], axis = 1)
kl_0_sbl_femur =  pd.concat( [kl_0_femur.loc[:, 'F0_merged':'F199_merged'], kl_0_femur[
    ["time", "right_tkr"]]], axis = 1)
kl_0_sbl_tibia =  pd.concat( [kl_0_tibia.loc[:, 'T0_merged':'T199_merged'], kl_0_tibia[
    ["time", "right_tkr"]]], axis = 1)
kl_0_bml_merged = kl_0_merged[["time", "right_tkr", "bml_total_merged"]]
kl_0_bml_femur = kl_0_femur[["time", "right_tkr", "bml_total_femur"]]
kl_0_bml_tibia = kl_0_tibia[["time", "right_tkr", "bml_total_tibia"]]
kl_0_JSN_merged = kl_0_merged[["time", "right_tkr", "V00XRJSM",'V00XRJSL']]


kl_1_sbl_merged =  pd.concat( [kl_1_merged.loc[:, 'F0_merged':'T199_merged'], kl_1_merged[
    ["time", "right_tkr"]]], axis = 1)
kl_1_sbl_femur =  pd.concat( [kl_1_femur.loc[:, 'F0_merged':'F199_merged'], kl_1_femur[
    ["time", "right_tkr"]]], axis = 1)
kl_1_sbl_tibia =  pd.concat( [kl_1_tibia.loc[:, 'T0_merged':'T199_merged'], kl_1_tibia[
    ["time", "right_tkr"]]], axis = 1)
kl_1_bml_merged = kl_1_merged[["time", "right_tkr", "bml_total_merged"]]
kl_1_bml_femur = kl_1_femur[["time", "right_tkr", "bml_total_femur"]]
kl_1_bml_tibia = kl_1_tibia[["time", "right_tkr", "bml_total_tibia"]]
kl_1_JSN_merged = kl_1_merged[["time", "right_tkr", "V00XRJSM",'V00XRJSL']]

kl_2_sbl_merged =  pd.concat( [kl_2_merged.loc[:, 'F0_merged':'T199_merged'], kl_2_merged[
    ["time", "right_tkr"]]], axis = 1)
kl_2_sbl_femur =  pd.concat( [kl_2_femur.loc[:, 'F0_merged':'F199_merged'], kl_2_femur[
    ["time", "right_tkr"]]], axis = 1)
kl_2_sbl_tibia =  pd.concat( [kl_2_tibia.loc[:, 'T0_merged':'T199_merged'], kl_2_tibia[
    ["time", "right_tkr"]]], axis = 1)
kl_2_bml_merged = kl_2_merged[["time", "right_tkr", "bml_total_merged"]]
kl_2_bml_femur = kl_2_femur[["time", "right_tkr", "bml_total_femur"]]
kl_2_bml_tibia = kl_2_tibia[["time", "right_tkr", "bml_total_tibia"]]
kl_2_JSN_merged = kl_2_merged[["time", "right_tkr", "V00XRJSM",'V00XRJSL']]


kl_3_sbl_merged =  pd.concat( [kl_3_merged.loc[:, 'F0_merged':'T199_merged'], kl_3_merged[
    ["time", "right_tkr"]]], axis = 1)
kl_3_sbl_femur =  pd.concat( [kl_3_femur.loc[:, 'F0_merged':'F199_merged'], kl_3_femur[
    ["time", "right_tkr"]]], axis = 1)
kl_3_sbl_tibia =  pd.concat( [kl_3_tibia.loc[:, 'T0_merged':'T199_merged'], kl_3_tibia[
    ["time", "right_tkr"]]], axis = 1)
kl_3_bml_merged = kl_3_merged[["time", "right_tkr", "bml_total_merged"]]
kl_3_bml_femur = kl_3_femur[["time", "right_tkr", "bml_total_femur"]]
kl_3_bml_tibia = kl_3_tibia[["time", "right_tkr", "bml_total_tibia"]]
kl_3_JSN_merged = kl_3_merged[["time", "right_tkr", "V00XRJSM",'V00XRJSL']]

kl_4_sbl_merged =  pd.concat( [kl_4_merged.loc[:, 'F0_merged':'T199_merged'], kl_4_merged[
    ["time", "right_tkr"]]], axis = 1)
kl_4_sbl_femur =  pd.concat( [kl_4_femur.loc[:, 'F0_merged':'F199_merged'], kl_4_femur[
    ["time", "right_tkr"]]], axis = 1)
kl_4_sbl_tibia =  pd.concat( [kl_4_tibia.loc[:, 'T0_merged':'T199_merged'], kl_4_tibia[
    ["time", "right_tkr"]]], axis = 1)
kl_4_bml_merged = kl_4_merged[["time", "right_tkr", "bml_total_merged"]]
kl_4_bml_femur = kl_4_femur[["time", "right_tkr", "bml_total_femur"]]
kl_4_bml_tibia = kl_4_tibia[["time", "right_tkr", "bml_total_tibia"]]
kl_4_JSN_merged = kl_4_merged[["time", "right_tkr", "V00XRJSM",'V00XRJSL']]


# In[ ]:


# Determing number of patients in each KL Grade range
print(len(kl_0_sbl_merged))
print(len(kl_1_sbl_merged))
print(len(kl_2_sbl_merged))
print(len(kl_3_sbl_merged))
print(len(kl_4_sbl_merged))


# In[ ]:


# # from all_knee_sbl import cph_creator_SBL
# def cph_creator_SBL(dataframe, sbl_bool):
    
#     #import libraries from lifelines and Sci-kit survival
#     from sksurv.metrics import (
#     concordance_index_censored,
#     concordance_index_ipcw,
#     cumulative_dynamic_auc,
#     integrated_brier_score,
# )
#     import pandas as pd
#     from sksurv.preprocessing import OneHotEncoder
#     from scipy import interpolate
#     import numpy as np
#     from lifelines import CoxPHFitter
#     from sklearn.model_selection import train_test_split
#     from lifelines.utils.sklearn_adapter import sklearn_adapter
#     import lifelines
#     import matplotlib.pyplot as plt
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.model_selection import GridSearchCV, KFold
#     from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
#     import warnings
#     from lifelines.statistics import proportional_hazard_test
#     from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer,RobustScaler, PowerTransformer
#     import warnings
#     warnings.filterwarnings("ignore")

 
#     # checking dataframe to make sure there is high enough variance in each data column
#     ################################################################################################
#     # #initalize lists for columns that might have no variance
#     bad_cols = []
#     vars = []
#     #columns not to check if variance is bad
#     col_not_check = ["time", "right_tkr"]
#     # events = dataframe["right_tkr"].astype(bool)
#     # for i in dataframe.columns:
#     #     if i not in col_not_check:
#     #         tempy1 = dataframe.loc[~events, i].var()
#     #         tempy2 = dataframe.loc[events, i].var()
#     #         vars.append(tempy1)
#     #         vars.append(tempy2)
#     #         # print(i, tempy1, tempy2)
#     #         if dataframe.loc[~events, i].var() < 0.001 or dataframe.loc[events, i].var() < 0.001:
#     #             # tempy1 = dataframe.loc[~events, i].var()
#     #             # tempy2 = dataframe.loc[events, i].var()
#     #             # print(i, tempy1, tempy2)
#     #             dataframe = dataframe.drop(labels=i, axis=1)
#     #             bad_cols.append(i)
#     # print("bad cols: ", bad_cols)
#     # print('variance quartiles: ', np.percentile(vars, [0, 25, 50, 75, 100]))
#     ###############################################################################################
#     print('test1')
    
#     time_col = list(dataframe['time'].copy())
#     right_tkr_col = list(dataframe['right_tkr'].copy())
#     # print(len(time_col), len(right_tkr_col))
#     dataframe = dataframe.drop(['time', 'right_tkr'], axis=1)
#     scaler = MinMaxScaler()
#     dataframe = scaler.fit_transform(dataframe)
#     dataframe = pd.DataFrame(dataframe)
#     dataframe["right_tkr"] = right_tkr_col
#     dataframe["time"] = time_col
#     # print(len(dataframe))
#     # print(dataframe['right_tkr'])
#     # Prepping dataframe for CPH input
#     # Creating Structured Array for Sci-Kit Survival models, while keeping traditional dataframe for lifelines package. 
#     # print('df_cols length:', len(df_cols))
#     # merged_tkr_time = dataframe[["right_tkr", "time"]].copy()
#     # merged_tkr_time["right_tkr"] = merged_tkr_time["right_tkr"].astype(bool)
#     # merged_tkr_time = merged_tkr_time.to_numpy()
#     # aux_test = [(e1, e2) for e1, e2 in merged_tkr_time]
#     # merged_structured = np.array(
#     # aux_test, dtype=[("Status", "?"), ("Survival_in_days", "<f8")]
#     # )
#     # males_merged_sbl = dataframe.copy().drop("time", axis=1)  # keep as a dataframe
#     # biomarker_vals = males_merged_sbl.copy().drop("right_tkr", axis=1)  # keep as a dataframe

#     # # Used for min and max timeframe for brier score computation
#     # train, test = train_test_split(dataframe, test_size=0.2, random_state=120)
#     # print(dataframe.isnull().sum())
#     X_all = dataframe.copy().drop("time", axis=1)
#     Y_all = dataframe.copy().pop("time")

#     base_class = sklearn_adapter(CoxPHFitter, event_col="right_tkr")
#     wf = base_class()

#     #Using GridSearchCV to perform hyperparameter tuning
#     from sklearn.model_selection import GridSearchCV

#     clf = GridSearchCV(
#         wf,
#         {
#             "penalizer": 10.0 ** np.arange(-2, 3),
#         },
#         cv=4,
#     )
#     clf.fit(X_all, Y_all)

#     # Picking best penalizer value for CPH model
#     penalizer_val = clf.best_params_.get("penalizer")
#     print("old penalizer_val: ", penalizer_val)
#     # Fitting Cph model to data

#     cph = CoxPHFitter( penalizer=penalizer_val)
#     #Testing Strata
#     # cph.fit(dataframe_strata_femur, "time", event_col="right_tkr",show_progress=True, robust=True, strata=['F8_femur_strata'])
#     cph.fit(dataframe, "time", event_col="right_tkr",   show_progress=False, robust=True )
#     # cph.fit(strata_df, "time", event_col="right_tkr", strata = strata_columns,   show_progress=True, robust=True )
#     # cph.print_summary(decimals=3)
  

#     print('old dataframe length: ', len(dataframe.columns))
#     # Checks Proportional Hazards Assumption
#     print('SBL_bool', sbl_bool)

#     if sbl_bool == True:
#         results = proportional_hazard_test(cph, dataframe, time_transform='rank')
#         for variable in cph.params_.index.intersection(cph.params_.index):
#             minumum_observed_p_value = results.summary.loc[variable, "p"].min()
#             minumum_observed_p_value = np.round(minumum_observed_p_value,3)
#             # print(variable,minumum_observed_p_value)
#             # print(pairings)
#             if minumum_observed_p_value < 0.05:
#                 # print(pairings)
#                 dataframe = dataframe.drop(columns=[variable])
#         print('new dataframe length: ', len(dataframe.columns))
 
        
#         # X_all = dataframe.copy().drop("time", axis=1)
#         # Y_all = dataframe.copy().pop("time")

#         # base_class = sklearn_adapter(CoxPHFitter, event_col="right_tkr")
#         # wf = base_class()

#         # #Using GridSearchCV to perform hyperparameter tuning
#         # from sklearn.model_selection import GridSearchCV

#         # clf = GridSearchCV(
#         #     wf,
#         #     {
#         #         "penalizer": 10.0 ** np.arange(-2, 3),
#         #     },
#         #     cv=4,
#         # )
#         # clf.fit(X_all, Y_all)

#         # # Picking best penalizer value for CPH model
#         # penalizer_val = clf.best_params_.get("penalizer")
#         # print("new penalizer_val: ", penalizer_val) 
#         # cph = CoxPHFitter( penalizer=penalizer_val)
#         cph.fit(dataframe, "time", event_col="right_tkr",   show_progress=False, robust=True )
#     # cph.print_summary(decimals=3)
#     print('#################################### checkpoint 3')
#     cph_data = cph.predict_survival_function(dataframe)
#     merged_tkr_time = dataframe[["right_tkr", "time"]].copy()
#     merged_tkr_time["right_tkr"] = merged_tkr_time["right_tkr"].astype(bool)
#     merged_tkr_time = merged_tkr_time.to_numpy()
#     aux_test = [(e1, e2) for e1, e2 in merged_tkr_time]
#     merged_structured = np.array(
#     aux_test, dtype=[("Status", "?"), ("Survival_in_days", "<f8")]
#     )
#     males_merged_sbl = dataframe.copy().drop("time", axis=1)  # keep as a dataframe
#     biomarker_vals = males_merged_sbl.copy().drop("right_tkr", axis=1)  # keep as a dataframe

#     # Used for min and max timeframe for brier score computation
#     train, test = train_test_split(dataframe, test_size=0.2, random_state=120)
    
    
#     x_all_T = OneHotEncoder().fit_transform(biomarker_vals)
    
#     # Selecting Timeframe in which to observe event occurence
#     t_max = min(test["time"].max(), train["time"].max())
#     t_min = max(test["time"].min(), train["time"].min())
#     # Sci-kit survival CPH model
#     estimator = CoxPHSurvivalAnalysis(alpha = penalizer_val).fit(biomarker_vals, merged_structured)
#     survs = estimator.predict_survival_function(x_all_T)
#     survs_predict = estimator.predict(x_all_T)
#     times = np.arange(t_min, t_max)
#     preds = np.asarray([[fn(t) for t in times] for fn in survs])
#     # Integrated brier score in order to account for change to patients over the course of the study
#     # print(survs)
#     # print('preds', preds, 'times', times)
#     score = integrated_brier_score(merged_structured, merged_structured, preds, times)
#     # Concordance index mdoel output
#     c_harrell = concordance_index_censored(merged_structured["Status"], merged_structured["Survival_in_days"], survs_predict)

#     print('concordance index censored scikit:{:.3f}'.format(c_harrell[0]))
#     va_times = np.arange(t_min, t_max, 10)
    
#     # Determining the dynamic AUC values to account for the change to patients over the course of the study
#     cph_auc, cph_mean_auc = cumulative_dynamic_auc(merged_structured, merged_structured, survs_predict, va_times)

#     print("Brier Score {:.3f}".format(score))
#     auc_stuff = (va_times,cph_auc, cph_mean_auc)
#     return score, cph_data, auc_stuff


# In[ ]:


from all_knee_sbl_bootstrap import cph_creator_SBL


# In[ ]:


# CPH Model for SBL Merged data for patients KL Grade 1
cindex_1, brier_1, auc_1 = cph_creator_SBL(kl_1_sbl_merged, True)


# In[ ]:


# CPH Model for SBL Femur data for patients KL Grade 1
cindex_2, brier_2, auc_2 = cph_creator_SBL(kl_1_sbl_femur, True)


# In[ ]:


# CPH Model for SBL Tibia data for patients KL Grade 1
cindex_3, brier_3, auc_3 = cph_creator_SBL(kl_1_sbl_tibia, True)


# In[ ]:


# CPH Model for BML Merged data for patients KL Grade 1
cindex_4, brier_4, auc_4 = cph_creator_SBL(kl_1_bml_merged, False)


# In[ ]:


# CPH Model for BML Femur data for patients KL Grade 1
cindex_5, brier_5, auc_5 = cph_creator_SBL(kl_1_bml_femur, False)


# In[ ]:


# CPH Model for BML Tibia data for patients KL Grade 1
cindex_6, brier_6, auc_6 = cph_creator_SBL(kl_1_bml_tibia, False)


# In[ ]:


# CPH Model for JSN data for patients KL Grade 1
cindex_7, brier_7, auc_7 = cph_creator_SBL(kl_1_JSN_merged, True)


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
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/KL_1_SBL_RAW_CINDEX.pdf', format='pdf', bbox_inches = 'tight')
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
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/KL_1_SBL_RAW_BRIER.pdf', format='pdf', bbox_inches = 'tight')
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
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/KL_1_SBL_RAW_AUC.pdf', format='pdf', bbox_inches = 'tight')
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
table_df_1.to_csv('/home/tsurendr/TABLES/KL_1_SBL_RAW.csv', index=False)


# In[ ]:


# CPH Model for SBL Merged data for patients KL Grade 2
cindex_8, brier_8, auc_8 = cph_creator_SBL(kl_2_sbl_merged, True)


# In[ ]:


# CPH Model for SBL Femur data for patients KL Grade 2
cindex_9, brier_9, auc_9 = cph_creator_SBL(kl_2_sbl_femur, True)


# In[ ]:


# CPH Model for SBL Tibia data for patients KL Grade 2
cindex_10, brier_10, auc_10 = cph_creator_SBL(kl_2_sbl_tibia, True)


# In[ ]:


# CPH Model for BML Merged data for patients KL Grade 2
cindex_11, brier_11, auc_11 = cph_creator_SBL(kl_2_bml_merged, False)


# In[ ]:


# CPH Model for BML Femur data for patients KL Grade 2
cindex_12, brier_12, auc_12 = cph_creator_SBL(kl_2_bml_femur, False)


# In[ ]:


# CPH Model for BML Tibia data for patients KL Grade 2
cindex_13, brier_13, auc_13 = cph_creator_SBL(kl_2_bml_tibia, False)


# In[ ]:


# CPH Model for JSN data for patients KL Grade 2
cindex_14, brier_14, auc_14 = cph_creator_SBL(kl_2_JSN_merged, True)


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
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/KL_2_SBL_RAW_CINDEX.pdf', format='pdf', bbox_inches = 'tight')
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
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/KL_2_SBL_RAW_BRIER.pdf', format='pdf', bbox_inches = 'tight')
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
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/KL_2_SBL_RAW_AUC.pdf', format='pdf', bbox_inches = 'tight')
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
table_df_2.to_csv('/home/tsurendr/TABLES/KL_2_SBL_RAW.csv', index=False)


# In[ ]:


# CPH Model for SBL Merged data for patients KL Grade 3
cindex_15, brier_15, auc_15 = cph_creator_SBL(kl_3_sbl_merged, True)


# In[ ]:


# CPH Model for SBL Femur data for patients KL Grade 3
cindex_16, brier_16, auc_16 = cph_creator_SBL(kl_3_sbl_femur, True)


# In[ ]:


# CPH Model for SBL Tibia data for patients KL Grade 3
cindex_17, brier_17, auc_17 = cph_creator_SBL(kl_3_sbl_tibia, True)


# In[ ]:


# CPH Model for BML Merged data for patients KL Grade 3
cindex_18, brier_18, auc_18 = cph_creator_SBL(kl_3_bml_merged, False)


# In[ ]:


# CPH Model for BML Femur data for patients KL Grade 3
cindex_19, brier_19, auc_19 = cph_creator_SBL(kl_3_bml_femur, False)


# In[ ]:


# CPH Model for BML Tibia data for patients KL Grade 3
cindex_20, brier_20, auc_20 = cph_creator_SBL(kl_3_bml_tibia, False)


# In[ ]:


# CPH Model for JSN data for patients KL Grade 3
cindex_21, brier_21, auc_21 = cph_creator_SBL(kl_3_JSN_merged, True)


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
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/KL_3_SBL_RAW_CINDEX.pdf', format='pdf', bbox_inches = 'tight')
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
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/KL_3_SBL_RAW_BRIER.pdf', format='pdf', bbox_inches = 'tight')
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
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/KL_3_SBL_RAW_AUC.pdf', format='pdf', bbox_inches = 'tight')
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
table_df_3.to_csv('/home/tsurendr/TABLES/KL_3_SBL_RAW.csv', index=False)


# In[ ]:


# CPH Model for SBL Merged data for patients KL Grade 4
cindex_22, brier_22, auc_22 = cph_creator_SBL(kl_4_sbl_merged, True)


# In[ ]:


# CPH Model for SBL Femur data for patients KL Grade 4
cindex_23, brier_23, auc_23 = cph_creator_SBL(kl_4_sbl_femur, True)


# In[ ]:


# CPH Model for SBL Tibia data for patients KL Grade 4
cindex_24, brier_24, auc_24 = cph_creator_SBL(kl_4_sbl_tibia, True)


# In[ ]:


# CPH Model for BML Merged data for patients KL Grade 4
cindex_25, brier_25, auc_25 = cph_creator_SBL(kl_4_bml_merged, False)


# In[ ]:


# CPH Model for BML Femur data for patients KL Grade 4
cindex_26, brier_26, auc_26 = cph_creator_SBL(kl_4_bml_femur, False)


# In[ ]:


# CPH Model for BML Tibia data for patients KL Grade 4
cindex_27, brier_27, auc_27 = cph_creator_SBL(kl_4_bml_tibia, False)


# In[ ]:


# CPH Model for JSN data for patients KL Grade 4
cindex_28, brier_28, auc_28 = cph_creator_SBL(kl_4_JSN_merged, True)


# In[ ]:


mean_value_1 = cindex_22[1]
ci1_lower = cindex_22[0][0]
ci1_upper = cindex_22[0][1]
plt.errorbar(x=[1], y=[mean_value_1], yerr=[[mean_value_1 - ci1_lower], [ci1_upper - mean_value_1]], fmt='o', capsize=5,label='SBL Merged')


mean_value_2 = cindex_23[1]
ci2_lower = cindex_23[0][0]
ci2_upper = cindex_23[0][1]
plt.errorbar(x=[2], y=[mean_value_2], yerr=[[mean_value_2 - ci2_lower], [ci2_upper - mean_value_2]], fmt='o', capsize=5,label='SBL Femur')


mean_value_3 = cindex_24[1]
ci3_lower = cindex_24[0][0]
ci3_upper = cindex_24[0][1]
plt.errorbar(x=[3], y=[mean_value_3], yerr=[[mean_value_3 - ci3_lower], [ci3_upper - mean_value_3]], fmt='o', capsize=5,label='SBL Tibia')

mean_value_4 = cindex_25[1]
ci4_lower = cindex_25[0][0]
ci4_upper = cindex_25[0][1]
plt.errorbar(x=[4], y=[mean_value_4], yerr=[[mean_value_4 - ci4_lower], [ci4_upper - mean_value_4]], fmt='o', capsize=5,label='BML Merged')

mean_value_5 = cindex_26[1]
ci5_lower = cindex_26[0][0]
ci5_upper = cindex_26[0][1]
plt.errorbar(x=[5], y=[mean_value_5], yerr=[[mean_value_5 - ci5_lower], [ci5_upper - mean_value_5]], fmt='o', capsize=5,label='BML Femur')

mean_value_6 = cindex_27[1]
ci6_lower = cindex_27[0][0]
ci6_upper = cindex_27[0][1]
plt.errorbar(x=[6], y=[mean_value_6], yerr=[[mean_value_6 - ci6_lower], [ci6_upper - mean_value_6]], fmt='o', capsize=5,label='BML Tibia')

mean_value_7 = cindex_28[1]
ci7_lower = cindex_28[0][0]
ci7_upper = cindex_28[0][1]
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
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/KL_4_SBL_RAW_CINDEX.pdf', format='pdf', bbox_inches = 'tight')
plt.clf()

# In[ ]:


mean_value_1 = brier_22[1]
ci1_lower = brier_22[0][0]
ci1_upper = brier_22[0][1]
print(ci1_lower, ci1_upper, mean_value_1)
plt.errorbar(x=[1], y=[mean_value_1], yerr=[[mean_value_1 - ci1_lower], [ci1_upper - mean_value_1]], fmt='o', capsize=5,label='SBL Merged')


mean_value_2 = brier_23[1]
ci2_lower = brier_23[0][0]
ci2_upper = brier_23[0][1]
plt.errorbar(x=[2], y=[mean_value_2], yerr=[[mean_value_2 - ci2_lower], [ci2_upper - mean_value_2]], fmt='o', capsize=5,label='SBL Femur')


mean_value_3 = brier_24[1]
ci3_lower = brier_24[0][0]
ci3_upper = brier_24[0][1]
plt.errorbar(x=[3], y=[mean_value_3], yerr=[[mean_value_3 - ci3_lower], [ci3_upper - mean_value_3]], fmt='o', capsize=5,label='SBL Tibia')

mean_value_4 = brier_25[1]
ci4_lower = brier_25[0][0]
ci4_upper = brier_25[0][1]
plt.errorbar(x=[4], y=[mean_value_4], yerr=[[mean_value_4 - ci4_lower], [ci4_upper - mean_value_4]], fmt='o', capsize=5,label='BML Merged')

mean_value_5 = brier_26[1]
ci5_lower = brier_26[0][0]
ci5_upper = brier_26[0][1]
plt.errorbar(x=[5], y=[mean_value_5], yerr=[[mean_value_5 - ci5_lower], [ci5_upper - mean_value_5]], fmt='o', capsize=5,label='BML Femur')

mean_value_6 = brier_27[1]
ci6_lower = brier_27[0][0]
ci6_upper = brier_27[0][1]
plt.errorbar(x=[6], y=[mean_value_6], yerr=[[mean_value_6 - ci6_lower], [ci6_upper - mean_value_6]], fmt='o', capsize=5,label='BML Tibia')

mean_value_7 = brier_28[1]
ci7_lower = brier_28[0][0]
ci7_upper = brier_28[0][1]
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
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/KL_4_SBL_RAW_BRIER.pdf', format='pdf', bbox_inches = 'tight')
plt.clf()

# In[ ]:


mean_value_1 = auc_22[1]
ci1_lower = auc_22[0][0]
ci1_upper = auc_22[0][1]
print(ci1_lower, ci1_upper, mean_value_1)
plt.errorbar(x=[1], y=[mean_value_1], yerr=[[mean_value_1 - ci1_lower], [ci1_upper - mean_value_1]], fmt='o', capsize=5,label='SBL Merged')


mean_value_2 = auc_23[1]
ci2_lower = auc_23[0][0]
ci2_upper = auc_23[0][1]
plt.errorbar(x=[2], y=[mean_value_2], yerr=[[mean_value_2 - ci2_lower], [ci2_upper - mean_value_2]], fmt='o', capsize=5,label='SBL Femur')


mean_value_3 = auc_24[1]
ci3_lower = auc_24[0][0]
ci3_upper = auc_24[0][1]
plt.errorbar(x=[3], y=[mean_value_3], yerr=[[mean_value_3 - ci3_lower], [ci3_upper - mean_value_3]], fmt='o', capsize=5,label='SBL Tibia')

mean_value_4 = auc_25[1]
ci4_lower = auc_25[0][0]
ci4_upper = auc_25[0][1]
plt.errorbar(x=[4], y=[mean_value_4], yerr=[[mean_value_4 - ci4_lower], [ci4_upper - mean_value_4]], fmt='o', capsize=5,label='BML Merged')

mean_value_5 = auc_26[1]
ci5_lower = auc_26[0][0]
ci5_upper = auc_26[0][1]
plt.errorbar(x=[5], y=[mean_value_5], yerr=[[mean_value_5 - ci5_lower], [ci5_upper - mean_value_5]], fmt='o', capsize=5,label='BML Femur')

mean_value_6 = auc_27[1]
ci6_lower = auc_27[0][0]
ci6_upper = auc_27[0][1]
plt.errorbar(x=[6], y=[mean_value_6], yerr=[[mean_value_6 - ci6_lower], [ci6_upper - mean_value_6]], fmt='o', capsize=5,label='BML Tibia')

mean_value_7 = auc_28[1]
ci7_lower = auc_28[0][0]
ci7_upper = auc_28[0][1]
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
plt.savefig('/home/tsurendr/Error_bar_Plots/Raw_SBL/KL_4_SBL_RAW_AUC.pdf', format='pdf', bbox_inches = 'tight')
plt.clf()

# In[ ]:


# Column names
columns = ['Cindex', 'Brier', 'AUC']

# Create an empty DataFrame
table_df_4 = pd.DataFrame(columns=columns)


row_22 = {'Cindex': cindex_22, 'Brier': brier_22, 'AUC': auc_22}
row_23 = {'Cindex': cindex_23, 'Brier': brier_23, 'AUC': auc_23}
row_24 = {'Cindex': cindex_24, 'Brier': brier_24, 'AUC': auc_24}
row_25 = {'Cindex': cindex_25, 'Brier': brier_25, 'AUC': auc_25}
row_26 = {'Cindex': cindex_26, 'Brier': brier_26, 'AUC': auc_26}
row_27 = {'Cindex': cindex_27, 'Brier': brier_27, 'AUC': auc_27}
row_28 = {'Cindex': cindex_28, 'Brier': brier_28, 'AUC': auc_28}
table_df_4 = table_df_4.append(row_22, ignore_index=True)
table_df_4 = table_df_4.append(row_23, ignore_index=True)
table_df_4 = table_df_4.append(row_24, ignore_index=True)
table_df_4 = table_df_4.append(row_25, ignore_index=True)
table_df_4 = table_df_4.append(row_26, ignore_index=True)
table_df_4 = table_df_4.append(row_27, ignore_index=True)
table_df_4 = table_df_4.append(row_28, ignore_index=True)
table_df_4.to_csv('/home/tsurendr/TABLES/KL_4_SBL_RAW.csv', index=False)

