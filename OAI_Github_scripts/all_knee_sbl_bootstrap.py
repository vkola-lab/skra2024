
#import libraries from lifelines and Sci-kit survival
from sksurv.metrics import (
concordance_index_censored,
concordance_index_ipcw,
cumulative_dynamic_auc,
integrated_brier_score,
)
import pandas as pd
from sksurv.preprocessing import OneHotEncoder
from scipy import interpolate
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from lifelines.utils.sklearn_adapter import sklearn_adapter
import lifelines
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
import warnings
from lifelines.statistics import proportional_hazard_test
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer,RobustScaler
import warnings
import random
warnings.filterwarnings("ignore")


def cph_creator_SBL(dataframe, sbl_bool):

 
    # checking dataframe to make sure there is high enough variance in each data column
    ################################################################################################
    # #initalize lists for columns that might have no variance
    bad_cols = []
    vars = []
    #columns not to check if variance is bad
    col_not_check = ["time", "right_tkr"]
    # events = dataframe["right_tkr"].astype(bool)
    # for i in dataframe.columns:
    #     if i not in col_not_check:
    #         tempy1 = dataframe.loc[~events, i].var()
    #         tempy2 = dataframe.loc[events, i].var()
    #         vars.append(tempy1)
    #         vars.append(tempy2)
    #         # print(i, tempy1, tempy2)
    #         if dataframe.loc[~events, i].var() < 0.001 or dataframe.loc[events, i].var() < 0.001:
    #             # tempy1 = dataframe.loc[~events, i].var()
    #             # tempy2 = dataframe.loc[events, i].var()
    #             # print(i, tempy1, tempy2)
    #             dataframe = dataframe.drop(labels=i, axis=1)
    #             bad_cols.append(i)
    # print("bad cols: ", bad_cols)
    # print('variance quartiles: ', np.percentile(vars, [0, 25, 50, 75, 100]))
    ###############################################################################################
    # print('test1')
    
    time_col = list(dataframe['time'].copy())
    right_tkr_col = list(dataframe['right_tkr'].copy())
    # print(len(time_col), len(right_tkr_col))
    dataframe = dataframe.drop(['time', 'right_tkr'], axis=1)
    scaler = MinMaxScaler()
    dataframe = scaler.fit_transform(dataframe)
    dataframe = pd.DataFrame(dataframe)
    dataframe["right_tkr"] = right_tkr_col
    dataframe["time"] = time_col
    # print(len(dataframe))
    # print(dataframe['right_tkr'])
    # Prepping dataframe for CPH input
    # Creating Structured Array for Sci-Kit Survival models, while keeping traditional dataframe for lifelines package. 
    # print('df_cols length:', len(df_cols))
    # merged_tkr_time = dataframe[["right_tkr", "time"]].copy()
    # merged_tkr_time["right_tkr"] = merged_tkr_time["right_tkr"].astype(bool)
    # merged_tkr_time = merged_tkr_time.to_numpy()
    # aux_test = [(e1, e2) for e1, e2 in merged_tkr_time]
    # merged_structured = np.array(
    # aux_test, dtype=[("Status", "?"), ("Survival_in_days", "<f8")]
    # )
    # males_merged_sbl = dataframe.copy().drop("time", axis=1)  # keep as a dataframe
    # biomarker_vals = males_merged_sbl.copy().drop("right_tkr", axis=1)  # keep as a dataframe

    # # Used for min and max timeframe for brier score computation
    # train, test = train_test_split(dataframe, test_size=0.2, random_state=120)
    # print(dataframe.isnull().sum())
    X_all = dataframe.copy().drop("time", axis=1)
    Y_all = dataframe.copy().pop("time")

    base_class = sklearn_adapter(CoxPHFitter, event_col="right_tkr")
    wf = base_class()

    #Using GridSearchCV to perform hyperparameter tuning
    from sklearn.model_selection import GridSearchCV

    clf = GridSearchCV(
        wf,
        {
            "penalizer": 10.0 ** np.arange(-2, 3),
        },
        cv=4,
    )
    clf.fit(X_all, Y_all)

    # Picking best penalizer value for CPH model
    penalizer_val = clf.best_params_.get("penalizer")
    print("old penalizer_val: ", penalizer_val)
    # Fitting Cph model to data

    cph = CoxPHFitter( penalizer=penalizer_val)
    #Testing Strata
    # cph.fit(dataframe_strata_femur, "time", event_col="right_tkr",show_progress=True, robust=True, strata=['F8_femur_strata'])
    cph.fit(dataframe, "time", event_col="right_tkr",   show_progress=False, robust=True )
    # cph.fit(strata_df, "time", event_col="right_tkr", strata = strata_columns,   show_progress=True, robust=True )
    # cph.print_summary(decimals=3)
  

    print('old dataframe length: ', len(dataframe.columns))
    # Checks Proportional Hazards Assumption
    print('SBL_bool', sbl_bool)

    if sbl_bool == True:
        results = proportional_hazard_test(cph, dataframe, time_transform='rank')
        for variable in cph.params_.index.intersection(cph.params_.index):
            minumum_observed_p_value = results.summary.loc[variable, "p"].min()
            minumum_observed_p_value = np.round(minumum_observed_p_value,3)
            # print(variable,minumum_observed_p_value)
            # print(pairings)
            if minumum_observed_p_value < 0.05:
                # print(pairings)
                dataframe = dataframe.drop(columns=[variable])
        print('new dataframe length: ', len(dataframe.columns))
 
        
        # X_all = dataframe.copy().drop("time", axis=1)
        # Y_all = dataframe.copy().pop("time")

        # base_class = sklearn_adapter(CoxPHFitter, event_col="right_tkr")
        # wf = base_class()

        # #Using GridSearchCV to perform hyperparameter tuning
        # from sklearn.model_selection import GridSearchCV

        # clf = GridSearchCV(
        #     wf,
        #     {
        #         "penalizer": 10.0 ** np.arange(-2, 3),
        #     },
        #     cv=4,
        # )
        # clf.fit(X_all, Y_all)

        # # Picking best penalizer value for CPH model
        # penalizer_val = clf.best_params_.get("penalizer")
        # print("new penalizer_val: ", penalizer_val) 
        # cph = CoxPHFitter( penalizer=penalizer_val)
        cph.fit(dataframe, "time", event_col="right_tkr",   show_progress=False, robust=True )
    # cph.print_summary(decimals=3)
    # print('#################################### checkpoint 3')
    cph_data = cph.predict_survival_function(dataframe)
    
    
    


    
    num_samples = 1000
    skips = 0
    print('number of bootstrap samples: ', num_samples)
    bootstrap_means_cindex = np.zeros(num_samples)
    bootstrap_means_brier = np.zeros(num_samples)
    bootstrap_means_auc = np.zeros(num_samples)
    for i in range(num_samples):
        val = round(random.uniform(0.1, 1.000000001),2)
        # print('val: ', val)
        dataframe_copy = dataframe.copy()
        dataframe_copy = dataframe_copy.groupby('right_tkr', group_keys=False).apply(lambda x: x.sample(frac=val,replace=True)) 
        # print('new dataframe len: ', len(dataframe_copy), 'expected length: ', 1974*val)

        # print('biomarker_vals: ', biomarker_vals)
        # biomarker_vals = biomarker_vals.reset_index(drop=True, inplace=True)
        # Used for min and max timeframe for brier score computation
        # print('dataframe',dataframe_copy["right_tkr"])
        X_train, X_test,ytrain,ytest = train_test_split(dataframe_copy, dataframe_copy["right_tkr"], test_size=0.2, random_state=120,stratify=dataframe_copy["right_tkr"])
        
        
        merged_tkr_time_train = X_train[["right_tkr", "time"]].copy()
        merged_tkr_time_train["right_tkr"] = merged_tkr_time_train["right_tkr"].astype(bool)
        merged_tkr_time_train = merged_tkr_time_train.to_numpy()
        aux_train = [(e1, e2) for e1, e2 in merged_tkr_time_train]
        y_train = np.array(
        aux_train, dtype=[("Status", bool), ("Survival_in_days", "<f8")]
        )
        
        
        merged_tkr_time_test = X_test[["right_tkr", "time"]].copy()
        merged_tkr_time_test["right_tkr"] = merged_tkr_time_test["right_tkr"].astype(bool)
        merged_tkr_time_test = merged_tkr_time_test.to_numpy()
        aux_test = [(e1, e2) for e1, e2 in merged_tkr_time_test]
        y_test = np.array(
        aux_test, dtype=[("Status", "?"), ("Survival_in_days", "<f8")]
        )       
        
        X_train = X_train.drop(['time', 'right_tkr'], axis=1)
        X_test = X_test.drop(['time', 'right_tkr'], axis=1)
        
        # print('y_train', y_train)
        # print('y_test', y_test)
        
        
        # print('merged_structured: ', merged_structured)

        # train_indices = X_train.index
        # test_indices = X_test.index
        # # Convert the indices to a list if needed
        # train_indices_list = train_indices.tolist()
        # test_indices_list = test_indices.tolist()
        # y_train, y_test = merged_structured[train_indices_list], merged_structured[test_indices_list]
        # print(y_train)
        # x_all_T = OneHotEncoder().fit_transform(biomarker_vals)
        
        X_train = OneHotEncoder().fit_transform(X_train)
        X_test = OneHotEncoder().fit_transform(X_test)
        
        # Selecting Timeframe in which to observe event occurence
        t_max = min(y_train["Survival_in_days"].max(), y_test["Survival_in_days"].max())
        t_min = max(y_train["Survival_in_days"].min(), y_test["Survival_in_days"].min())
        # Sci-kit survival CPH model
        estimator = CoxPHSurvivalAnalysis(alpha = penalizer_val).fit(X_train, y_train)

        survs = estimator.predict_survival_function(X_test)
        survs_predict = estimator.predict(X_test)
        times = np.arange(t_min, t_max)
        preds = np.asarray([[fn(t) for t in times] for fn in survs])
        # Integrated brier score in order to account for change to patients over the course of the study
        # print(survs)
        # print('preds', preds, 'times', times)
        # print(y_test)
        # print('np all test:', (np.any(y_test["Status"]) == True))
        if np.any(y_test["Status"]) == True:
            score = integrated_brier_score(y_test, y_test, preds, times)
            bootstrap_means_brier[i] = (score)
            # Concordance index mdoel output
            c_harrell = concordance_index_censored(y_test["Status"], y_test["Survival_in_days"], survs_predict)
            bootstrap_means_cindex[i] = (c_harrell[0])
            # print('everything was not false')


            # print('concordance index censored scikit:{:.3f}'.format(c_harrell[0]))
            va_times = np.arange(t_min, t_max, 10)
            
            # Determining the dynamic AUC values to account for the change to patients over the course of the study
            cph_auc, cph_mean_auc = cumulative_dynamic_auc(y_test, y_test, survs_predict, va_times)
            # print('cph_mean_auc: ', cph_mean_auc)
            bootstrap_means_auc[i] = (cph_mean_auc)
            # print("Brier Score {:.3f}".format(score))
            auc_stuff = (va_times,cph_auc, cph_mean_auc)
        else:
            skips += 1
            # print('Everything was false')
        
    # average_cindex = sum(cindex_scores) / n_splits
    # average_brier = sum(brier_scores) / n_splits
    # average_mean_auc = sum(mean_auc_scores) / n_splits
    # print("Average Concordance Index:", round(average_cindex,3))
    # print("Concordance Index stdev:", round(np.std(cindex_scores),3))    
    # print("Average Brier Score:", round(average_brier,3))
    # print("Brier Score stdev:", round(np.std(brier_scores),3))
    # print("Average Mean AUC:", round(average_mean_auc,3))
    # print("Mean AUC stdev:", round(np.std(mean_auc_scores),3))
    
    cleanedList_cindex = [x for x in bootstrap_means_cindex if str(x) != 'nan']
    confidence_interval_cindex = np.percentile(cleanedList_cindex, [2.5, 97.5])

    cindex_vals = (confidence_interval_cindex,np.mean(cleanedList_cindex))
    
    # print(bootstrap_means_cindex)
    # print(cleanedList_cindex)
    print("95% Confidence interval for C-index: ", confidence_interval_cindex, ' Average C-Index: ', np.mean(cleanedList_cindex))
  
    cleanedList_brier = [x for x in bootstrap_means_brier if str(x) != 'nan'] 
    confidence_interval_brier = np.percentile(cleanedList_brier, [2.5, 97.5])
    brier_vals = (confidence_interval_brier,np.mean(cleanedList_brier))

    print("95% Confidence interval for Brier Score: ", confidence_interval_brier, ' Average Brier Score: ', np.mean(cleanedList_brier))

   
    # print(bootstrap_means_auc)
    cleanedList_auc = [x for x in bootstrap_means_auc if str(x) != 'nan']
    confidence_interval_auc = np.percentile(cleanedList_auc, [2.5, 97.5])
    auc_vals = (confidence_interval_auc,np.mean(cleanedList_auc))

    print("95% Confidence interval for Mean AUC: ", confidence_interval_auc, ' Average Mean AUC: ', np.mean(cleanedList_auc)) 
    
    
    
    # num_samples = 1000
 
    # bootstrap_means_cindex = np.zeros(num_samples)
    # bootstrap_means_brier = np.zeros(num_samples)
    # bootstrap_means_auc = np.zeros(num_samples)
    
    # # Perform bootstrap sampling
    # for i in range(num_samples):
    #     bootstrap_sample = np.random.choice(cindex_scores, size=len(cindex_scores), replace=True)
    #     bootstrap_mean = np.mean(bootstrap_sample)
    #     bootstrap_means_cindex[i] = bootstrap_mean
    
    # confidence_interval_cindex = np.percentile(bootstrap_means_cindex, [2.5, 97.5])
    
    # print("95% Confidence interval for C-index: ", confidence_interval_cindex, ' Average C-Index: ', np.mean(bootstrap_means_cindex))
        
    
    #     # Perform bootstrap sampling
    # for i in range(num_samples):
    #     bootstrap_sample = np.random.choice(brier_scores, size=len(brier_scores), replace=True)
    #     bootstrap_mean = np.mean(bootstrap_sample)
    #     bootstrap_means_brier[i] = bootstrap_mean
    
    # confidence_interval_brier = np.percentile(bootstrap_means_brier, [2.5, 97.5])
    
    # print("95% Confidence interval for Brier Score: ", confidence_interval_brier, ' Average Brier Score: ', np.mean(bootstrap_means_brier))
    
    
    #     # Perform bootstrap sampling
    # for i in range(num_samples):
    #     bootstrap_sample = np.random.choice(cindex_scores, size=len(cindex_scores), replace=True)
    #     bootstrap_mean = np.mean(bootstrap_sample)
    #     bootstrap_means_auc[i] = bootstrap_mean
    
    # confidence_interval_auc = np.percentile(bootstrap_means_auc, [2.5, 97.5])
    
    # print("95% Confidence interval for Mean AUC: ", confidence_interval_auc, ' Average Mean AUC: ', np.mean(bootstrap_means_auc))
    
    return cindex_vals, brier_vals, auc_vals


def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X[:, j : j + 1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores
