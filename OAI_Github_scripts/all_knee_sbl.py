def cph_creator_SBL(dataframe, title):
    
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
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV, KFold
    from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
    import warnings
    
    #initalize lists for columns that might have no variance
    bad_cols = []
    
    #columns not to check if variance is bad
    col_not_check = ["time", "right_tkr"]
 
    # checking dataframe to make sure there is high enough variance in each data column
    events = dataframe["right_tkr"].astype(bool)
    for i in dataframe.columns:
        if i not in col_not_check:
            if dataframe.loc[~events, i].var() < 0.001 or dataframe.loc[events, i].var() < 0.001:
                dataframe = dataframe.drop(labels=i, axis=1)
                bad_cols.append(i)
    print("bad cols: ", bad_cols)
    brier_scores_SBL = []


    # Prepping dataframe for CPH input
    # Creating Structured Array for Sci-Kit Survival models, while keeping traditional dataframe for lifelines package. 
    merged_tkr_time = dataframe[["right_tkr", "time"]].copy()
    merged_tkr_time["right_tkr"] = merged_tkr_time["right_tkr"].astype(bool)
    merged_tkr_time = merged_tkr_time.to_numpy()
    aux_test = [(e1, e2) for e1, e2 in merged_tkr_time]
    merged_structured = np.array(
    aux_test, dtype=[("Status", "?"), ("Survival_in_days", "<f8")]
    )
    males_merged_sbl = dataframe.copy().drop("time", axis=1)  # keep as a dataframe
    biomarker_vals = males_merged_sbl.copy().drop("right_tkr", axis=1)  # keep as a dataframe

    # Used for min and max timeframe for brier score computation
    train, test = train_test_split(dataframe, test_size=0.2, random_state=120)
    
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
    print("penalizer_val: ", penalizer_val)
    # Fitting Cph model to data
    cph = CoxPHFitter(penalizer=penalizer_val, l1_ratio = 0.0)
    cph.fit(dataframe, "time", event_col="right_tkr", show_progress=True, robust=True)

    # Predicting outputs based on model
    cph_data = cph.predict_survival_function(dataframe)


    x_all_T = OneHotEncoder().fit_transform(biomarker_vals)
    
    # Selecting Timeframe in which to observe event occurence
    t_max = min(test["time"].max(), train["time"].max())
    t_min = max(test["time"].min(), train["time"].min())
    # Sci-kit survival CPH model
    estimator = CoxPHSurvivalAnalysis(alpha = penalizer_val).fit(biomarker_vals, merged_structured)
    survs = estimator.predict_survival_function(x_all_T)
    survs_predict = estimator.predict(x_all_T)
    times = np.arange(t_min, t_max)
    preds = np.asarray([[fn(t) for t in times] for fn in survs])
    # Integrated brier score in order to account for change to patients over the course of the study
    score = integrated_brier_score(merged_structured, merged_structured, preds, times)
    # Concordance index mdoel output
    c_harrell = concordance_index_censored(merged_structured["Status"], merged_structured["Survival_in_days"], survs_predict)

    print('concordance index censored scikit:{:.3f}'.format(c_harrell[0]))
    va_times = np.arange(t_min, t_max, 10)
    
    # Determining the dynamic AUC values to account for the change to patients over the course of the study
    cph_auc, cph_mean_auc = cumulative_dynamic_auc(merged_structured, merged_structured, survs_predict, va_times)
    brier_scores_SBL.append(score)
    print("Brier Score {:.3f}".format(score))
   
    auc_stuff = (va_times,cph_auc, cph_mean_auc)
    return brier_scores_SBL, cph_data, auc_stuff
