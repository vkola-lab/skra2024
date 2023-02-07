def add_time(oai_SBL_KL_WOMAC_merge, knee):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    # Defining fixed number of days per month in order to estimate most recent followup for patients who did not get a tkr
    num_days_in_month = 30
    if "time" in oai_SBL_KL_WOMAC_merge.columns:

        oai_SBL_KL_WOMAC_merge = oai_SBL_KL_WOMAC_merge.drop("time", axis=1)

    # Estimating number of days till last followup for patients who did not get a tkr
    time_val = pd.Series([], dtype="float64")
    for i in range(len(oai_SBL_KL_WOMAC_merge)):
        if oai_SBL_KL_WOMAC_merge["right_tkr"][i] == 0:
            if oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 0.0:
                time_val[i] = 0 * num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 1.0:
                time_val[i] = 12 * num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 2.0:
                time_val[i] = 24 * num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 3.0:
                time_val[i] = 36 * num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 4.0:
                time_val[i] = 48 * num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 5.0:
                time_val[i] = 60 * num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 6.0:
                time_val[i] = 72 * num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 7.0:
                time_val[i] = 84 * num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 8.0:
                time_val[i] = 96 * num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 9.0:
                time_val[i] = 108 * num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 10.0:
                time_val[i] = 120 * num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 11.0:
                time_val[i] = 132 * num_days_in_month

        # For patients who did have a tkr use the date
        elif oai_SBL_KL_WOMAC_merge["right_tkr"][i] == 1:
            # Use information for left knee
            if knee == "left":
                time_val[i] = oai_SBL_KL_WOMAC_merge["V99ELKDAYS"][i]
            # Use information for right knee
            elif knee == "right":
                time_val[i] = oai_SBL_KL_WOMAC_merge["V99ERKDAYS"][i]
    # Add time to event information to dataframe
    oai_SBL_KL_WOMAC_merge.insert(2, "time", time_val)


    # Create a Kaplan-Meier curve in order to check and make sure there is no major errors within the tkr data
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=oai_SBL_KL_WOMAC_merge["time"],
        event_observed=oai_SBL_KL_WOMAC_merge["right_tkr"],
        label=knee,
    )
    
    # Plotting Kaplan-Meier curve visualization
    kmf.plot_survival_function(
    )
    plt.ylabel("Survival Probability")
    plt.title("kmf plot - km_estimate")
    return oai_SBL_KL_WOMAC_merge
