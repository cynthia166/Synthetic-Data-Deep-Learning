
remote = True


if remote: 
    folder = "generated_synthcity_tabular/ARF/ARF GRU/"
    model_path = folder + "deepecho_model2_more_subset_visits.pkl"
    real_data_path = folder + "train_ehr_datasetARF_fixed_v.pkl"
    synthetic_data_gru = folder + "synthetic_results2_more_subset_visits.pkl"
    synthetic_data_arf = folder + "synthetic_ehr_datasetARF_fixed_v.pkl"