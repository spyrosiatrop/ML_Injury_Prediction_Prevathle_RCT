# MODELS - SHAP
import shap
import experiment
import pandas as pd
import numpy as np

models = experiment.open_pickle(r"model_objects\f_all72_models_240513_1649.pickle")
rr= experiment.find_best_models(r"model_objects\f_all72_models_240513_1649.pickle", 'roc_auc')

ids = rr['id'].unique()
best_models = models.loc[models['id'].isin(ids)]

# IMPORT DATA
df = pd.read_csv(r"./prevathle_full_pre.csv", index_col=0) # ./Data
df = df.drop(['base_event', 'base_sex'], axis=1)


raw = ['wk_prev', 'wk_sleep', 'wk_sport_hr', 'wk_sport_int', 'wk_status', 'wk_train_hr', 'wk_train_int', 'wk_comp_hr', 'wk_comp_int']
# column categories
col_dict={'base_cols' : ['base_age', 'base_bodyweight', 'base_event_end', 'base_height', 'base_injury', 'base_intervention', 'base_sex_f', 'base_sport_hr', 'base_train_hr'],
            'outcome_cols': [c for c in df.columns if 'outcome' in c],
            'wk2_cols' : [c+'_-2' for c in raw],
            'wk1_cols' : [c+'_-1' for c in raw],
            'wk0_cols'  : ['wk_prev', 'wk_sleep', 'wk_sport_hr', 'wk_sport_int', 'wk_status', 'wk_train_hr', 'wk_train_int', 'wk_comp_hr', 'wk_comp_int'],
            'roll_cols' : [c for c in df.columns if 'roll' in c],
            'cum_cols' : [c for c in df.columns if 'cum' in c and 'ai_' not in c],
            'vel_cols' : [c for c in df.columns if '_d01' in c],
            'acc_cols' : [c for c in df.columns if 'acc02' in c],
            'stand_cols' : ['code', 'wk_num']}

all_cols = col_dict['base_cols'] + col_dict['wk0_cols'] + col_dict['wk2_cols']+ col_dict['wk1_cols'] + col_dict['roll_cols'] + col_dict['vel_cols'] + col_dict['acc_cols']

# healthy for a window of 3 consecutive weeks
healthy_2 = df[(df['outcome_prai']==0) & (df['outcome_prai_-1']==0) & (df['outcome_prai_-2']==0)].dropna()


trained_models = list(best_models['trained_object'])
X= healthy_2[all_cols]
explainers = [shap.Explainer(tm.predict, X ) for tm in trained_models]

# Extract and store SHAP VALUES
models = ['LOG', 'LIN-SVM', 'RBF-SVM', 'DT', 'RF', 'ADA', 'XGB']
shap_values_list = []
for i, exp in enumerate(explainers):
    shap_values = exp.shap_values(X)
    shap_values_list.append(shap_values)

# calculate the average shap_values (a list of 7 lists (1 per algorithm) of 72 elements (1 per feature))
average_shap_values = [np.abs(sv).mean(axis=0) for sv in shap_values_list]
feature_names = X.columns  # Assuming feature names are in X.columns
feature_importances_df = pd.DataFrame()
for i, asv in enumerate(average_shap_values):
    importance_df = pd.DataFrame({'model': models[i],'feature': feature_names, 'importance': asv})
    feature_importances_df = pd.concat([feature_importances_df, importance_df], axis=0)

# find the rank of each feature in each algorithm
feature_importances_df['rank'] = feature_importances_df.groupby('model')['importance'].rank(method='min', ascending=False)
feature_importances_df= feature_importances_df.drop('importance', axis=1)
# find the median and range of ranks of each feature
medians = feature_importances_df.groupby('feature')['rank'].median().rename('Median').apply(lambda x: int(x))
mins = feature_importances_df.groupby('feature')['rank'].min().rename('Min').apply(lambda x: int(x))
maxs = feature_importances_df.groupby('feature')['rank'].max().rename('Max').apply(lambda x: int(x))

feature_importances_df= feature_importances_df.pivot(index='feature', columns='model', values='rank').apply(lambda x: x.astype(int))
feature_importances_df = pd.concat([feature_importances_df, medians, mins, maxs], axis=1).sort_values('Median')