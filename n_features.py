# IMPORT MODULES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT FUNCTIONS
import experiment

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

# N of FEATURES EXPERIMENT
features = healthy_2[all_cols]

random_results= pd.DataFrame()

for r in range(10):             
    random_features = list(features.sample(72)['feature']) # random sequence of features
    for i in range(72):
        _, df, _, _ = experiment.my_internal_validation(X= healthy_2[random_features[:i+1]], 
                                y= healthy_2['outcome_prai_+1'], n_bootstrap= 10,
                                scalers= ['mms'], classifiers= ['log', 'dt', 'lin-svm', 'rbf-svm', 'rf', 'ada', 'xgb'], 
                                scorers= ['roc_auc', 'f1', 'recall', 'precision'], 
                                param_C= [1], param_class_weight= [{0:1, 1:v} for v in [64]], 
                                param_n_estimators= [75], param_tree_depth= [9],
                                report_title= 'random_features_progression', 
                                exp_path= None)
        corrected = df[(df['sample']=='corrected')]
        means = corrected.groupby(['pipe', 'id', 'metric'])['value'].mean().reset_index()
        best_means= means.loc[means.groupby(['pipe', 'metric'])['value'].idxmax()]
        best_means['n_features'] = i+1
        best_means['round'] = r
        
        random_results = pd.concat([random_results, best_means], axis=0)
        
        

# VISUAL
nrandom = random_results
fig, axes = plt.subplots(2, 2, figsize= (8, 8), sharex=True, width_ratios=[1.618, 1], height_ratios=[1.618, 1])
axes = axes.reshape(-1)
def epv(n):
    return 149/n
repv = epv
for m, metric in enumerate(['roc_auc', 'precision', 'recall', 'f1']):
    axes[m].set_ylabel(f'{metric}'.upper(), fontsize=10)
    if m==0:
        axes[m].set_ylabel('AUROC')

    sns.lineplot(nrandom[(nrandom['metric']==metric)], x='n_features', y='value', hue='pipe', ax=axes[m])
    mylabels = ['ADA', 'DT', 'LIN-SVM', 'LOG', 'RBF-SVM', 'RF', 'XGB']
    h, l = axes[m].get_legend_handles_labels()
    
    if m==0 or m==1:
        secax = axes[m].secondary_xaxis('top', functions=(epv, repv))
        secax.set_xlabel('Events per variable')
        secax.set_xticks([24, 6, 4, 3, 2.4, 2.1])
    
    if m==1:
        axes[m].legend(handles = h, labels = mylabels, frameon= True, framealpha=0.5)
    else:
        axes[m].legend().remove()
    axes[m].set_xlabel('Number of features')


    axes[m].set_xlim(-1,73)

fig.tight_layout()




