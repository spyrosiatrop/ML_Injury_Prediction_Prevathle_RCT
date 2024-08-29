# IMPORT MODULES
import pandas as pd
import numpy as np
import os
from statsmodels.stats.weightstats import ttest_ind

# IMPORT DATA
scores = [] # data import here
# CONCAT THE BEST MODELS OF EACH EXPERIMENT TO COMPARE BETWEEN EXPERIMENTS
best_df = pd.DataFrame()
for s in scores:
    experiment = s.split('f_', maxsplit=1)[1].split('_scores')[0]
    best = experiment.find_best_models(rf'{s}', 'roc_auc')
    best['experiment'] = experiment
    best_df = pd.concat([best_df, best], axis=0)
    
# renaming
best_df['pipe'] = best_df['pipe'].apply(lambda x: x.split('_&_')[1])
best_df = best_df.rename({'experiment': 'Features', 'pipe': 'Model'}, axis=1)
best_df['Features'].replace({'wk012': '3wk monitoring (27)', 'all72': 'All (72)', 'base': 'Baseline (9)', 'diff': 'Differential (18)', 'wk01': '2wk monitoring (18)', 'base_wk0': 'Baseline & 1wk monitoring (18)',
                             'roll': 'Rolling (18)', 'wk0': '1wk monitoring (9)'}, inplace=True)

# T-TESTS between bootstrapped models
experiments = best_df['Features'].unique()
pipes = best_df['Model'].unique()

ttests = pd.DataFrame()
for p in pipes:
    for i, e1 in enumerate(experiments):
        x1 = best_df[(best_df['metric']=='roc_auc') & (best_df['Model']==p)& (best_df['Features']==e1)]['value']
        x1_low, x1_high  = round(np.percentile(x1, 2.5), 2), round(np.percentile(x1, 97.5), 2)
        for e2 in experiments[:]:
            x2 = best_df[(best_df['metric']=='roc_auc') & (best_df['Model']==p)& (best_df['Features']==e2)]['value']
            x2_low, x2_high  = round(np.percentile(x2, 2.5), 2), round(np.percentile(x2, 97.5), 2)

            tstat, pval, freedom = ttest_ind(x1, x2, usevar= 'unequal')
            
            ttests = pd.concat([ttests, 
                                pd.DataFrame([p, e1, round(x1.mean(), 2), round(x1.std(), 2), x1_low, x1_high, e2, round(x2.mean(), 2), round(x2.std(), 2), x2_low, x2_high, tstat, pval, freedom]).T],
                                axis=0)
            
ttests = ttests.rename({0: 'model', 1: 'features1', 2: 'mean1', 3: 'std1', 4: 'low_ci1', 5: 'high_ci1', 
                    6: 'features2', 7: 'mean2', 8: 'std2', 9: 'low_ci2', 10: 'high_ci2', 11: 't_stat', 12: 'p_val', 13:'freedom'}, axis=1).reset_index(drop=True)
ttests=  ttests.loc[ttests['features1'] != ttests['features2']]



# COUNT TABLE TO COMPARE FETAURES WITHOUT THE DIMENSION OF MODELS
#3.7 is the value of t-statistic for which p<0.000255
smaller = ttests[ttests['t_stat']<-3.7].groupby(['features1', 'features2'])['t_stat'].count().rename('<')
larger = ttests[ttests['t_stat']>3.7].groupby(['features1', 'features2'])['t_stat'].count().rename('>')
nodiff = ttests[ttests['p_val']> 0.000255].groupby(['features1', 'features2'])['t_stat'].count().rename('=')
count_table = pd.concat([smaller, nodiff, larger], axis=1).fillna(0)
count_table = count_table.unstack().reorder_levels([1,0], axis=1).sort_index(axis=1)
count_table 
