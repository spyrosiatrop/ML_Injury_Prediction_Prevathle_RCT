# IMPORT MODULES ########################################################################################
import numpy as np
import pandas as pd

# IMPORT DATA ###########################################################################################
# original dataset from the RCT study
df= pd.read_csv(r"original_dataset_RCT_study.csv", decimal= ',') # df.size (6435, 30)

# CLEANING ##############################################################################################

## 1. wrong code label fixed
df['code']= df['code'].replace('y', 'prevathle117')

## 2. baseline sex should be M,F,nan (replace 'M ', 'F ')
df = df.replace({'F ':'F','M ': 'M'})

## 3. change week_date from str type to datetime
df['week_date'] = pd.to_datetime(df['week_date'], format= '%d/%m/%Y') # (will be probably dropped later though)

## 4. create a different column for age than baseline age
df['week_age'] = df['baseline_age'] # ((will be probably dropped later though))

## 5. keep only the first age of each athlete as baseline_age
df.loc[df['week_number']!=1, 'baseline_age'] = np.nan

## 6. create a numerical variable for injury history (0-3)
df['baseline_historyinjury_num'] = df['baseline_historyinjury'].replace({'NON, Aucune blessure ni problème physique la saison dernière': 0, 
                                                                         "OUI, mais participation réduite à l'entrainement ou en compétition, à cause d'une blessure/problème physique":2, 
                                                                         "OUI, mais participation complète à l'entrainement ou en compétition, malgré une blessure/problème physique":1, 
                                                                         "OUI, mais aucune participation possible à l'entrainement ou en compétition, à cause d'une blessure/problème physique":3})
df['base_icpr']= df['baseline_historyinjury_num'].replace({1:0, 2:1, 3:1})

## 7. drop columns not needed
df = df.drop(['baseline_historyinjury', 'week_date', 'week_totalathletics', 
              'week_trainingwl', 'week_competitionwl', 'week_sportwl', 'week_age'], axis=1)

## 8. forward fill baseline values
for c in df.columns:
    if 'base' in c:
        df[c].ffill(inplace=True)
        
## 9. sex in numeric (male=0, female=1)
df['base_sex_f'] = df['baseline_sex'].replace({'M': 0, 'F': 1})

## 10. discipline in binary (power: 0, endurance: 1)
df['base_event_end'] = df['baseline_discipline'].replace({'Haies': 0, 'Course sur route': 1,
                                                         'Trail': 1, 'Epreuves combinées': 0,
                                                         'Demi-fond et Fond (piste)': 1, 'Sauts': 0, 
                                                         'Lancers': 0, 'Marche Athlétique': 1, 'Sprints': 0})
# 11. group intervemtion: 1, group control:0
df['baseline_randomised'] = df['baseline_randomised'].replace({2:0})

# RENAMING COLUMNS #########################################################################################
df = df.rename({'baseline_sex': 'base_sex', 'baseline_age': 'base_age', 'baseline_height': 'base_height',
       'baseline_bodymass': 'base_bodyweight', 'baseline_discipline': 'base_event', 'baseline_training': 'base_train_hr',
       'baseline_trainingoutathletics': 'base_sport_hr', 'baseline_randomised': 'base_intervention', 'week_number': 'wk_num',
       'week_trainingh': 'wk_train_hr', 'week_trainingi': 'wk_train_int', 'week_competitionh': 'wk_comp_hr',
       'week_competitioni': 'wk_comp_int', 'week_sporth': 'wk_sport_hr', 'week_sporti': 'wk_sport_int', 'week_prevention': 'wk_prev',
       'week_fatigue': 'wk_status', 'week_sleep': 'wk_sleep', 'week_illness': 'wk_illness', 'week_prnai': 'wk_nonathletics_prai',
       'outcome1_prai': 'outcome_prai', 'outcome2_allai_0-3': 'outcome_allai', 'outcome3_prai-hamstring': 'outcome_ham',
       'baseline_historyinjury_num': 'base_injury'}, axis=1)

# TIME INVOLVEMENT ##########################################################################################

## 1. for every feature and outcome get the previous 3 weeks in the same line and for outcomes get next 2 weeks
cols = [c for c in list(df.columns) if ((('wk_' in c) or ('outcome' in c)) and (c!= 'wk_num'))]
for c in cols:
    df[c+'_-1'] = df[c].shift(1)
    df[c+'_-2'] = df[c].shift(2)
    df[c+'_-3'] = df[c].shift(3)
    # None for the early weeks
    df.loc[df['wk_num']==1, [c+'_-1', c+'_-2', c+'_-3']] = None
    df.loc[df['wk_num']==2, [c+'_-2', c+'_-3']] = None
    df.loc[df['wk_num']==3, [c+'_-3']] = None
    
    ## 2. add the future injury outcome
    if 'outcome' in c:
        df[c+'_+1'] = df[c].shift(-1)
        df[c+'_+2'] = df[c].shift(-2)
        df[c+'_+3'] = df[c].shift(-3)
        # None for the last weeks
        df.loc[df['wk_num']==39, [c+'_+1', c+'_+2', c+'_+3']] = None
        df.loc[df['wk_num']==38, [c+'_+2', c+'_+3']] = None
        df.loc[df['wk_num']==37, [c+'_+3']] = None

        
        
# FEATURE ENGINEERING ######################################################################################
## 1. Velocities (v= dx/dt), Accelerations (a= dv/dt)

# only for monitoring variables (wk in col) and avoid x_-1_-1 (col+'_-1' in res_df.columns)
cols = [c for c in df.columns  if 'wk' in c and c+'_-1' in df.columns and 'illness' not in c and 'prai' not in c]

for col in cols:
    ### calculate Delta between 2 successive weeks for each feature
    ### eg. wk_feature_d01 = wk_feature- wk_feature_-1
    df[col+'_d01'] = df[col] - df[col+ '_-1']
    df[col+'_d12'] = df[col+'_-1'] - df[col+ '_-2']
    df[col+'_d23'] = df[col+'_-2'] - df[col+ '_-3']
    ## 2. calculate Delta of Delta between 3 successive weeks for each feature (acceleration Delta(Dx/dt))
    df[col+'_acc02'] = df[col+'_d01'] - df[col+'_d12']
    df[col+'_acc13'] = df[col+'_d12'] - df[col+'_d23']
    
## 3. cumulative and rolling mean and std of each feature by athlete each week
for col in cols:
    df[col+'_cum_mean'] = df.groupby('code')[col].expanding().mean().reset_index('code')[col]
    df[col+'_cum_std'] = df.groupby('code')[col].expanding().std().reset_index('code')[col]
    df[col+'_roll3_mean'] = df.groupby('code')[col].rolling(window= 3).mean().reset_index('code')[col]
    df[col+'_roll3_std'] = df.groupby('code')[col].rolling(window= 3).std().reset_index('code')[col]

# THE FOLLOWING (4, 5, 6, 7) ARE NOT NECESSARY FOR THIS STUDY
## 4. past outcomes as features
## 4a. "burden" variable = mean of all weeks with outcome_prai=1 and outcome_alli=1
df['wk_prai_cum_mean'] = df.groupby('code')['outcome_prai'].expanding().mean().reset_index('code')['outcome_prai']
df['wk_allai_cum_mean'] = df.groupby('code')['outcome_allai'].expanding().mean().reset_index('code')['outcome_allai']


## 4b. outcome_season = binary 0: no injury over the season, 1: at least 1 injury over the season
### (only for grouping purposes not to be used as feature beacuse of future info leakage)
df = pd.merge(df, df.groupby('code')['outcome_prai'].max().rename('outcome_season'), on='code')

## 4c. injury-free streak (weeks until last injury) 
### (from https://predictivehacks.com/count-the-consecutive-events-in-python/)
df['wk_no_prai_streak'] = df['outcome_prai'].groupby((df['outcome_prai'] != df.groupby(['code'])['outcome_prai'].shift()).cumsum()).cumcount() + 1
### diregard streaks of injuries
df.loc[df['outcome_prai']==1, 'wk_no_prai_streak'] = 0

## 4d. new injury =1, same as previous week=0
df['outcome_prai_new'] = df['outcome_prai'] - df['outcome_prai_-1']
df.loc[df['outcome_prai_-1'].isna(), 'outcome_prai_new'] = df['outcome_prai']
df.loc[df['outcome_prai_new']==-1, 'outcome_prai_new'] = 0

## 4e. count new injuries so far
df['wk_prai_num'] = df.groupby('code')['outcome_prai_new'].expanding().sum().reset_index('code')['outcome_prai_new']


## 5. total volume (train+comp) and (train+comp+sport)
df['wk_athletics_hr'] = df['wk_train_hr'] + df['wk_comp_hr']
df['wk_total_hr'] = df['wk_athletics_hr'] + df['wk_sport_hr']

## 6. Exposure
df['wk_train_exp_hr'] = df.groupby('code')['wk_train_hr'].cumsum()
df['wk_athletics_exp_hr'] = df.groupby('code')['wk_athletics_hr'].cumsum()
df['wk_total_exp_hr'] = df.groupby('code')['wk_total_hr'].cumsum()

# 7. EXPOSURE hrs / week
df['wk_athletics_exp_avg'] = df['wk_athletics_exp_hr'].divide(df['wk_num'])


# COLUMN ORGANISATION #####################################################################################
## reordering columns
df = df[df.columns.sort_values()]

# EXPORT CSV ##############################################################################################
df.to_csv(r"./prevathle_full_pre.csv", sep=',', decimal='.')