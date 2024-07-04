# IMPORT MODULES £££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
# data-wise
import pandas as pd
import numpy as np
import pickle
# # visualising
# import matplotlib.pyplot as plt
# import seaborn as sns
# ML household
from sklearn.preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv  
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV,  StratifiedGroupKFold, ParameterGrid
from sklearn.base import clone
from sklearn.utils import resample
from sklearn.calibration import calibration_curve, CalibrationDisplay, CalibratedClassifierCV
# ML classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
# ML metrics
from sklearn.metrics import *
# for dating exports
from datetime import datetime
#statistics
from  scipy.optimize import curve_fit 
import time



# DEFINE FUNCTIONS £££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
# 1. experiment 2 pickle
def experiment2pickle(report, report_title, path):
    #create title
    from datetime import datetime
    now = datetime.now().strftime('%y%m%d_%H%M')
    title = f'{report_title}_{now}'
    
    with open(fr'{path}/{title}.pickle', 'wb') as file:
        pickle.dump(report, file, protocol= pickle.HIGHEST_PROTOCOL)
        
    return None

# 2. create pipelines
def create_pipelines(scalers_dict, scalers, classifiers_dict, classifiers):
    pipes_dict = {}
    for s in [sc for sc in scalers_dict.keys() if sc in scalers]:
        for c in [cl for cl in classifiers_dict.keys() if cl in classifiers]:
            ppl = Pipeline([
                ('scaler', scalers_dict[s]),
                ('model', classifiers_dict[c])])
            # name the pipeline
            pipes_dict[s+'_&_'+c] = ppl    
    return pipes_dict

# 3. Creates custom dictionary for grid search
def my_models(pipes_dict, parameters_dict):
    models_dict = {}
    
    for p in pipes_dict:
        pipe = pipes_dict[p]
        param_grid = ParameterGrid(parameters_dict[p.split('_&_')[-1]])
        model_list = []
        
        for params in param_grid:
            model = clone(pipe) # otherwise params will be rewritten
            model = model.set_params(**params)
            model_list.append(model)            
            
        models_dict[p] = model_list
        
    return models_dict


# 4. create a list of bootstrap samples
def my_bootstrap(X, y, iterations, size, stratify=None):
    b_samples = []
    for i in range(iterations):
        X_bs, y_bs = resample(X, y, replace=True, n_samples = size, stratify= stratify)
        b_samples.append([X_bs, y_bs])
        
    return b_samples

# 5. fit the linear function to get calibration slope
def cal_slope(y, x):

    def linear(x, a, b):
        return a*x + b
    
    # fit the curve linking the probablities to the actual results (it's linear beacuse the prob is already logistic transformed)
    linear, _ = curve_fit(linear, x, y)
    linear_slope = linear[0]   

    return linear_slope

# 6. bootstrap function
def my_internal_validation(X, y, n_bootstrap, scalers, classifiers, scorers,
                           param_C, param_class_weight, param_n_estimators, param_tree_depth,
                           report_title, exp_path):
    # TIME KEEPING
    start  = time.time()
    until = start
    
    # PREPROCESSING
    scalers_dict = {'mms': MinMaxScaler(), 'std': StandardScaler(), 'pol2': PolynomialFeatures(degree=2)}
    
    # MODELS
    classifiers_dict = {'log': LogisticRegression(max_iter= 1000, n_jobs= -1), 'lin-svm': SVC(kernel='linear'), 
                        'rbf-svm': SVC(kernel='rbf'), 'dt': DecisionTreeClassifier(), 
                        'sgd': SGDClassifier(n_jobs=-1), 'rf': RandomForestClassifier(n_jobs=-1),
                        'ada': AdaBoostClassifier(), 'xgb': GradientBoostingClassifier(),
                        'nb': GaussianNB(), 'knn':  KNeighborsClassifier(n_jobs=-1)}
    
    # SCORERS dictionary
    score_functions= {'accuracy': accuracy_score,
          'precision': precision_score,
          'recall': recall_score,
          'f1': f1_score,
          'roc_auc': roc_auc_score, 
          'f2': fbeta_score, 
          'brier': brier_score_loss}
    
    scorers_dict = {'accuracy': make_scorer(accuracy_score),
          'precision': make_scorer(precision_score),
          'recall': make_scorer(recall_score),
          'f1': make_scorer(f1_score),
          'roc_auc': make_scorer(roc_auc_score),
          'f2': make_scorer(fbeta_score, beta=2), 
          'brier': make_scorer(brier_score_loss)}
    
    # PARAMETERS
    parameters_dict= {
        'log':{'model__C': param_C,
            'model__class_weight': param_class_weight},
        'lin-svm':{'model__C': param_C,
            'model__gamma': ['scale'],
            'model__class_weight': param_class_weight},
        'rbf-svm':{'model__C': param_C, 
            'model__gamma': ['scale'], 
            'model__class_weight': param_class_weight},
        'sgd': {'model__loss': ['modified_huber'],
            'model__alpha': [0.00003, 0.0001, 0.0003], 
            'model__class_weight': param_class_weight},
        'dt':{'model__criterion': ['gini'],
            'model__splitter': ['best'],
            'model__class_weight': param_class_weight,
            'model__max_depth': param_tree_depth},
        'rf':{'model__n_estimators': param_n_estimators,
            'model__criterion': ['gini'],
            'model__class_weight': param_class_weight, 
            'model__max_depth': param_tree_depth},
        'ada':{'model__estimator': [DecisionTreeClassifier(max_depth=5)], 
               'model__n_estimators': param_n_estimators},
        'xgb':{'model__learning_rate': [0.03, 0.1, 0.3], 
               'model__n_estimators': param_n_estimators},
        'nb':{},
        'knn':{'model__n_neighbors': [3,5,7],
            'model__weights': ['uniform', 'distance'],
            'model__algorithm': ['auto']}
        }

    # PIPELINES for each scaler and classifier combination
    pipes_dict= create_pipelines(scalers_dict, scalers, classifiers_dict, classifiers)
    
    # create all possible MODELS
    models_dict = my_models(pipes_dict, parameters_dict)
        
    # 1. CREATE BOOTSTRAP SAMPLES
    bs_pairs = my_bootstrap(X, y, iterations=n_bootstrap, size = len(y), stratify= y)
    
    # 2. START THE EXPERIMENT
    prob_results= pd.DataFrame()
    score_results = pd.DataFrame()
    models_df = pd.DataFrame()
    # add the true outcomes
    prob_results = pd.concat([prob_results, 
                                pd.DataFrame([None, None, None, 'true', -1, list(y)]).T],
                                axis=0, ignore_index=True)
    
    # 3. LOOP over pipes
    for i, pipe in enumerate(models_dict):
        # 4. LOOP over different models (parameters) of the same pipe
        for m, model in enumerate(models_dict[pipe]):
            # unique ID for each model
            id = pipe + f"_{m}"
            # get the model's params
            details = [id, pipe, m]
            params= model.get_params()
            details.append(params)
            
            # 5. TRAIN on original data
            o_model = clone(model).fit(X, y)
            # 6. add the trained model as object (SHAP)
            details.append(o_model) 
            # 7. APPEND the details line to the dataframe
            models_df = pd.concat([models_df, pd.DataFrame(details).T], axis=0, ignore_index=True)
            
            # 8a. PREDICT on original data
            y_orig_pred = o_model.predict(X)
            # 8b. PREDICT probabilities (for svm get the decion function output and pass it through logistic function)
            y_orig_prob = o_model.decision_function(X) if 'svm' in pipe else o_model.predict_proba(X)
            # logistic function for svm or keep the positive class probability for all other
            y_orig_prob = [1/(1 + np.exp(-pr)) for pr in y_orig_prob] if 'svm' in pipe else [pr[1] for pr in y_orig_prob]
            # 8c round them in 2 digits for memory save
            y_orig_prob = [round(p, 2) for p in y_orig_prob]
            # 8d. APPEND the raw probabilities for later
            # the original set is the same to be tested on so self and test are identical
            prob_results = pd.concat([prob_results, 
                                pd.DataFrame([id, pipe, m, 'original', -1, y_orig_prob]).T],
                                axis=0, ignore_index=True)
            # 9. SCORES
            # 9a. calculate metrics
            apparent = {}
            for metric in scorers:
                if metric == 'brier':
                    value = score_functions[metric](y, y_orig_prob)
                elif metric == 'f2':
                    value = score_functions[metric](y, y_orig_pred, beta = 2)
                else:
                    value = score_functions[metric](y, y_orig_pred)             
                # concat to results
                apparent[metric] = value
                score_results = pd.concat([score_results, 
                                    pd.DataFrame([id, pipe, m, 'original', -1, metric, round(value, 2)]).T],
                                    axis=0, ignore_index=True)
                
            # 9b. calibration-in-the-large
            apparent_citl = np.mean(y_orig_prob) / np.mean(y)
            score_results = pd.concat([score_results, 
                                    pd.DataFrame([id, pipe, m, 'original', -1, 'cal_in_the_large', round(apparent_citl, 2)]).T],
                                    axis=0, ignore_index=True)
            
            # 9c. calculate calibration slope
            apparent_slope = cal_slope(y, y_orig_prob)
            score_results = pd.concat([score_results, 
                                pd.DataFrame([id, pipe, m, 'original', -1, 'cal_slope', round(apparent_slope, 2)]).T],
                                axis=0, ignore_index=True)            
        
            ### 10. repeat the BOOTSTRAPPING n_bootstrap times
            for i, bs in enumerate(bs_pairs):
                # 10a. train on each bootstrap sample
                b_model = clone(model).fit(bs[0], bs[1])
                
                # 10b. PREDICT on TEST/ORIGINAL data
                y_test_pred = b_model.predict(X)
                # predict probabilities (for svm get the decion function output and pass it through logistic function)
                y_test_prob = b_model.decision_function(X) if 'svm' in pipe else b_model.predict_proba(X)
                # logistic function for svm or keep the positive class probability for all other
                y_test_prob = [1/(1 + np.exp(-pr)) for pr in y_test_prob] if 'svm' in pipe else [pr[1] for pr in y_test_prob]
                # round in 2 digits
                y_test_prob = [round(p, 2) for p in y_test_prob]
                                
                # 10c. predict on SELF set
                y_self_pred = b_model.predict(bs[0])
                # predict probabilities (for svm get the decion function output and pass it through logistic function)
                y_self_prob = b_model.decision_function(bs[0]) if 'svm' in pipe else b_model.predict_proba(bs[0])
                # logistic function for svm or keep the positive class probability for all other
                y_self_prob = [1/(1 + np.exp(-pr)) for pr in y_self_prob] if 'svm' in pipe else [pr[1] for pr in y_self_prob]
                # round in 2 digits
                y_self_prob = [round(p, 2) for p in y_self_prob]
                
                # 10d. APPEND the raw probabilities for later
                prob_results = pd.concat([prob_results, 
                                     pd.DataFrame([id, pipe, m, 'test', i, y_test_prob]).T],
                                    axis=0, ignore_index=True)
                                ### CALCULATE BOOTSTRAP PROBS & SCORES

                # 10e. calculate SCORES on SELF set
                optimism_self = {}
                for metric in scorers:
                    if metric == 'brier':
                        value = score_functions[metric](bs[1], y_self_prob)
                    elif metric == 'f2':
                        value = score_functions[metric](bs[1], y_self_pred, beta = 2)
                    else:
                        value = score_functions[metric](bs[1], y_self_pred)   
                    #  assign value for optimism calculation 
                    optimism_self[metric] = value
                                            

                # 10f. CALIBRATON in the large
                self_citl = np.mean(y_self_prob) / np.mean(bs[1])
                # calculate calibration slope and intercept
                self_slope = cal_slope(bs[1], y_self_prob)

                # 10g. calculate SCORES on TEST set
                optimism_test = {}
                for metric in scorers:
                    if metric == 'brier':
                        value = score_functions[metric](y, y_test_prob)
                    elif metric == 'f2':
                        value = score_functions[metric](y, y_test_pred, beta = 2)
                    else:
                        value = score_functions[metric](y, y_test_pred)  
                    #  assign value for optimism caculation
                    optimism_test[metric] = value    
                
                # 10h. CALIBRATION in the large
                test_citl = np.mean(y_test_prob) / np.mean(y)
                # calculate calibration slope
                test_slope = cal_slope(y, y_test_prob) 
       
                # 11. calculate OPTIMISM
                for metric in scorers:
                    optimism = optimism_self[metric] - optimism_test[metric]
                    corrected = apparent[metric] - optimism
                    score_results = pd.concat([score_results, 
                                        pd.DataFrame([id, pipe, m, 'corrected', i, metric, round(corrected, 2)]).T],
                                        axis=0, ignore_index=True)
                    
                # 11a. citl optimism
                citl_optimism =  test_citl/self_citl
                corrected_citl = apparent_citl * citl_optimism
                score_results = pd.concat([score_results, 
                                            pd.DataFrame([id, pipe, m, 'corrected', i, 'cal_in_the_large', round(corrected_citl, 2)]).T],
                                            axis=0, ignore_index=True)
                # 11b. calculate slope optimism
                slope_optimism  = test_slope / self_slope
                corrected_slope = apparent_slope * slope_optimism
                score_results = pd.concat([score_results, 
                                        pd.DataFrame([id, pipe, m, 'corrected', i, 'cal_slope', round(corrected_slope, 2)]).T],
                                        axis=0, ignore_index=True)
           
        #  TIME KEEPING
        lap = until
        until = time.time()
        print(f"{pipe} with {len(models_dict[pipe])} models completed in {round(until - lap)} seconds")

                
    # 12. change NAMES
            # change column names in the score_results
    score_results.columns= ['id', 'pipe','model', 'sample', 'number', 'metric', 'value']
    score_results.reset_index(drop=True)
    
    # change names
        # change column names in the prob_results
    prob_results.columns= ['id', 'pipe','model', 'sample', 'number', 'probs']
    prob_results.reset_index(drop=True)
    
    # change names
        # change column names in the models_df
    models_df.columns= ['id', 'pipe','model', 'params', 'trained_object']
    models_df.reset_index(drop=True)
                
                
    ### 13. FINAL OUTPUT ###
    report= {'title': [report_title],
            'datetime': [datetime.now()],
            'features_list': [X.columns],
            'features_n': [len(X.columns)],
            'label': [y.name],
            'n_bootstrap': [n_bootstrap],
            'scalers' : [scalers],
            'classifiers': [classifiers],
            'scorers' : [scorers], 
            'param_C' : [param_C],
            'param_class_weight' : [param_class_weight],
            'param_n_estimators': [param_n_estimators],
            'param_tree_depth': [param_tree_depth],
            'report_title': [report_title],
            'exp_path': [exp_path]}   
        
    # 14. EXPORT experiment results as csv
    experiment2pickle(report, report_title, exp_path)
    experiment2pickle(models_df, f'{report_title}_models', exp_path)
    experiment2pickle(score_results, f'{report_title}_scores', exp_path)
    experiment2pickle(prob_results, f'{report_title}_probs', exp_path)


    return report, score_results, prob_results, models_df

# END OF PREPARATORY CODE £££££££££££££££££££££££££££££££££££££££££££££££££££££

###############################################################################
# # # # START OF MODELDEVELOPMENT, VALIDATION, AND RESULTS EXTRACTION # # # # #
###############################################################################

# IMPORT DATA £££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
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

# EXPERIMENTS £££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££

n_btstrap = 200

# 1. baseline features (9)
my_internal_validation(X= healthy_2[col_dict['base_cols']], 
                               y= healthy_2['outcome_prai_+1'], n_bootstrap= n_btstrap,
                               scalers= ['mms'], classifiers= ['log', 'dt', 'lin-svm', 'rbf-svm', 'rf', 'ada', 'xgb'], 
                               scorers= ['roc_auc', 'recall', 'precision', 'f1', 'f2','accuracy', 'brier'], 
                               param_C= [0.7, 1, 1.3], param_class_weight= [{0:1, 1:v} for v in [1, 8, 64]]+['balanced'], 
                               param_n_estimators= [50, 75, 100], param_tree_depth= [5, 9, 15],
                               report_title= 'f_base', 
                               exp_path= "./")

# 2. monitoring current-week features (9)
my_internal_validation(X= healthy_2[col_dict['wk0_cols']], 
                               y= healthy_2['outcome_prai_+1'], n_bootstrap= n_btstrap,
                               scalers= ['mms'], classifiers= ['log', 'dt', 'lin-svm', 'rbf-svm', 'rf', 'ada', 'xgb'], 
                               scorers= ['roc_auc', 'recall', 'precision', 'f1', 'f2','accuracy', 'brier'], 
                               param_C= [0.7, 1, 1.3], param_class_weight= [{0:1, 1:v} for v in [1, 8, 64]]+['balanced'], 
                               param_n_estimators= [50, 75, 100], param_tree_depth= [5, 9, 15],
                               report_title= 'f_wk0', 
                               exp_path= "./")

# 3. monitoring current-week and previous-week features (18)
my_internal_validation(X= healthy_2[col_dict['wk0_cols']+ col_dict['wk1_cols']], 
                               y= healthy_2['outcome_prai_+1'], n_bootstrap= n_btstrap,
                               scalers= ['mms'], classifiers= ['log', 'dt', 'lin-svm', 'rbf-svm', 'rf', 'ada', 'xgb'], 
                               scorers= ['roc_auc', 'recall', 'precision', 'f1', 'f2','accuracy', 'brier'], 
                               param_C= [0.7, 1, 1.3], param_class_weight= [{0:1, 1:v} for v in [1, 8, 64]]+['balanced'], 
                               param_n_estimators= [50, 75, 100], param_tree_depth= [5, 9, 15],
                               report_title= 'f_wk01', 
                               exp_path= "./")

# 4. monitoring current-week and 2 previous-weeks features (27)
my_internal_validation(X= healthy_2[col_dict['wk0_cols']+ col_dict['wk1_cols']+ col_dict['wk2_cols']], 
                               y= healthy_2['outcome_prai_+1'], n_bootstrap= n_btstrap,
                               scalers= ['mms'], classifiers= ['log', 'dt', 'lin-svm', 'rbf-svm', 'rf', 'ada', 'xgb'], 
                               scorers= ['roc_auc', 'recall', 'precision', 'f1', 'f2','accuracy', 'brier'], 
                               param_C= [0.7, 1, 1.3], param_class_weight= [{0:1, 1:v} for v in [1, 8, 64]]+['balanced'], 
                               param_n_estimators= [50, 75, 100], param_tree_depth= [5, 9, 15],
                               report_title= 'f_wk012', 
                               exp_path= "./")

# 5. rolling features (18)
my_internal_validation(X= healthy_2[col_dict['roll_cols']], 
                               y= healthy_2['outcome_prai_+1'], n_bootstrap= n_btstrap,
                               scalers= ['mms'], classifiers= ['log', 'dt', 'lin-svm', 'rbf-svm', 'rf', 'ada', 'xgb'], 
                               scorers= ['roc_auc', 'recall', 'precision', 'f1', 'f2','accuracy', 'brier'], 
                               param_C= [0.7, 1, 1.3], param_class_weight= [{0:1, 1:v} for v in [1, 8, 64]]+['balanced'], 
                               param_n_estimators= [50, 75, 100], param_tree_depth= [5, 9, 15],
                               report_title= 'f_roll', 
                               exp_path= "./")

# 6. differentiating features (18)
my_internal_validation(X= healthy_2[col_dict['vel_cols']+ col_dict['acc_cols']], 
                               y= healthy_2['outcome_prai_+1'], n_bootstrap= n_btstrap,
                               scalers= ['mms'], classifiers= ['log', 'dt', 'lin-svm', 'rbf-svm', 'rf', 'ada', 'xgb'], 
                               scorers= ['roc_auc', 'recall', 'precision', 'f1', 'f2','accuracy', 'brier'], 
                               param_C= [0.7, 1, 1.3], param_class_weight= [{0:1, 1:v} for v in [1, 8, 64]]+['balanced'], 
                               param_n_estimators= [50, 75, 100], param_tree_depth= [5, 9, 15],
                               report_title= 'f_diff', 
                               exp_path= "./")


# 7. Baseline and current-week monitoring features (18)
my_internal_validation(X= healthy_2[col_dict['base_cols']+ col_dict['wk0_cols']], 
                               y= healthy_2['outcome_prai_+1'], n_bootstrap= n_btstrap,
                               scalers= ['mms'], classifiers= ['log', 'dt', 'lin-svm', 'rbf-svm', 'rf', 'ada', 'xgb'], 
                               scorers= ['roc_auc', 'recall', 'precision', 'f1', 'f2','accuracy', 'brier'], 
                               param_C= [0.7, 1, 1.3], param_class_weight= [{0:1, 1:v} for v in [1, 8, 64]]+['balanced'], 
                               param_n_estimators= [50, 75, 100], param_tree_depth= [5, 9, 15],
                               report_title= 'f_all', 
                               exp_path= "./")

# 8. All features (72)
my_internal_validation(X= healthy_2[col_dict['base_cols'] + col_dict['wk0_cols'] + col_dict['wk2_cols']+ col_dict['wk1_cols'] + col_dict['roll_cols'] + col_dict['vel_cols'] + col_dict['acc_cols']], 
                               y= healthy_2['outcome_prai_+1'], n_bootstrap= 200,
                               scalers= ['mms'], classifiers= ['log', 'dt', 'lin-svm', 'rbf-svm', 'rf', 'ada', 'xgb'], 
                               scorers= ['roc_auc', 'recall', 'precision', 'f1', 'f2','accuracy', 'brier'], 
                               param_C= [0.7, 1, 1.3], param_class_weight= [{0:1, 1:v} for v in [1, 8, 64]]+['balanced'], 
                               param_n_estimators= [50, 75, 100], param_tree_depth= [5, 9, 15],
                               report_title= 'f_all72', 
                               exp_path= "./")



