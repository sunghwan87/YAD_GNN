###############  Parameter search space for ML models ############### 
import numpy as np
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.anova import anova_lm
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression as PLSR
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.utils.fixes import loguniform  # for param grid

class ParameterSML(object):
    def __init__(self):
        pass
    
    def get(self, model_type, task_type, kernel='linear'):
    #def __call__(self, model_type, task_type, opt_mode, kernel='linear'): # matching proper parameters sets
        if task_type=='classification':
            if model_type=='SVM':
                mdl = SVC(class_weight='balanced')
                param = {
                    'grid': {
                        'C': [0.01, 0.1, 1, 10, 100, 1000],
                        'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'kernel': [kernel]
                    },
                    'rand': {
                        'C': loguniform(1e-2, 1e3),
                        'gamma': loguniform(1e-4, 1e3),
                        'kernel': [kernel]
                    },
                    'bayes': {
                        'C': (1e-2, 1e3, 'log-uniform'),
                        'gamma': (1e-4, 1e3, 'log-uniform'),
                        'kernel': [kernel]
                    },
                }
            elif model_type=='LR':
                mdl = LogisticRegression(class_weight='balanced', solver='liblinear')
                param = {
                    'grid': {
                        'penalty': ['l1', 'l2'],
                        'C': [0.01, 0.1, 1, 10, 100],
                        'intercept_scaling': [0.1, 1, 10],
                    },
                    'bayes': {
                        'penalty': ['l1', 'l2'],
                        'C': (1e-2, 1e2, 'log-uniform'),
                        'intercept_scaling': (1e-1, 1e1, 'log-uniform'),
                    }
                }
            elif model_type=='RF':
                mdl = RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1)
                param = {
                    'grid': {
                        'bootstrap': [True],
                        'max_depth': [50, 80, 100, 150],
                        'max_features': [2, 3, 4, 5],
                        'min_samples_leaf': [3, 4, 5],
                        'min_samples_split': [8, 10, 12],
                        'n_estimators': [100, 200, 300, 1000]
                    },
                }
            elif model_type=='GB':
                mdl = GradientBoostingClassifier()
                param ={ 
                    'grid': { 
                        "loss":['deviance', 'exponential'],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "min_samples_split": np.linspace(0.1, 0.5, 5),
                        "min_samples_leaf": np.linspace(0.1, 0.5, 5),
                        "max_depth":[3,5,8],
                        "max_features":["log2","sqrt"],
                        "criterion": ["friedman_mse"],
                        "subsample":[0.5, 0.8, 1.0],
                        "n_estimators":[10, 50, 100],
                    },
                }
            elif model_type=='XGB':
                mdl = Pipeline([('xgb', xgb.XGBClassifier()),])
                param = {
                    'grid': {
                        'xgb__max_depth': [3,5,8],
                        'xgb__n_estimators': [10, 50, 100],
                        'xgb__subsample': [0.5, 0.8, 1.0],
                        'xgb__learning_rate':  [0.01, 0.1, 0.2],
                        'xgb__colsample_bytree':  [0.3],
                        'xgb__alpha':  [1,10], # L1 regularization on leaf weights
                        'xgb__lambda':  [1,10], # L2 regularization on leaf weights
                    },

                }



        elif task_type=='regression':
            if model_type=='SVM':
                mdl = SVR()
                param = {
                    'grid' : {
                        'C': [0.01, 0.1, 1, 10, 100, 1000],
                        'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'epsilon':[0.05, 0.2, 0.5, 2],
                        'kernel': [kernel]
                    },
                    'rand' : {
                        'C': loguniform(1e-2, 1e3),
                        'gamma': loguniform(1e-4, 1e3),
                        'epsilon':loguniform(0.05, 2),
                        'kernel': [kernel]
                    },
                    'bayes' : {
                        'C': (1e-2, 1e3, 'log-uniform'),
                        'gamma': (1e-4, 1e3, 'log-uniform'),
                        'epsilon':[0.05, 0.2, 0.5, 2],
                        'kernel': [kernel]
                    },
                }
                
            elif model_type == 'PLS':
                mdl = PLSR()
                param = {
                    'grid': {
                        'n_components': np.arange(2,10)
                    },
                }
            elif model_type=='RF':
                mdl = RandomForestRegressor(n_jobs=-1)
                param = {
                    'grid' : {
                        'bootstrap': [True],
                        'max_depth': [80, 90, 100, 110],
                        'max_features': [2, 3],
                        'min_samples_leaf': [3, 4, 5],
                        'min_samples_split': [8, 10, 12],
                        'n_estimators': [100, 200, 300, 1000]
                    },
                }
            elif model_type=='GB':
                mdl = GradientBoostingRegressor()
                param = {
                    'grid': { 
                        "loss":['ls', 'lad', 'huber'],
                        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                        "min_samples_split": np.linspace(0.1, 0.5, 12),
                        "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                        "max_depth":[3,5,8],
                        "max_features":["log2","sqrt"],
                        "criterion": ["friedman_mse"],
                        "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                        "n_estimators":[10, 50]  
                    }
                }
            elif model_type=='LR': 
                mdl = LogisticRegression()
                param = {
                    'grid':{
                        "C_range_lin": np.logspace(-20, 10, 10, base=2)
                    },
                }
        return mdl, param
