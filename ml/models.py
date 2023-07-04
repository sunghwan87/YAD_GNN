###############  ML competing models ############### 
import logging
import torch
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn import metrics

from imblearn.over_sampling import SMOTE
from collections import Counter
#from nested_cv import NestedCV
from ml.parameters import ParameterSML
from ml.utils import *
from config import base_dir

# Standard ML models
class ModelSML(object):
    def __init__(self, exp_name=None, model_type="svm", task_type="classification", opt_mode="bayes"):
        self.prefix = "[ModelSML]"
        self.exp_name = exp_name
        self.logger = logging.getLogger(exp_name)
        self.model_type = model_type
        self.task_type = task_type 
        self.opt_mode = opt_mode
        self.model = None
        
    def load_model(self, model_path):
        self.model = torch.load(model_path)
        return self.model

    def resample(self, X, y):
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        self.logger.info(f"{self.prefix} Resampling done: Original dataset {Counter(y)} to resampled {Counter(y_res)}")
        return X_res, y_res

    def train(self, X, y, cv=None, verbosity=1):
        self.logger.info(f'{self.prefix} ... Parameter optimization for {self.model_type}-{self.task_type} by {self.opt_mode}')
        start_time = time.time()
        n_iter = 50
        if cv is None: cv = 5
        model, param = ParameterSML().get(model_type=self.model_type, task_type=self.task_type)
        if self.opt_mode == 'grid':  # Grid search
            model_search = GridSearchCV(model, param[self.opt_mode], refit=True, verbose=verbosity, cv=cv, n_jobs=-1)
        elif self.opt_mode == 'random':  # random search
            model_search = RandomizedSearchCV(model, param[self.opt_mode], refit=True, verbose=verbosity, cv=cv, n_iter=n_iter, n_jobs=-1)
        elif self.opt_mode == 'bayes':  # Bayes search
            model_search = BayesSearchCV(model, param[self.opt_mode], refit=True, verbose=verbosity, cv=cv, n_iter=n_iter, n_jobs=-1)

        model_search.fit(X, y)
        self.trained_model = model_search.best_estimator_

        elapsed_time = time.time() - start_time
        self.training_time = elapsed_time
        if verbosity>0:
            self.logger.info(model_search.best_estimator_)
            self.logger.info(f"{self.prefix} ... Training time: {elapsed_time:.01f} sec ")

        return model_search.best_estimator_

    def test(self, X, y, verbosity=0):
        model = self.trained_model
        y_pred = model.predict(X)
        if self.task_type=="classification":
            n_class = len(set(y)) 
            self.confusion_matrix =  metrics.confusion_matrix(y, y_pred)
            self.classif_report =  metrics.classification_report(y, y_pred)
            res_metrics = dict()
            res_metrics['roc']  = metrics.roc_auc_score(y, y_pred) if n_class==2 else roc_auc_score_multiclass(y, y_pred, average="macro")
            res_metrics['avp'] = metrics.average_precision_score(y, y_pred, average='macro')
            res_metrics['f1'] = metrics.f1_score(y, y_pred, average='macro')
            res_metrics['acc'] =  metrics.accuracy_score(y, y_pred)
            res_metrics['pr'] = metrics.precision_score(y, y_pred, average='macro')
            res_metrics['rc'] = metrics.recall_score(y, y_pred, average='macro')
            
        elif self.task_type=="regression":
            res_metrics['score']  = model.score(X, y)
            res_metrics['r2'] = metrics.r2_score(y, y_pred)
            res_metrics['MAE'] = metrics.mean_absolute_error(y, y_pred)
            res_metrics['MSE'] = metrics.mean_squared_error(y, y_pred)
            self.logger.info(res_metrics)
        self.metrics = res_metrics
        self.logger.info(f"{self.prefix} testing model is done.")

        return self.metrics