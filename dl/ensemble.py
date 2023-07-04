# The `EnsembleVotingModel` will take our custom LightningModule and                        
# several checkpoint_paths.                                                              
import os.path as osp
import sys
from typing import Any, Dict, List, Optional, Type
import numpy as np
import torch
from pytorch_lightning import seed_everything, LightningModule, Trainer
from torchmetrics import AUROC, Accuracy, AveragePrecision, Precision, Recall, Specificity, F1Score
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError, ExplainedVariance
from sklearn.metrics import confusion_matrix

from config import base_dir
if not base_dir in sys.path: sys.path.append(base_dir)

class EnsembleVotingModel(LightningModule):

    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str], num_classes, task='classification'):
        super().__init__()
        self.prefix = "[EnsembleVotingModel]"
        # Create `num_folds` models with their associated fold weights
        self.task = task
        self.num_classes = num_classes 
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        
        if self.task=='classification':
            self.metrics = ['roc', 'avp', 'acc', 'pr', 'rc', 'f1'] 
            if self.num_classes==2: self.metrics += ['sp']
            self.metric_avg_method = "macro"  #if self.num_classes>2 else 'micro'
            self.aucroc_score = AUROC(num_classes = self.num_classes, average=self.metric_avg_method)
            self.avg_precision_score = AveragePrecision(num_classes = self.num_classes, average=self.metric_avg_method)
            self.accuracy_score = Accuracy(num_classes = self.num_classes, average=self.metric_avg_method, multiclass=True)            
            self.precision_score = Precision(num_classes = self.num_classes, average=self.metric_avg_method, multiclass=True)
            self.recall_score = Recall(num_classes = self.num_classes, average=self.metric_avg_method, multiclass=True)
            self.f1_score = F1Score(num_classes = self.num_classes, average=self.metric_avg_method, multiclass=True)
            self.specificity_score = Specificity(num_classes = self.num_classes, average=self.metric_avg_method, multiclass=True)       
        elif self.task=='regression':
            self.metrics = ['rmse', 'mae', 'r2', 'ev']
            self.rmse = MeanSquaredError(squared=False)
            self.mae = MeanAbsoluteError()
            self.r2 = R2Score(num_outputs=1, adjusted=0)
            self.ev = ExplainedVariance()
        print(f"{self.prefix} initialized.")

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # Compute the averaged predictions over the `num_folds` models.
        outputs = torch.stack([ m.forward(batch)[3] for m in self.models]).mean(0)
        labels = self.models[0].get_label(batch).to(outputs.device)
        loss = self.models[0].target_loss_module(outputs, labels.float())
        return {"loss": loss, "labels": labels, "outputs":outputs}

    def test_epoch_end(self, batch_outputs):
        avg_loss = torch.concat([x[f"loss"].unsqueeze(0) for x in batch_outputs]).mean()
        labels = torch.concat([x[f"labels"] for x in batch_outputs]).squeeze()
        outputs = torch.concat([x[f"outputs"] for x in batch_outputs]).squeeze()
        self.log(f"test_loss", avg_loss)
        for metric in self.metrics:
            metric_value = self.calculate_metric(metric, labels, outputs)            
            self.log(f"test_{metric}", metric_value)
        cm = self.calculate_metric('cm', labels, outputs)  # confusion_matrix
        print(f"{self.prefix} test epoch ended.")
        print(cm)   
 
    def predict_step(self, batch, batch_idx):
        outputs = torch.stack([ m.forward(batch)[3] for m in self.models], dim=-1) # batch x fold
        embedding =  torch.stack([ m.forward(batch)[-1] for m in self.models], dim=-1)  # batch x z_dim x fold
        labels = self.models[0].get_label(batch).to(outputs.device)
        subject_id = self.models[0].get_subject_id(batch)
        if self.task=='graph_classification':
            pred_class, pred_probs = self.models[0].get_pred(outputs, self.num_classes)
            return {"labels": labels, "pred_class": pred_class, "pred_probs": pred_probs, "outputs":outputs, "embedding":embedding, "subject_id":subject_id}
        elif self.task=='graph_regression': 
            return {"labels": labels, "outputs":outputs, "embedding":embedding, "subject_id":subject_id}
    
    def on_predict_epoch_end(self, results):
        batch_results = results[0]
        subject_id = np.concatenate([x[f"subject_id"] for x in batch_results ]).squeeze()
        labels = torch.concat([x[f"labels"] for x in batch_results]).squeeze()
        outputs = torch.concat([x[f"outputs"] for x in batch_results]).squeeze()  # subjects x fold
        embedding = torch.concat([x[f"embedding"] for x in batch_results]).squeeze()  # subjects x z_dim x fold
        if 'pred_class' in batch_results[0].keys(): pred_class = torch.concat([x[f"pred_class"] for x in batch_results]).squeeze()
        if 'pred_probs' in batch_results[0].keys(): pred_probs = torch.concat([x[f"pred_probs"] for x in batch_results]).squeeze()
        return {"labels": labels, "outputs":outputs, "embedding":embedding, "subject_id":subject_id}

    def calculate_metric(self, metric, labels, outputs):
        if self.task=='classification':
            labels = labels.long()            
        
            if self.num_classes==2: # binary classification: output_dim = 1
                pred_probs = torch.stack([1-torch.sigmoid(outputs), torch.sigmoid(outputs)], dim=-1)
                pred_class = (outputs > 0).type_as(outputs)
                
            elif self.num_classes>2: # self.num_classes=2
                pred_probs = torch.softmax(outputs, dim=-1)
                pred_class = outputs.argmax(dim=-1)            
                

        #print(pred_probs, pred_class, labels)
        if metric=='roc': return self.aucroc_score(pred_probs, labels) 
        if metric=='avp': return self.avg_precision_score(pred_probs, labels)  
        if metric=='acc': return self.accuracy_score(pred_probs, labels) 
        if metric=='pr':  return self.precision_score(pred_class, labels) 
        if metric=='rc':  return self.recall_score(pred_class, labels) 
        if metric=='f1':  return self.f1_score(pred_class, labels) 
        if metric=='sp':  return self.specificity_score(pred_class, labels)
        if metric=='cm':  return confusion_matrix(labels.detach().cpu().numpy(), pred_class.detach().cpu().numpy())
        if metric=='rmse': return self.rmse(outputs, labels)
        if metric=='mae': return self.mae(outputs, labels)
        if metric=='r2': return self.r2(outputs, labels)
        if metric=='ev': return self.ev(outputs, labels)
