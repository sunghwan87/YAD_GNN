import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geom_nn
from torch_geometric import utils as tgutils
from torch_geometric_signed_directed.nn.directed.complex_relu import complex_relu_layer

from torchmetrics import Accuracy, AUROC, AveragePrecision, Precision, Recall, Specificity, F1Score
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError, ExplainedVariance
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix

import pytorch_lightning as pl

# custom modules
from dl.modules import GNN, DSGNN, NoisyGNN
from dl.modules import SIGDecoder, SIGEncoder, GNNEncoder, VGAEGNNEncoder, GraphInnerMLPDecoder, GraphMLPDecoder
from dl.vae.optimizer import loss_function
from dl.layers import GradientReversalFunction, wGINConv
from dl.loss import ConfusionLoss
from dl.utils import mask_feature
#from dl.magnet import MagNetConv, MagNet

EPS = 1E-15

###################### Pytorch Lightning Module ###################
#https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html

class GraphLevelGNN(pl.LightningModule):
    def __init__(self, args, **model_kwargs):
        super().__init__()
        self.save_hyperparameters() # Saving hyperparameters
        
        ### model setting
        if args.gnn_type in ["MagNet", "MSGNN"]:
            self.gnn_embedding = DSGNN(
                n_nodes = args.n_nodes,
                input_dim = args.input_dim,
                hidden_dims = args.hidden_dims,
                output_dim = args.z_dim,
                hidden_concat = args.hidden_concat,
                activation = complex_relu_layer(),
                batch_norm = geom_nn.GraphNorm, #nn.BatchNorm1d, 
                dropout = args.dropout,
                device = args.device,
                batch_size = args.batch_size,
                pooling = args.pooling,
                gnn_type = args.gnn_type,
                laplace_norm = "sym", # None,
                q = 0.25,
                K = 2,
            ) 
        elif args.gnn_type == 'MLP':
            self.channel_list = [args.n_nodes*args.input_dim, args.z_dim] #et_mlp_channel_list(args.input_dim*args.n_nodes, args.z_dim)
            self.gnn_embedding = geom_nn.MLP(
                channel_list = self.channel_list, 
                dropout = args.dropout,
                act = 'relu',
                batch_norm = True,
                plain_last = True,
            )
        else: 
            self.gnn_embedding = GNN(
                gnn_type = args.gnn_type,
                #binary_edge = args.binarize,
                n_nodes = args.n_nodes,
                input_dim = args.input_dim, 
                hidden_dims = args.hidden_dims,
                output_dim = args.z_dim,
                hidden_concat = args.hidden_concat,
                activation = nn.LeakyReLU(), #nn.SELU(), #nn.ReLU(),
                batch_norm= nn.BatchNorm1d,
                dropout = args.dropout, 
                device = args.device,
                batch_size = args.batch_size,
                pooling = args.pooling,
            )

        self.task = args.task
        self.gnn_type = args.gnn_type
        self.num_classes = args.n_classes
        self.num_sites = args.n_sites
        self.lr_init = args.lr
        self.batch_size = args.batch_size
        self.binarize = args.binarize
        self.epochs = args.epochs
        self.embedding_dim = args.z_dim
        self.dropout = args.dropout
        # if args.hidden_concat:
        #     self.embedding_dim = sum(args.hidden_dims) # args.hidden_dims[-1] #sum(args.hidden_dims) pre-trained
        # else:
        #     self.embedding_dim = args.hidden_dims[-1]

        if self.task=='graph_classification':
            # define target classifier which train using label
            if args.n_classes == 2: 
                self.target_output_dim = 1
            elif args.n_classes > 2: 
                self.target_output_dim = args.n_classes
            self.target_loss_module = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos_weight])) if self.target_output_dim == 1 \
                else nn.CrossEntropyLoss(label_smoothing=args.label_smooth, weight=torch.Tensor(args.label_weights))
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

        elif self.task=='graph_regression':
            self.target_output_dim = args.target_output_dim
            self.target_loss_module = nn.MSELoss()
            self.metrics = ['rmse', 'mae', 'r2', 'ev']
            self.rmse = MeanSquaredError(squared=False)
            self.mae = MeanAbsoluteError()
            self.r2 = R2Score(num_outputs=1, adjusted=0)
            self.ev = ExplainedVariance()

        #print(f"Embedding dimension: {self.embedding_dim}")
        self.target_input_dim = self.embedding_dim
        self.target_channel_list = [self.target_input_dim, round(self.embedding_dim/2),  self.target_output_dim]
        #self.target_channel_list = [self.target_input_dim, self.target_output_dim]
        self.target_classifier = geom_nn.MLP(
                channel_list = self.target_channel_list, 
                dropout = args.dropout,
                act = "relu",
                norm = "batch_norm",
                plain_last = True,
                )


        if args.grl or args.domain_unlearning:
            # define domain classifier which train using antilabel via GRL layer -- https://github.com/jvanvugt/pytorch-domain-adaptation
            self.domain_output_dim = args.n_sites
            self.domain_input_dim = self.embedding_dim
            #self.domain_channel_list = [self.domain_input_dim, round(self.embedding_dim)*2, round(self.embedding_dim), round(self.embedding_dim), self.domain_output_dim]
            self.domain_channel_list = [self.domain_input_dim, self.domain_input_dim,  self.domain_input_dim, self.domain_output_dim]            
            self.domain_classifier = geom_nn.MLP(
                channel_list = self.domain_channel_list, 
                dropout = args.dropout,
                act = "relu",
                norm = "batch_norm",
                plain_last = True,
            )
            #nn.Sequential(
                #GradientReversalLayer(lambda_=args.reverse_grad_weight),
                #GradientReversalLayer2(alpha=args.reverse_grad_weight),
                #nn.Linear(self.domain_input_dim, self.domain_output_dim),
                
            #     nn.Linear(self.domain_input_dim, self.domain_hidden_dims[0]),
            #     nn.ReLU(),
            #     nn.BatchNorm1d(self.domain_hidden_dims[0]),
            #     nn.Linear(self.domain_hidden_dims[0], self.domain_output_dim),
            #     nn.Softmax(dim=-1),
            # )
            self.domain_loss_module = nn.CrossEntropyLoss(reduction='mean')
            if args.grl: 
                self.grl = True
            else:
                self.grl = False
            if args.domain_unlearning:            
                self.pretrain_epochs = 50
                self.domain_unlearning = True  # use multiple optimizer
                self.automatic_optimization = False # Important: This property activates manual optimization.     
                self.domain_unlearning_alpha = args.domain_unlearning_alpha
                self.domain_unlearning_beta = args.domain_unlearning_beta
                self.confusion_loss_module = ConfusionLoss()
                
            else:
                self.domain_unlearning = False
        else:
            self.domain_unlearning = False
            self.domain_unlearning_alpha = 0
            self.domain_unlearning_beta = 0
            self.grl = False

    def configure_optimizers(self):
        #print(self.parameters())
        if self.domain_unlearning ==False:
            parameter_list = [
                {"params": self.target_classifier.parameters(), "lr": self.lr_init},
                {"params": self.gnn_embedding.parameters(), "lr": self.lr_init},
            ]
            if self.grl:
                parameter_list += [{"params": self.domain_classifier.parameters(), "lr": self.lr_init}]
                #torch.optim.AdamW()
            optimizer = torch.optim.Adam(parameter_list, weight_decay=1e-2)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=self.batch_size*10, verbose=1)
            #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, verbose=0)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}
        elif self.domain_unlearning ==True:
            pretrain_parameter_list = [
                {"params": self.target_classifier.parameters(), "lr": self.lr_init},
                {"params": self.domain_classifier.parameters(), "lr": self.lr_init},
                {"params": self.gnn_embedding.parameters(), "lr": self.lr_init},
            ]
            target_parameter_list = [
                {"params": self.target_classifier.parameters(), "lr": self.lr_init},
                {"params": self.gnn_embedding.parameters(), "lr": self.lr_init},
            ]
            domain_parameter_list = [
                {"params": self.domain_classifier.parameters(), "lr": self.lr_init*0.1},
                #{"params": self.gnn_embedding.parameters(), "lr": self.lr_init*0.1},
            ]
            confusion_parameter_list = [
                {"params": self.gnn_embedding.parameters(), "lr": self.lr_init*0.1},
            ]
            pretrain_optimizer  = torch.optim.AdamW(pretrain_parameter_list)
            target_optimizer    = torch.optim.AdamW(target_parameter_list)
            domain_optimizer    = torch.optim.AdamW(domain_parameter_list)
            confusion_optimizer = torch.optim.AdamW(confusion_parameter_list)
            opts = [ pretrain_optimizer, target_optimizer, domain_optimizer, confusion_optimizer]
            pretrain_lrs = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(pretrain_optimizer, mode='min', factor=0.8, patience=20, verbose=0),
                "monitor": "val_loss",
            }
            target_lrs = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(target_optimizer, mode='min', factor=0.8, patience=20, verbose=0),
                "monitor": "val_loss",
            }
            domain_lrs = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(domain_optimizer, mode='min', factor=0.8, patience=20, verbose=0),
                "monitor": "val_loss",
            }
            confusion_lrs = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(confusion_optimizer, T_max=self.batch_size*5, verbose=0),
                #"monitor": "val_loss",
            }
            lrs = [pretrain_lrs, target_lrs, domain_lrs, confusion_lrs]
            return opts, lrs
            # return (
            #     {"optimizer": target_optimizer, "lr_scheduler": target_lrs},
            #     {"optimizer": domain_optimizer, "lr_scheduler": domain_lrs},
            #     {"optimizer": confusion_optimizer, "lr_scheduler": confusion_lrs},
            #     )

    def get_label(self, data, smoothing=False):
        if self.task=='graph_classification' and self.num_classes==2:
            return torch.Tensor(data.y).long()
        elif self.task=='graph_classification' and self.num_classes>2:
            return F.one_hot(torch.tensor(data.y), self.num_classes)
        else:
            return torch.tensor(data.y)

    def get_site(self, data):
        if self.num_sites==2:
            return torch.Tensor(data.site).long()
        else:
            return F.one_hot(torch.tensor(data.site), self.num_sites)

    def get_subject_id(self, data):
        return data.subject_id

    def get_pred(self, logits, n_classes):
        if n_classes==2: # binary classification: output_dim = 1
            pred_class = (logits > 0).type_as(logits)
            pred_probs = torch.sigmoid(logits)
            return pred_class, pred_probs
        elif n_classes>2: # self.num_classes=2
            pred_class = logits.argmax(dim=-1)            
            pred_probs = torch.softmax(logits, dim=-1)
            return pred_class, pred_probs
        else:
            ValueError(f"Inappropriate number of class: {n_classes}")


    def forward(self, data, training=False, *kwargs):

        ### prepare data
        features = data.x
        sites = self.get_site(data)        
        batch = data.batch
        edge_index, edge_weight = data.edge_index, data.edge_attr
        if tgutils.contains_self_loops(edge_index): 
            edge_index, edge_weight = tgutils.add_self_loops(edge_index=edge_index, edge_attr=edge_weight, fill_value=0)
        # if training: # feature masking & edge dropping
        #     #features, feature_mask = mask_feature(x=features, p=self.dropout)
        #     edge_index, edge_weight = tgutils.dropout_adj(edge_index=edge_index, edge_attr=edge_weight, p=0.25, training=training)

        ### Feedforward
        if self.gnn_type != 'MLP':         
            hidden = self.gnn_embedding(x=features, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
            #print(f"hidden:{hidden.shape}")
        else: # MLP
            adjs = tgutils.to_dense_adj(edge_index=edge_index, edge_attr=edge_weight, batch=batch)
            adjs_flatten = torch.flatten(adjs, start_dim=1, end_dim=-1)
            hidden = self.gnn_embedding(adjs_flatten)
        target_outputs = self.target_classifier(hidden)
        target_labels = self.get_label(data).type_as(target_outputs)
        if self.num_classes>2: 
            target_labels = target_labels.argmax(dim=1)
            target_outputs = F.softmax(target_outputs, dim=1)
        else:
            target_outputs = target_outputs.squeeze(dim=-1)
        # print("target_outputs:", target_outputs.shape)
        # print("target_labels:", target_labels.shape) 
        target_loss = self.target_loss_module(target_outputs, target_labels)
        #print("target loss: ", target_loss)   
            
        if self.grl:
            ## Gradient reversal
            p = self.current_epoch / self.epochs
            alpha = 2.0/(1+np.exp(-10*p)) - 1  # Ganin and Lempitsky, 2015
            #print(p, alpha)
            hidden = GradientReversalFunction.apply(hidden, alpha)
            domain_outputs = self.domain_classifier(hidden).squeeze(dim=-1)
            #print(domain_outputs)
            #domain_outputs = F.softmax(domain_outputs, dim=1)
            domain_labels = sites.type_as(domain_outputs)
            domain_loss = self.domain_loss_module(domain_outputs, domain_labels)  
            confusion_loss = torch.tensor([0.0]).type_as(target_loss)          

        elif self.domain_unlearning and training: 

            pretrain_optimizer, target_optimizer, domain_optimizer, confusion_optimizer = self.optimizers()   
            target_loss = 0
            for site in np.unique(data.site): # calculate loss for each domain -- prevent largest-domain dominate optimization                
                ## check!!
                # print(site)
                # print(target_labels[data.site==site])
                target_loss += self.target_loss_module(target_outputs[data.site==site], target_labels[data.site==site])   
            domain_outputs = self.domain_classifier(hidden.detach()) # hidden should be detached!!!
            domain_labels = sites.type_as(domain_outputs)
            domain_loss = self.domain_unlearning_alpha * self.domain_loss_module(domain_outputs, domain_labels) 
            # print("domain_outputs: ", domain_outputs)
            # print("domain_labels: ", domain_labels)      
            # print("domain loss: ", domain_loss)  

            # STAGE 1
            if self.current_epoch < self.pretrain_epochs: # unlearning after pretraining   
                pretrain_optimizer.zero_grad()
                loss = target_loss + domain_loss
                self.manual_backward(loss) 
                pretrain_optimizer.step()
                confusion_loss = self.domain_unlearning_beta * self.confusion_loss_module(domain_outputs, domain_labels)  
            # STAGE 2
            else:                
                p = self.current_epoch / self.epochs
                beta = 2.0/(1+np.exp(-10*p)) - 1  #  [0-->1] Ganin and Lempitsky, 2015
                alpha = 1 - beta   # [1 --> 0]     

                # First update the encoder and target classifier    
                # target_loss = self.target_loss_module(target_outputs, target_labels)            
                #target_loss.requires_grad_(True)                 
                target_optimizer.zero_grad()    
                self.manual_backward(target_loss, retain_graph=True)
                target_optimizer.step()
                
                # Now update just the domain classifier
                domain_optimizer.zero_grad()       
                domain_outputs = self.domain_classifier(hidden.detach()) # hidden should be detached!!!
                domain_labels = sites.type_as(domain_outputs)
                domain_loss = self.domain_unlearning_alpha * alpha * self.domain_loss_module(domain_outputs, domain_labels)                 
                self.manual_backward(domain_loss)
                domain_optimizer.step()
                
                # Now update just the encoder using the domain loss      
                confusion_optimizer.zero_grad()         
                domain_preds = self.domain_classifier(hidden.detach())
                #print("domain_preds: ", domain_preds.shape)            
                confusion_loss = self.domain_unlearning_beta * beta *  self.confusion_loss_module(domain_preds, domain_labels)                
                self.manual_backward(confusion_loss)
                confusion_optimizer.step()
                

            
        else:
            # ### Feedforward
            # if self.gnn_type != 'MLP':         
            #     hidden = self.gnn_embedding(x=features, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
            # else: # MLP
            #     adjs = tgutils.to_dense_adj(edge_index=edge_index, edge_attr=edge_weight, batch=batch)
            #     adjs_flatten = torch.flatten(adjs, start_dim=1, end_dim=-1)
            #     hidden = self.gnn_embedding(adjs_flatten)
            # target_outputs = self.target_classifier(hidden).squeeze(dim=-1)   #logits = out.squeeze(dim=-1)
            # target_labels = self.get_label(data).type_as(features)            
            # if self.num_classes>2: 
            #     target_labels = target_labels.argmax(dim=1)
            #     target_outputs = F.softmax(target_outputs, dim=1)    
            # target_loss = self.target_loss_module(target_outputs, target_labels)
            domain_loss = torch.tensor([0.0]).type_as(target_loss)
            confusion_loss = torch.tensor([0.0]).type_as(target_loss)
        #print("edge_weights:", edge_weight.shape, edge_weight)
        #print("hidden dim:", hidden)

        torch.cuda.empty_cache()
        return target_loss, domain_loss, confusion_loss, target_outputs, target_labels, hidden
    

    def training_step(self, batch, batch_idx):
        target_loss, domain_loss, confusion_loss, _, _, _ = self.forward(batch, training=True) # return loss, target_loss, target_logits, labels, hidden
        total_loss = target_loss + domain_loss + confusion_loss
        return {"loss": total_loss, "target_loss": target_loss, "domain_loss": domain_loss, "confusion_loss": confusion_loss}

    def training_epoch_end(self, batch_outputs): 
        avg_target_loss = torch.concat([x["target_loss"].unsqueeze(0) for x in batch_outputs]).mean()
        avg_domain_loss = torch.concat([x["domain_loss"].unsqueeze(0) for x in batch_outputs]).mean()
        avg_confusion_loss = torch.concat([x["confusion_loss"].unsqueeze(0) for x in batch_outputs]).mean()
        avg_loss = torch.concat([x["loss"].unsqueeze(0) for x in batch_outputs]).mean()
        self.log('train_loss', avg_loss, batch_size=self.batch_size)
        self.log('target_loss', avg_target_loss, batch_size=self.batch_size)
        self.log('domain_loss', avg_domain_loss, batch_size=self.batch_size)       
        self.log('confusion_loss', avg_confusion_loss, batch_size=self.batch_size)
        if self.domain_unlearning:
            #print(self.trainer.callback_metrics)
            pretrain_lrs, target_lrs, domain_lrs, confusion_lrs = self.lr_schedulers()
            pretrain_lrs.step(self.trainer.callback_metrics["val_loss"])
            target_lrs.step(self.trainer.callback_metrics["val_loss"])
            domain_lrs.step(self.trainer.callback_metrics["domain_loss"])
            confusion_lrs.step(self.trainer.callback_metrics["confusion_loss"])
            
    def validation_step(self, batch, batch_idx):
        target_loss, domain_loss, confusion_loss, outputs, labels, _ = self.forward(batch)
        total_loss = target_loss
        return {"loss": total_loss, "labels": labels, "outputs":outputs}

    def validation_epoch_end(self, batch_outputs):
        avg_loss = torch.concat([x["loss"].unsqueeze(0) for x in batch_outputs]).mean()
        labels = torch.concat([x[f"labels"] for x in batch_outputs])
        outputs = torch.concat([x[f"outputs"] for x in batch_outputs])
        self.log(f"val_loss", avg_loss)

        for metric in self.metrics:
            metric_value = self.calculate_metric(metric, labels, outputs)            
            self.log(f"val_{metric}", metric_value)


    def test_step(self, batch, batch_idx):
        target_loss, domain_loss, confusion_loss, outputs, labels, _ = self.forward(batch) 
        loss = target_loss
        return {"loss": loss, "labels": labels, "outputs":outputs}

    def test_epoch_end(self, batch_outputs):
        avg_loss = torch.concat([x[f"loss"].unsqueeze(0) for x in batch_outputs]).mean()
        labels = torch.concat([x[f"labels"] for x in batch_outputs]).squeeze()
        outputs = torch.concat([x[f"outputs"] for x in batch_outputs]).squeeze()
        self.log(f"test_loss", avg_loss)
        for metric in self.metrics:
            metric_value = self.calculate_metric(metric, labels, outputs)            
            self.log(f"test_{metric}", metric_value)
        if self.task=='graph_classification':
            cm = self.calculate_metric('cm', labels, outputs)  # confusion_matrix
            print(cm)        

    def predict_step(self, batch, batch_idx):
        _, _, _, outputs, labels, embedding = self.forward(batch)
        subject_id = self.get_subject_id(batch)
        if self.task=='graph_classification':
            pred_class, pred_probs = self.get_pred(outputs, self.num_classes)
            return {"labels": labels, "pred_class": pred_class, "pred_probs": pred_probs, "outputs":outputs, "embedding":embedding, "subject_id":subject_id}
        elif self.task=='graph_regression': 
            return {"labels": labels, "outputs":outputs, "embedding":embedding, "subject_id":subject_id}
    
    def on_predict_epoch_end(self, results):
        batch_results = results[0]
        labels = torch.concat([x[f"labels"] for x in batch_results]).squeeze()
        outputs = torch.concat([x[f"outputs"] for x in batch_results]).squeeze()
        embedding = torch.concat([x[f"embedding"] for x in batch_results]).squeeze()
        subject_id = np.concatenate([x[f"subject_id"] for x in batch_results ]).squeeze()
        if 'pred_class' in batch_results[0].keys(): pred_class = torch.concat([x[f"pred_class"] for x in batch_results]).squeeze()
        if 'pred_probs' in batch_results[0].keys(): pred_probs = torch.concat([x[f"pred_probs"] for x in batch_results]).squeeze()
        return {"labels": labels, "outputs":outputs, "embedding":embedding, "subject_id":subject_id}

    
    def calculate_metric(self, metric, labels, outputs):
        if self.task=='graph_classification':
            labels = labels.long()
            #pred_class, pred_probs = self.get_pred(outputs, self.num_classes)
            if self.num_classes==2: # binary classification: output_dim = 1
                pred_class = (outputs > 0).type_as(outputs)
                pred_probs = torch.stack([1-torch.sigmoid(outputs), torch.sigmoid(outputs)], dim=-1)
                #print(pred_probs.shape, labels.shape)

            elif self.num_classes>2: # self.num_classes=2
                pred_class = outputs.argmax(dim=-1)            
                pred_probs = torch.softmax(outputs, dim=-1)
                
        # print("metric:", metric)
        # print("label:", labels)
        # print("logit:", logits.shape)
        # print("pred_class:", pred_class)
        # print("pred_probs:", pred_probs)
        #if self.num_classes == 2:
        #     if metric=='roc': return self.aucroc_score(pred_probs, labels) #roc_auc_score(labels, pred_probs)
        #     if metric=='avp': return self.avg_precision_score(pred_probs, labels)  #average_precision_score(labels, pred_probs) 
        #     if metric=='acc': return self.accuracy_score(pred_probs, labels) #accuracy_score(labels, pred_class)
        #     if metric=='pr': return  self.precision_score(pred_class, labels) #precision_score(labels, pred_class)
        #     if metric=='rc': return  self.recall_score(pred_class, labels) #recall_score(labels, pred_class)
        # else:
        
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



class EdgeLevelAE(pl.LightningModule):
    def __init__(self, args, **model_kwargs):
        super().__init__()
        self.save_hyperparameters() # Saving hyperparameters
        ### model setting
        if args.encoder_type=='SIGVAE': 
            self.model = ModelSIGVAE(
                input_dim=args.input_dim, 
                noise_dim=args.noise_dim, 
                # Lu=len(args.hidden_u), 
                # Lmu=len(args.hidden_mu), 
                # Lsigma=len(args.hidden_mu), 
                output_dims_u=args.hidden_u, 
                output_dims_mu=args.hidden_mu, 
                output_dims_sigma=args.hidden_mu, 
                gnn_type=args.gnn_type,
                copyK=args.K,
                copyJ=args.J,
                decoder_type=args.decoder_type,
                device=args.device
                )
            self.loss_module = loss_function
        elif args.encoder-type=='vanillaGVAE':
            self.model = geom_nn.VGAE(
                encoder = NoisyGNN(
                    input_dim = args.input_dim,
                    output_dims = [32, 32, 16, 16],
                    noise_dim = 0,
                    activation = nn.ReLU(),
                    dropout = args.dropout,
                    gnn_type = args.gnn_type,
                    device = args.device
                ),
                decoder = torch_geometric.nn.models.InnderProductDecoder
            )
            self.loss_module = self.model.kl_loss
        self.encoder_type = args.encoder_type
        self.lr = args.lr
        self.batch_size = args.batch_size

    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(), lr=argv.lr, momentum=0.9, weight_decay=2e-3)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        return optimizer
    
    def get_label(self, data):
        return data.y

    def forward(self, data, mode="train"):
        
        ### prepare data
        features = data.x        
        if len(features.shape) == 2:
            features = features.view([1, features.shape[0], features.shape[1]])
        edge_index_without_selfloop, _ = tgutils.remove_self_loops(data.edge_index)
        adj_without_selfloop = tgutils.to_scipy_sparse_matrix(edge_index_without_selfloop)
        adj_with_selfloop = tgutils.to_scipy_sparse_matrix(data.edge_index)
        adj_label = torch.FloatTensor(adj_with_selfloop.todense())
        n_nodes = adj_label.shape[0]
        pos_weight = torch.tensor([float(n_nodes * n_nodes - adj_without_selfloop.sum()) / adj_without_selfloop.sum()])
        norm = torch.tensor([float(n_nodes * n_nodes / float((n_nodes * n_nodes - adj_without_selfloop.sum()) * 2))])  

        ### Feedforward
        adj = data.edge_index.to(self.model.device)
        features = features.to(self.model.device)
        adj_label = adj_label.to(self.model.device)
        pos_weight = pos_weight.to(self.model.device)
        norm = norm .to(self.model.device)

                
        if self.encoder_type=='SIGVAE':
            recovered, mu, logvar, z, z_scaled, eps, rk, _ = self.model(adj, features)
            #self.model.hidden_emb = z_scaled.clone().detach() # copying tensor
        
            ### Calculate loss
            loss_rec, loss_prior, loss_post = self.loss_module(
                preds=recovered, 
                labels=adj_label,
                mu=mu, 
                logvar=logvar, 
                emb=z_scaled, 
                eps=eps, 
                n_nodes=n_nodes,
                norm=norm, 
                pos_weight=pos_weight, 
            )
            #WU = np.min([epoch/100., 1.])
            loss = loss_rec + (loss_post - loss_prior) * 1 / (n_nodes**2)     
            #curr_loss = loss.item()
            hidden_emb = z_scaled
            del z, z_scaled, recovered, mu, logvar, eps, rk, features, adj, adj_label, loss_rec, loss_prior, loss_post   
        
        elif self.encoder_type=="vanillaGVAE":
            z = self.model.encode(adj)
            loss = self.loss_module()

        if mode=='train':
            del hidden_emb
            gc.collect()
            torch.cuda.empty_cache()
            return loss, None, None
        
        elif mode=='eval':
            self.model.eval()
            pos_edge, neg_edge = data.pos_edge_label_index, data.neg_edge_label_index
            roc, ap = self.get_roc_score(
                    hidden_emb,
                    pos_edge,
                    neg_edge,
                    self.model.decoder_type
                )
            del pos_edge, neg_edge, hidden_emb
            gc.collect() 
            torch.cuda.empty_cache()
            return loss, roc, ap

        elif mode=='pred':
            gc.collect()
            torch.cuda.empty_cache()
            return hidden_emb


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        loss, _, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss, batch_size=self.batch_size)
        #val_loss, roc, ap = self.forward(batch, mode="eval")
        #self.log('val_acc', roc)        
        #print(f"train_loss={loss:.5f}, val_loss={val_loss:.5f}, val_roc={roc:.4f}, val_ap={ap:.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, roc, ap = self.forward(batch, mode="eval")
        self.log('val_loss', loss, batch_size=self.batch_size)
        self.log('val_roc', roc, batch_size=self.batch_size)
        self.log('val_ap', ap, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        loss, roc, ap = self.forward(batch, mode="eval")
        self.log('test_loss', loss, batch_size=self.batch_size)
        self.log('test_roc', roc, batch_size=self.batch_size)
        self.log('test_ap', ap, batch_size=self.batch_size)

    def predict_step(self, batch, batch_idx):
        hidden_emb = self.forward(batch, mode="pred")
        labels = self.get_label(batch)
        return {"embeddings": hidden_emb, "labels": labels}

    def get_roc_score(self, emb, edges_pos, edges_neg, decoder_type):
        from sklearn.metrics import roc_auc_score, average_precision_score
        J = emb.shape[0]
        preds = []
        for edges in [edges_pos, edges_neg]:
            emb_src, emb_dst = emb[:, edges[0], :], emb[:, edges[1], :]
            pred = torch.einsum('ijk,ijk->ij', emb_src, emb_dst) # dot for k
            if decoder_type=='inner':
                pred = 1 / (1 + torch.exp(-pred))
            elif decoder_type=='bp':
                pred = 1 - torch.exp( - torch.exp(pred)) 
            preds.append(pred)
        preds_all = torch.hstack(preds).detach().cpu().numpy()
        labels_all = torch.hstack([torch.ones(preds[0].shape[-1]), torch.zeros(preds[1].shape[-1])]).detach().cpu().numpy()

        roc_score = torch.tensor( [roc_auc_score(labels_all, pred_all.flatten()) for pred_all in np.vsplit(preds_all, J)]).mean()
        ap_score = torch.tensor( [average_precision_score(labels_all, pred_all.flatten()) for pred_all in np.vsplit(preds_all, J)] ).mean()
        return roc_score.to(self.model.device), ap_score.to(self.model.device)

class GraphAutoEncoder(pl.LightningModule):
    def __init__(self, args, **model_kwargs):
        self.hybrid=False
        super().__init__()
        self.save_hyperparameters() # Saving hyperparameters
        ### model setting
        if args.decoder_type=="mlp": decoder = GraphMLPDecoder(z_dim=args.z_dim, n_nodes=args.n_nodes)
        elif args.decoder_type=="innermlp": decoder = GraphInnerMLPDecoder(args.n_nodes)
        elif args.decoder_type=="inner": decoder = geom_nn.InnerProductDecoder()
        
        if args.encoder_type=='SIG': 
            self.model = ModelSIGVAE(
                input_dim=args.input_dim, 
                noise_dim=args.noise_dim, 
                output_dims_u=args.hidden_u, 
                output_dims_mu=args.hidden_mu, 
                output_dims_sigma=args.hidden_mu, 
                gnn_type=args.gnn_type,
                copyK=args.K,
                copyJ=args.J,
                decoder_type=args.decoder_type,
                device=args.device
                )
            self.loss_module = loss_function
        elif args.encoder_type=='GAE':
            self.model = geom_nn.GAE(
                encoder = GNNEncoder(args),
                decoder = decoder
            )
        elif args.encoder_type=='VGAE':
            self.model = geom_nn.VGAE(
                encoder = VGAEGNNEncoder(args),
                decoder = decoder
            )
        elif args.encoder_type=='ARGVA':
            discriminator_dim = 64
            discriminator = nn.Sequential(
                nn.Linear(args.z_dim, discriminator_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(discriminator_dim),
                nn.Linear(discriminator_dim, discriminator_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(discriminator_dim),
                nn.Linear(discriminator_dim, 1),
            )
            self.discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
            self.model = geom_nn.ARGVA(
                encoder = VGAEGNNEncoder(args),
                decoder = decoder,
                discriminator = discriminator
            )


        # if self.hybrid:
        #     self.embedding_dim = args.hidden_dims[-1]
        #     # define target classifier which train using label
        #     if args.n_classes == 2: self.target_output_dim = 1
        #     else: self.target_output_dim = args.n_classes
        #     self.target_input_dim = self.embedding_dim
        #     self.target_hidden_dims = [round(self.target_input_dim/2)]
        #     self.target_classifier = nn.Sequential(
        #         nn.Linear(self.target_input_dim, self.target_hidden_dims[0]),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(self.target_hidden_dims[0]),
        #         nn.Linear(self.target_hidden_dims[0], self.target_output_dim),
        #     )
        #     self.target_loss_module = nn.BCEWithLogitsLoss() if self.target_output_dim == 1 else nn.CrossEntropyLoss()

        #     if args.grl:
        #         # define domain classifier which train using antilabel via GRL layer -- https://github.com/jvanvugt/pytorch-domain-adaptation
        #         if args.n_sites == 2: self.domain_output_dim = 1
        #         else: self.domain_output_dim = args.n_sites
        #         self.domain_input_dim = self.embedding_dim
        #         self.domain_hidden_dims = [round(self.domain_input_dim/2)]
        #         self.domain_classifier = nn.Sequential(
        #             GradientReversalLayer(lambda_=args.reverse_grad_weight),
        #             nn.Linear(self.domain_input_dim, self.domain_hidden_dims[0]),
        #             nn.ReLU(),
        #             nn.BatchNorm1d(self.domain_hidden_dims[0]),
        #             nn.Linear(self.domain_hidden_dims[0], self.domain_output_dim),
        #         )
        #         self.domain_loss_module = nn.BCEWithLogitsLoss() if self.domain_output_dim == 1 else nn.CrossEntropyLoss()
        #         self.grl = True
        #     else:
        #         self.grl = False

        self.n_nodes = args.n_nodes
        self.encoder_type = args.encoder_type
        self.decoder_type = args.decoder_type
        
        self.lr_init = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        if args.pooling=='sum': self.pool = geom_nn.global_add_pool
        elif args.pooling=='mean': self.pool = geom_nn.global_mean_pool
        elif args.pooling=='max': self.pool = geom_nn.global_max_pool
        elif args.pooling=='gmt': self.pool = geom_nn.GraphMultisetTransformer(
            in_channels = args.hiddem_dim_mu, hidden_channels=int(args.hiddem_dim_mu), output_channels=1,
            Conv=geom_nn.GCNConv,
            num_nodes = self.n_nodes,
            pooling_ratio = 0.25,
            pool_sequences= ['GMPool_G', 'SelfAtt', 'GMPool_I'],
            num_heads=4,
            layer_norm=False
        )
        else: NotImplementedError(f"{args.pooling} is still not implemented.")

        if args.binarize:
            self.loss_recon = nn.BCEWithLogitsLoss(reduction='sum')
            self.binary = True
            self.recon_activation = None
        else:
            self.loss_recon = nn.MSELoss(reduction='sum') 
            #self.loss_recon = nn.L1Loss(reduction='mean') #nn.MSELoss(reduction='mean') #nn.BCELoss()
            # L1 loss performs better: Ma, Y., Li, Y., Liang, X. et al. Graph autoencoder for directed weighted network. Soft Comput 26, 1217–1230 (2022). https://doi.org/10.1007/s00500-021-06580-w
            self.binary = False
            self.recon_activation = None #nn.Linear(1, 1) #lambda x:x  #nn.Tanh() --> not good idea (lower loss but poor recon quality)
        
        self.loss_mse = torch.tensor(1.)  # initial value to calculate balancing parameter gamma   
        self.lamb = 0.1
        self.adj_scale = 1.
        self.feat_scale = 1.
        self.freeze_decoder = False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_init)
        # if self.epochs>5:
        #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(self.epochs/5), T_mult=2, eta_min=1e-7)  #https://sanghyu.tistory.com/113
        # else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train_loss"}
    
    def get_label(self, data):
        return torch.Tensor(data.y)

    def get_site(self, data):
        return torch.Tensor(data.site)

    def forward(self, data, mode='train'):        
        ### prepare data
        features = data.x * (self.feat_scale)
        batch = data.batch
        subject_id = data.subject_id
        edge_index, edge_weight = data.edge_index, data.edge_attr
     
        # edge weight [-1,1] --> [0,1]
        # edge_weight = (edge_weight + 1)/2
        if tgutils.contains_self_loops(edge_index): 
            edge_index, edge_weight = tgutils.remove_self_loops(edge_index, edge_weight)
        #edge_index_label, edge_weight_label = tgutils.add_self_loops(edge_index=edge_index, edge_attr=edge_weight, fill_value=1)
        edge_index_label, edge_weight_label = edge_index, edge_weight
        adj_label = tgutils.to_dense_adj(edge_index=edge_index_label, edge_attr=edge_weight_label, batch=batch).squeeze()
        #adj_label += torch.eye(adj_label.shape[0]).type_as(adj_label)
        #print("adj_orig:", tgutils.to_dense_adj(edge_index=edge_index, edge_attr=edge_weight).squeeze().shape )
        #edge_index_label, edge_weight_label = tgutils.add_self_loops(edge_index=edge_index, edge_attr=edge_weight, fill_value=1)


        ### Feedforward encoder
        z = self.model.encode(x=features, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        #print("z:",z.shape)

        # train ARGVA if used
        if self.encoder_type=='ARGVA' and mode=='train': # ARGVA trains discriminator
            for i in range(5):
                idx = range(self.n_nodes)  
                self.model.discriminator.train()
                self.discriminator_optimizer.zero_grad()
                discriminator_loss = self.model.discriminator_loss(z[idx]) # Comment
                discriminator_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()
            loss_reg = self.model.reg_loss(z)
        else:
            loss_reg = 0
        
        # Feedforward decoder
        if self.decoder_type =='mlp': 
            # READOUT
            if isinstance(self.pool, geom_nn.GraphMultisetTransformer): z_pooled = self.pool(z, batch, edge_index)
            else: z_pooled = self.pool(z, batch) 
            z_pooled = self.pool(z, batch)
            #print(z.shape, z_pooled.shape)
            adj_recon = self.model.decoder.forward_all(z_pooled, sigmoid=False)
        elif self.decoder_type=='innermlp':
            adj_recon = self.model.decoder.forward_all(z, batch, sigmoid=False)
        else:
            adj_recon = self.model.decoder.forward_all(z, sigmoid=True)
        edge_index_recon, edge_attr_recon = tgutils.dense_to_sparse(adj_recon)
        adj_recon = tgutils.to_dense_adj(edge_index=edge_index_recon, edge_attr=edge_attr_recon, batch=batch)#.unsqueeze(-1)        
        if self.recon_activation is not None: adj_recon = self.recon_activation(adj_recon)#.squeeze()
        #adj_recon = torch.block_diag(*adj_recon.repeat(self.batch_size,1,1)).type_as(adj_recon) # match with batched adj
        
        

        ### Calculate loss
        # print(adj_recon.shape)
        # print(adj_label.shape)
        #pos_weight
        #norm
        loss_recon = self.loss_recon(adj_recon.squeeze(), adj_label.squeeze())
        if self.model.encoder.gnn_embedding.linear is not None:
            loss_reg = 0
            #loss_reg = torch.norm(self.model.encoder.gnn_embedding.linear.weight, p=2) #torch.norm(z) #torch.norm(adj_recon) #nn.MSELoss()(torch.norm(adj_label), torch.norm(adj_recon))
        else:
            loss_reg = 0
        #print(adj_orig[0,0], adj_recon[0,0])
        if self.encoder_type=='VGAE':
            loss_kl = (1/self.n_nodes)*self.model.kl_loss()
        else:
            loss_kl = 0
        #print(loss_recon, loss_kl)

        # if self.hybrid:
        #     unbatched = unbatch(z, batch) 
        #     z_batched = torch.stack(unbatched).permute(0,2,1)  # z = (batch x node, z_dim) --> z_batched = (batch, node, z_dim) --> (batch, z_dim, node)
        #     z_pooled = z_batched.sum(axis=-1)
        #     target_logits = self.target_classifier(z_pooled).squeeze(dim=-1)   #logits = out.squeeze(dim=-1)
        #     labels = self.get_label(data).type_as(features)
        #     target_loss = self.target_loss_module(target_logits, labels)
            
        #     if self.grl:
        #         domain_logits = self.domain_classifier(z_pooled).squeeze(dim=-1)
        #         sites = self.get_site(data).type_as(features)
        #         domain_loss = self.domain_loss_module(domain_logits, sites)
        #     else: domain_loss=0
        # else: 
        #     target_loss=0
           
        # loss_reg += target_loss + domain_loss

        if self.binary: 
            adj_recon = (adj_recon>0.5).type_as(adj_recon)
        # if self.current_epoch==self.trainer.max_epochs:
        #     print(loss_recon)
        #     print(adj_label)
        #     print(z)
        #     print(adj_recon)

        if (self.current_epoch>=int(self.trainer.max_epochs/2)) and self.freeze_decoder==False:
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            self.freeze_decoder = True
            print("decoder freezed.")

        return loss_recon, loss_kl, loss_reg, z, adj_label, adj_recon, subject_id

    def training_step(self, batch, batch_idx):
        loss_recon, loss_kl, loss_reg, _, _, _, _ = self.forward(batch)
        self.loss_mse = self.loss_mse.type_as(loss_recon).detach()
        self.loss_mse = min(self.loss_mse, self.loss_mse*0.99+loss_recon*0.01)
        gamma = torch.sqrt(self.loss_mse) + EPS
        #if self.current_epoch<= round(self.epochs/10): loss = (loss_kl + loss_reg) / self.batch_size  # warm-up
        #else: loss = (loss_recon + loss_kl + loss_reg) / self.batch_size
        loss = (loss_recon + gamma*loss_kl + self.lamb*loss_reg) / self.batch_size
        
        #loss = (1/(2*gamma))*loss_recon + loss_kl + loss_reg  #Asperti et al.(2020), DOI: 10.1109/ACCESS.2020.3034828, https://github.com/asperti/BalancingVAE/blob/d9f399774259190d423fe12915d126a44e9ab9a0/computed_gamma.py#L322
        self.log('train_loss', loss, batch_size=self.batch_size)
        self.log('train_recon', loss_recon, batch_size=self.batch_size)
        self.log('train_kl', loss_kl, batch_size=self.batch_size)
        self.log('train_regul', loss_reg, batch_size=self.batch_size)
        self.log('gamma', gamma, batch_size=self.batch_size)
        return loss

    # def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
    #     return super().training_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        loss_recon, loss_kl, loss_reg, _, _, _, _  = self.forward(batch, mode='test')
        loss = loss_recon + loss_kl + loss_reg
        self.log('test_loss', loss, batch_size=self.batch_size)
        self.log('test_recon', loss_recon, batch_size=self.batch_size)
        self.log('test_kl', loss_kl, batch_size=self.batch_size)
        self.log('test_regul', loss_reg, batch_size=self.batch_size)

    def predict_step(self, batch, batch_idx):
        _, _, _, z, adj_label, adj_recon, subject_id = self.forward(batch, mode='test')
        return z.detach().cpu().numpy().squeeze(), adj_label.detach().cpu().numpy().squeeze(), adj_recon.detach().cpu().numpy().squeeze(), subject_id

    def on_predict_end(self):
        # z, adj_label, adj_recon = z.detach().cpu().numpy().squeeze(), adj_label.detach().cpu().numpy().squeeze(), adj_recon.detach().cpu().numpy().squeeze() 
        # fig = draw_recon(adj_label[i,:,:], adj_recon[i,:,:], z[i,:])
        # plot_buf = io.BytesIO()
        # fig.savefig(plot_buf, format="png")
        # plot_buf.seek(0)
        # image = PIL.Image.open(plot_buf)
        # image = ToTensor()(image) #.unsqueeze(0)
        # self.logger.experiment.add_image(f"{subject_id[i]}", image)
        # plt.close(fig)
        # del plot_buf, image

        return super().on_predict_end()


###################### Integrating encoder and decoder ###################
class ModelSIGVAE(nn.Module):
    def __init__(self, input_dim, noise_dim, output_dims_u, output_dims_mu, output_dims_sigma, copyK=1, copyJ=1, decoder_type='inner', gnn_type='GIN', noise_dist='Bernoulli', activation=nn.ReLU, dropout=0, device='cpu'):
        super(ModelSIGVAE, self).__init__()
        self.device = device
        self.decoder_type = decoder_type
        assert output_dims_mu[-1]==output_dims_sigma[-1]
        self.z_dim = output_dims_mu[-1]
        self.encoder = SIGEncoder(
            #Lu=Lu, 
            #Lmu=Lmu, 
            #Lsigma=Lsigma, 
            input_dim=input_dim, 
            noise_dim=noise_dim, 
            output_dims_u=output_dims_u, 
            output_dims_mu=output_dims_mu, 
            output_dims_sigma=output_dims_sigma, 
            K=copyK,
            J=copyJ,
            activation=activation, 
            gnn_type=gnn_type,
            noise_dist=noise_dist,
            dropout=dropout,
            device=device,
            )

        self.decoder = SIGDecoder(zdim=self.z_dim, dropout=dropout, decoder_type=self.decoder_type)

    def forward(self, A, X):
        z, eps, mu, logvar = self.encoder.forward(A, X)
        adj_, z_scaled, rk = self.decoder(z)
        snr = None
        return adj_, mu, logvar, z, z_scaled, eps, rk, snr

class DualHyperTranformer(pl.LightningModule):
    pass        