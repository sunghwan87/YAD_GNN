from audioop import reverse
import gc
from torchmetrics import Accuracy, AUROC, AveragePrecision, Precision, Recall
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import utils as tgutils
import torch_geometric.nn  as geom_nn
import pytorch_lightning as pl
from dl.supervised.gradient_reversal import GradientReversalLayer
from dl.utils import minmax_scale
from dl.models import GNN

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GIN": geom_nn.GINConv,
    "GraphConv": geom_nn.GraphConv
}


class GraphLevelGNN(pl.LightningModule):
    def __init__(self, args, **model_kwargs):
        super().__init__()
        self.save_hyperparameters() # Saving hyperparameters

        ### model setting
        self.gnn_embedding = GNN(
            gnn_type = args.gnn_type,
            #binary_edge = args.binarize,
            n_nodes = args.n_nodes,
            input_dim = args.input_dim, 
            hidden_dims = args.hidden_dims,
            output_dim = args.z_dim,
            hidden_concat = args.hidden_concat,
            activation = nn.LeakyReLU(), #nn.SELU(), #nn.ReLU(),
            normalization= nn.BatchNorm1d,
            dropout = args.dropout, 
            device = args.device,
            batch_size = args.batch_size,
            pooling = args.pooling,
        )
        self.embedding_dim = args.z_dim
        # if args.hidden_concat:
        #     self.embedding_dim = sum(args.hidden_dims) # args.hidden_dims[-1] #sum(args.hidden_dims) pre-trained
        # else:
        #     self.embedding_dim = args.hidden_dims[-1]

        # define target classifier which train using label
        if args.n_classes == 2: 
            self.target_output_dim = 1
        elif args.n_classes > 2: 
            self.target_output_dim = args.n_classes
        self.target_input_dim = self.embedding_dim
        self.target_hidden_dims = [round(self.embedding_dim/2)]
        self.target_classifier = nn.Sequential(
            nn.Linear(self.target_input_dim, self.target_hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(self.target_hidden_dims[0]),
            nn.Linear(self.target_hidden_dims[0], self.target_output_dim),
        )

        self.target_loss_module = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos_weight])) if self.target_output_dim == 1 else nn.CrossEntropyLoss(weight=torch.Tensor(args.label_weights))

        if args.grl:
            # define domain classifier which train using antilabel via GRL layer -- https://github.com/jvanvugt/pytorch-domain-adaptation
            if args.n_sites == 2: self.domain_output_dim = 1
            elif args.n_sites > 2:self.domain_output_dim = args.n_sites
            self.domain_input_dim = self.embedding_dim
            self.domain_hidden_dims = [round(self.embedding_dim/2)]
            self.domain_classifier = nn.Sequential(
                GradientReversalLayer(lambda_=args.reverse_grad_weight),
                nn.Linear(self.domain_input_dim, self.domain_hidden_dims[0]),
                nn.ReLU(),
                nn.BatchNorm1d(self.domain_hidden_dims[0]),
                nn.Linear(self.domain_hidden_dims[0], self.domain_output_dim),
            )
            self.domain_loss_module = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos_weight])) if self.domain_output_dim == 1 else nn.CrossEntropyLoss(weight=torch.Tensor(args.label_weights))
            self.grl = True
        else:
            self.grl = False

        self.num_classes = args.n_classes
        self.num_sites = args.n_sites
        self.lr_init = args.lr
        self.batch_size = args.batch_size
        self.binarize = args.binarize
        self.epochs = args.epochs
        self.aucroc_score = AUROC(num_classes=self.target_output_dim, average="macro")
        self.accuracy_score = Accuracy(num_classes=self.target_output_dim, average="macro")
        self.avg_precision_score = AveragePrecision(num_classes=self.target_output_dim, average="macro")
        self.precision_score = Precision(num_classes=self.target_output_dim, average="macro")
        self.recall_score = Recall(num_classes=self.target_output_dim, average="macro")
        # from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_score, recall_score
        # self.aucroc_score = roc_auc_score
        # self.avg_precision_score = average_precision_score
        # self.precision_score = precision_score
        # self.recall_score = recall_score

        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr_init, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}

    def get_label(self, data, smoothing=False):
        if self.num_classes==2:
            return torch.Tensor(data.y).long()
        else:
            return F.one_hot(torch.tensor(data.y), self.num_classes)
    
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
        else: # self.num_classes=2
            pred_class = logits.argmax(dim=-1)            
            pred_probs = torch.softmax(logits, dim=-1)
        return pred_class, pred_probs

    def forward(self, data):
        
        ### prepare data
        features = data.x
        sites = data.site        
        batch = data.batch
        edge_index, edge_weight = data.edge_index, data.edge_attr
        if not tgutils.contains_self_loops(edge_index): 
            edge_index, edge_weight = tgutils.add_self_loops(edge_index=edge_index, edge_attr=edge_weight, fill_value=1)

        ### Feedforward
        hidden = self.gnn_embedding(x=features, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        target_logits = self.target_classifier(hidden).squeeze(dim=-1)   #logits = out.squeeze(dim=-1)
        target_labels = self.get_label(data).type_as(features)
        target_loss = self.target_loss_module(target_logits, target_labels)
        
        if self.grl:
            domain_logits = self.domain_classifier(hidden).squeeze(dim=-1)
            sites = self.get_site(data).type_as(features)
            domain_loss = self.domain_loss_module(domain_logits, sites)
            if self.current_epoch < round(self.epochs*0.5):
                loss = target_loss
                self.domain_classifier.requires_grad_(False)
            else: 
                self.domain_classifier.requires_grad_(True)
                loss = target_loss + domain_loss
        else:
            loss = target_loss
        #print(pred_probs)
        #print(labels)
        #print(loss)
        if self.num_classes>2: target_labels = target_labels.argmax(dim=1)
        return loss, target_loss, target_logits, target_labels, hidden
        
    def training_step(self, batch, batch_idx):
        loss, target_loss, _, _, _ = self.forward(batch) # return loss, target_loss, target_logits, labels, hidden
        self.log('train_loss', loss, batch_size=self.batch_size, sync_dist=True)
        self.log('target_loss', target_loss, batch_size=self.batch_size, sync_dist=True)
        self.log('domain_loss', loss - target_loss, batch_size=self.batch_size, sync_dist=True)
        return loss

    def training_epoch_end(self, outputs):
        
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        _, loss, logits, labels, _ = self.forward(batch)
        return {"loss": loss, "labels": labels, "logits":logits}

    def validation_epoch_end(self, outputs):
        metrics = ['roc', 'acc', 'avp', 'pr', 'rc']
        avg_loss = torch.concat([x["loss"].unsqueeze(0) for x in outputs]).mean()
        labels = torch.concat([x[f"labels"] for x in outputs]).squeeze()
        logits = torch.concat([x[f"logits"] for x in outputs]).squeeze()
        self.log(f"val_loss", avg_loss)
        for metric in metrics:
            metric_value = self.calculate_metric(metric, labels, logits)            
            self.log(f"val_{metric}", metric_value)

    def test_step(self, batch, batch_idx):
        _, loss, logits, labels, _ = self.forward(batch) # return loss, target_loss, target_logits, labels, hidden
        return {"loss": loss, "labels": labels, "logits":logits}

    def test_epoch_end(self, outputs):

        metrics = ['roc', 'acc', 'avp', 'pr', 'rc']
        avg_loss = torch.concat([x[f"loss"].unsqueeze(0) for x in outputs]).mean()
        labels = torch.concat([x[f"labels"] for x in outputs]).squeeze()
        logits = torch.concat([x[f"logits"] for x in outputs]).squeeze()
        self.log(f"test_loss", avg_loss)
        for metric in metrics:
            metric_value = self.calculate_metric(metric, labels, logits)            
            self.log(f"test_{metric}", metric_value)

    def predict_step(self, batch, batch_idx):
        _, _, logits, labels, embedding = self.forward(batch) # return loss, target_loss, target_logits, labels, hidden
        pred_class, pred_probs = self.get_pred(logits, self.num_classes)
        subject_id = self.get_subject_id(batch)
        return labels.detach().cpu().numpy(), pred_class.detach().cpu().numpy(), pred_probs.detach().cpu().numpy(), logits.detach().cpu().numpy(), embedding.detach().cpu().numpy(), subject_id
    
    def calculate_metric(self, metric, labels, logits):
        labels = labels.long()
        pred_class, pred_probs = self.get_pred(logits, self.num_classes)
        # print("metric:", metric)
        # print("label:", labels)
        # print("logit:", logits.shape)
        # print("pred_class:", pred_class)
        # print("pred_probs:", pred_probs)
        if self.num_classes == 2:
            if metric=='roc': return self.aucroc_score(pred_probs, labels) #roc_auc_score(labels, pred_probs)
            if metric=='avp': return self.avg_precision_score(pred_probs, labels)  #average_precision_score(labels, pred_probs) 
            if metric=='acc': return self.accuracy_score(pred_probs, labels) #accuracy_score(labels, pred_class)
            if metric=='pr': return  self.precision_score(pred_class, labels) #precision_score(labels, pred_class)
            if metric=='rc': return  self.recall_score(pred_class, labels) #recall_score(labels, pred_class)
        else:
            if metric=='roc': return self.aucroc_score(pred_probs, labels) #roc = roc_auc_score(labels, pred_probs)
            if metric=='avp': return self.avg_precision_score(pred_probs, labels)  #avp = average_precision_score(labels, pred_probs, average='macro') 
            if metric=='acc': return self.accuracy_score(pred_class, labels) #accuracy_score(labels.argmax(axis=0)  , pred_class)
            if metric=='pr': return  self.precision_score(pred_class, labels) #precision_score(labels.argmax(axis=0)  , pred_class, average='macro')
            if metric=='rc': return  self.recall_score(pred_class, labels) #recall_score(labels.argmax(axis=0)  , pred_class, average='macro')
        