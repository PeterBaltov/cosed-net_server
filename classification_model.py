import torch
from torch import nn, optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import *
import pytorch_lightning as pl
from torchmetrics import (Accuracy, CohenKappa, AUC, ConfusionMatrix, F1Score, Recall, Specificity, Precision)
import timm


import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (ModelCheckpoint, EarlyStopping)
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.loggers import TensorBoardLogger


from torchmetrics import (Accuracy, CohenKappa, AUC, ConfusionMatrix, F1Score, Recall, Specificity, Precision)

import timm
from torchinfo import summary

from efficientnet_pytorch import EfficientNet
from pprint import pprint


class ClassificationModel(pl.LightningModule):
    """
    This class creates the neural network model by also defining the loss function and the optimizer
    for the task of Diabetic Retinopathy.
    Addionally, this class provides additional functions that are used to train, validate and test the model.
    Futhermore, this class has combined the aforementioned functions into one function called `shared_step` and `shared_epoch_end
    by also calculating the diffrent performance metrics.
    Finally, this class has a function called `build_model` that is used to create the model by further spliting the
    feature extraction and classification parts.
    
    Args:
        timm_model_name: The name of the model that is used for feature extraction.
        pretrained: A boolean value that indicates whether the model is pretrained or not.
        num_classes: The number of classes that the model is trained on.
        batch_size: The batch size of the model.
        data_ratios: The ratios of the data that is used for future implementation of the Focal Loss Function.
        lr: The learning rate of the model.
        lr_scheduler_gamma: The gamma parameter of the learning rate scheduler.
        **kwargs: Additional arguments that are used for the model.

    Returns:
        model (nn.Module): The neural network model based on the Pytorch nn class.
    """

    def __init__(self, 
                 timm_model_name, 
                 pretrained, 
                 num_classes,
                 batch_size,
                 data_ratios = [],
                 lr = 1e-3,
                 lr_scheduler_gamma = 1e-1,
                 **kwargs):
        super().__init__()
    
        self.save_hyperparameters()
        self.timm_model_name = timm_model_name
        self.pretrained = pretrained
        self.num_classes= num_classes
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma,
        self.batch_size=batch_size
        
        # Preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(f"tu-{timm_model_name}")
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Initialization of different metrics
        # The metrcis used to evaluate the models are:
        # 1. Accuracy
        self.acc = Accuracy(num_classes=self.num_classes)
        # 2. Cohen Kappa
        self.kappa = CohenKappa(num_classes=self.num_classes, weights='quadratic')
        # 3. F1 Score
        self.f1_score = F1Score(num_classes=self.num_classes, average="macro")
        # 4. Weighted F1 Score
        self.weighted_f1 = F1Score(num_classes=self.num_classes, average="weighted")
        # 5. Precision
        self.precision_score = Precision(num_classes=self.num_classes, average="macro")
        # 6. Recall
        self.recall = Recall(num_classes=self.num_classes, average="macro")
        # 7. Specificity
        self.specificity = Specificity(num_classes=self.num_classes, average="macro")
        
        # Loss Function
        # the loss function is Cross Entropy Loss but for
        # further experimentation, the Focal Loss Function is established
        self.focal_loss = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            # alpha=torch.tensor(data_ratios, dtype=torch.float32),
            # gamma=0.25,
            reduction='mean',
            force_reload=False
        )
        self.build_model()
        
    def build_model(self):
        """
        This function is used to create the model by creating the feature extraction part and the classification part.
        """
        # Feature Extractor:
        self.feature_extractor = timm.create_model(self.timm_model_name, pretrained=self.pretrained, num_classes=0)

        # Classifier:
        self.fc = nn.Sequential(
            nn.Linear(in_features=1280, out_features=5, bias=True)
        )


    def forward(self, data):
        """
        This function is used to forward the data through the model for predictions.
        Args:
            data: The data that is used for the predictions.
        Returns:
            logits (torch.Tensor): The logits of the model.
        """
        # 1. Normalize image here
        data = (data - self.mean) / self.std
        
        # 2. Extract Features
        preds = self.feature_extractor(data)
        
        # 3. Classify Features (returns predictions/logits):
        preds = self.fc(preds)

        return preds

    def shared_step(self, batch, stage=None):
        """
        This function is used to calculate the loss and also
        store the predictions and ground truth labels in a dictionary format
        for a given step so that further evaluate is made available.
        
        Args:
            batch: The batch of data that is used for the training.
            stage: The stage of either training, validation or testing.
        Returns:
            loss (torch.Tensor): The loss of the model.
            predictions (dict): The predictions of the model.
            ground_truth (dict): The ground truth labels of the model.

        """
        # extracting the individual images and ground truth labels
        #  from the selected batch
        image, label, _= batch 

        # Checking the dimentations and shape of the image
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        
        # Forward pass
        logits  = self.forward(image) #.squeeze()   
        
        # Formaing the predictions to be used in the loss function calculations
        # This is done with the applicability of ArgMax operation
        preds = logits.argmax(1).long()
        
        # Calculating the loss
        loss = self.focal_loss(logits, label)
        
        # Returning the loss, the predictions and the ground truth labels
        return {
            "loss": loss,
            "preds": preds,
            "label": label
        }
    
    def shared_epoch_end(self, outputs, stage):
        """
        This function is used to calculate the average loss 
        and average metrics at the end of each epoch.

        Args: 
            outputs: The outputs of the model.
            stage: The stage of either training, validation or testing.
        """
        # All of the predictions and ground truth labels are concatinated from each step
        preds = torch.cat([x["preds"] for x in outputs])
        label = torch.cat([x["label"] for x in outputs])

        # Calculating the average loss
        loss = torch.stack([x["loss"] for x in outputs])
        mean_loss = torch.mean(loss[-100:])

        # Calculating the average metrics
        self.kappa(preds, label)
        self.acc(preds, label)
        self.f1_score(preds, label)
        self.weighted_f1(preds, label)
        self.precision_score(preds, label)
        self.recall(preds, label)
        self.specificity(preds, label)
        
        # Logging and printing the average metrics
        self.log(f'{stage}_loss', mean_loss, prog_bar=True)
        self.log(f'{stage}_qwkappa', self.kappa, prog_bar=True)
        self.log(f"{stage}_acc",  self.acc, prog_bar=True)
        self.log(f'{stage}_f1_score', self.f1_score, prog_bar=True)
        self.log(f'{stage}_weighted_f1', self.weighted_f1, prog_bar=True)
        self.log(f'{stage}_precision', self.precision_score, prog_bar=True)
        self.log(f'{stage}_recall', self.recall, prog_bar=True)
        self.log(f'{stage}_specificity', self.specificity, prog_bar=True)

    def training_step(self, batch, batch_idx):
        """
        This method will be called at every batch.
        It will call the shared_step method to calculate the loss and 
        it substitutes the traditional training step 

        Args: 
            batch (tuple): The batch of data.
            batch_idx (int): The index of the batch.
        Returns:
            dict: The loss and metrics.
        """
        return self.shared_step(batch, "train")  

    def training_epoch_end(self, outputs):
        """
        This method will be called at the end of every epoch.
        It will call the shared_epoch_end method to calculate the average loss and metrics.
        It substitutes the traditional training epoch end.

        Args:
            outputs (dict): The outputs of the model.
        
        Returns:
            dict: The average loss and metrics.
        """
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        """
        This method will be called at every batch.
        It will call the shared_step method to calculate the loss and 
        it substitutes the traditional validation step 

        Args: 
            batch (tuple): The batch of data.
            batch_idx (int): The index of the batch.
        Returns:
            dict: The loss and metrics.
        """
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        """
        This method will be called at the end of every epoch.
        It will call the shared_epoch_end method to calculate the average loss and metrics.
        It substitutes the traditional validation epoch end.

        Args:
            outputs (dict): The outputs of the model.
        
        Returns:
            dict: The average loss and metrics.
        """
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        """
        This method will be called at every batch.
        It will call the shared_step method to calculate the loss and 
        it substitutes the traditional testing step 

        Args: 
            batch (tuple): The batch of data.
            batch_idx (int): The index of the batch.
        Returns:
            dict: The loss and metrics.
        """
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        """
        This method will be called at the end of every epoch.
        It will call the shared_epoch_end method to calculate the average loss and metrics.
        It substitutes the traditional validation epoch end.

        Args:
            outputs (dict): The outputs of the model.
        
        Returns:
            dict: The average loss and metrics.
        """
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        """
        This method will be called to configure the optimizers and the learning rate scheduler.
        Additionally, it will identify the trainable parameters that will be optimized.

        Returns:
            list: optimizer and learning rate scheduler
        """
        # Identifying the trainable parameters
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        
        # Configuring the optimizer
        optimizer = optim.Adam(trainable_parameters, lr=self.lr) #self.parameters()

        # Configuring the learning rate scheduler
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                                 mode='min', 
                                                                                 patience=3,
                                                                                 min_lr=5.0e-05), 
                         "monitor": "train_loss"}
        return [optimizer], [scheduler]