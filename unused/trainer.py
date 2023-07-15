'''
Deep learning based burn severity prediction using remote sensing data
: Trainer setup (Set model architecture and hyperparameters, accuracy assessment metrics)
- Author: Minho Kim (2023)
'''

# Pytorch-Lightning
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

import segmentation_models_pytorch as smp
# from segmentation_models_pytorch import UnetPlusPlus
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, TverskyLoss
from segmentation_models_pytorch.metrics import get_stats, iou_score, accuracy, precision, recall, f1_score

from dataloader import *

# Input parameters
if not torch.cuda.is_available():
    device=torch.device("cpu")
    print("Current device:", device)
else:
    device=torch.device("cuda")
    print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))

class SegmentationModel(pl.LightningModule):
    def __init__(self,arch='unet',enc_name='none', loss=None, encoder_depth=5, decoder_channels=[256,128,64,32,16], in_channels=6,n_classes=5, lr=0.0001):
        super().__init__()

        if enc_name == 'none':
            self.model = smp.create_model(arch,
                    in_channels = in_channels,
                    activation="softmax",
                    classes = n_classes).to(device)
        elif enc_name == 'unet':
            self.model = smp.create_model(arch,
                        encoder_name = enc_name,
                        encoder_depth = encoder_depth,
                        in_channels = in_channels,
                        encoder_weights = None,
                        activation="softmax",
                        decoder_channels = decoder_channels,
                        classes = n_classes).to(device)
        else:
            self.model = smp.create_model(arch,
                        encoder_name = enc_name,
                        in_channels = in_channels,
                        encoder_weights = None,
                        activation="softmax",
                        classes = n_classes).to(device)

        # Set loss function
        if loss ==  1:
            self.criterion = DiceLoss(mode="multiclass", from_logits=True)
        elif loss == 2:
            self.criterion = FocalLoss(mode="multiclass")
        elif loss == 3:
            self.criterion = TverskyLoss(mode="multiclass", from_logits=True)

        self.loss=loss
        self.n_classes = n_classes
        self.lr = lr

    def forward(self, inputs, targets=None):
        
        outputs = self.model(inputs)
        if targets is not None:
            if self.loss == 2: # Only for focal loss
                targets = targets.squeeze(1)
            loss = self.criterion(outputs, targets.long()).to(device)
            # targets_cpu = targets.cpu().detach().numpy()
            # outputs_cpu = outputs.cpu().detach().numpy()
            # tp, fp, fn, tn = get_stats(np.expand_dims(np.argmax(outputs_cpu, axis=1),1).astype(int), np.expand_dims(targets_cpu, 1), mode='multiclass', num_classes=n_classes)            
            outs = torch.argmax(outputs, 1)
            tp, fp, fn, tn = get_stats(outs, targets.long().squeeze(1), mode='multiclass', num_classes=self.n_classes)
            
            metrics = {
                "Accuracy": accuracy(tp, fp, fn, tn, reduction="micro-imagewise"),
                "IoU": iou_score(tp, fp, fn, tn, reduction="micro-imagewise"),
                "Precision": precision(tp, fp, fn, tn, reduction="micro-imagewise"),
                "Recall": recall(tp, fp, fn, tn, reduction="micro-imagewise"),
                "F1score": f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
            }
            return loss, metrics, outputs
        else: 
            return outputs

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        return [opt], [sch]

    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks=masks.long()

        loss, metrics, outputs = self(images, masks)
        self.log_dict({
            "train/Loss": loss,
            "train/IoU": metrics['IoU'],
            "train/Accuracy": metrics['Accuracy'],
            "train/Precision": metrics['Precision'],
            "train/Recall": metrics['Recall'],
            "train/F1score": metrics['F1score']
        }, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        masks=masks.long()
        
        loss, metrics, outputs = self(images, masks)
        self.log_dict({
            "val/Loss": loss,
            "val/IoU": metrics['IoU'],
            "val/Accuracy": metrics['Accuracy'],
            "val/Precision": metrics['Precision'],
            "val/Recall": metrics['Recall'],
            "val/F1score": metrics['F1score']
        }, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch

        loss, metrics, outputs = self(images, masks)
        self.log_dict({
            "test/Loss": loss,
            "test/IoU": metrics['IoU'],
            "test/Accuracy": metrics['Accuracy'],
            "test/Precision": metrics['Precision'],
            "test/Recall": metrics['Recall'],
            "test/F1score": metrics['F1score']
        }, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, train_x, train_y, val_x, val_y, test_x, test_y, augmentations, seed, batch_size):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y
        self.batch_size = batch_size
        self.seed = seed
        self.augmentations = augmentations

    def setup(self, stage=None):
        self.train_dataset = tensorDataset(self.train_x, self.train_y, augmentations=self.augmentations, seed=self.seed)
        self.val_dataset = tensorDataset(self.val_x, self.val_y, augmentations=self.augmentations, seed=self.seed)
        self.test_dataset = tensorDataset(self.test_x, self.test_y, augmentations=self.augmentations, seed=self.seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size//2, drop_last=True, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size//2, drop_last=True, shuffle=False, num_workers=4)


def checkpoint_params(dirpath="checkpoints",filename="best-checkpoint"):

    # Set checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        save_top_k=1,
        verbose=True,
        monitor="val/Loss",
        mode="min"
    )
    return checkpoint_callback

def log_params(dirpath="lightning_logs",logname="log",patience=15):

    logger = CSVLogger(dirpath, name=logname)
    early_stopping_callback = EarlyStopping(monitor="val/Loss", mode="min", patience=patience)

    return logger, early_stopping_callback