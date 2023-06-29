'''
Deep learning based burn severity prediction using remote sensing data
: Main script
- Author: Minho Kim (2023)
'''

# Load libraries
import os, glob, warnings, datetime, argparse
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore') # Ignore warning messages

# Visualization
import matplotlib.pyplot as plt

# Deep learning library (PyTorch)
import torchvision.transforms as T
import pytorch_lightning as pl

# Custom
from utils import *
from preprocess import *
from dataloader import *
from trainer import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--patch_size', default=128, type=int, help='Patch size')
    parser.add_argument('--threshold', default=0.5, type=float, help='Threshold of zero-values to consider per image patch in input dataset')
    parser.add_argument('--seed', default=42, type=int, help='Seed for reproducibility')
    
    parser.add_argument('--loss_function', default=1, type=int, help='1: Dice Loss | 2: Focal Loss | 3: Tversky Loss')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--encoder_depth', default=5, type=int, help='Encoder depth')
    parser.add_argument('--decoder_channels', default=64, type=int, help='Decoder channel depth')
    parser.add_argument('--crop_size', default=32, type=int, help='Patch size for random cropping (augmentations')
    parser.add_argument('--train_fraction', default=0.2, type=float, help='Proportion of train dataset samples (eg. 0.2 = 80%)')
    parser.add_argument('--test_fraction', default=0.25, type=float, help='Proportion of test dataset samples')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--patience', default=20, type=int, help='Number of epochs to monitor for early stopping')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--dec_name', default='unet', type=str, help='Decoder model')
    parser.add_argument('--enc_name', default='efficentnet-b0', type=str, help='Encoder model')
    parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs')
 
    args = parser.parse_args()    

    torch.set_num_threads(4)

    # Load paths
    BASEPATH = "/home/minho/fires/caltrans/data"
    data_path = '/home/minho/fires/CE263N/old_data'
    # Input paths
    s2_path = sorted(glob.glob(os.path.join(data_path, '*s2_aoi*.tif')))
    dem_path = sorted(glob.glob(os.path.join(data_path, '*dem*.tif')))
    slope_path = sorted(glob.glob(os.path.join(data_path, '*slope*.tif')))
    lulc_path = sorted(glob.glob(os.path.join(data_path, '*lulc*.tif')))
    burn_severity_path = sorted(glob.glob(os.path.join(data_path, '*fire*.tif'))) # Label path

    # Input parameters
    patch_size = args.patch_size
    threshold = args.threshold
    _seed = args.seed

    # DL parameters
    mode = 'train'
    loss_function = args.loss_function
    crop_size = args.crop_size
    _batch_size = args.batch_size 
    _train_fraction = args.train_fraction
    _test_fraction = args.test_fraction
    _epochs = args.epochs
    _patience = args.patience
    _lr = args.lr
    gpu_devices = args.gpus
    dec_name = args.dec_name
    enc_name = args.enc_name
    encoder_depth = args.encoder_depth
    dim = args.decoder_channels

    if gpu_devices > 1:
        strategy = 'ddp_find_unused_parameters_true'
    else: 
        strategy = None
    
    # Augmentations
    augmentations = T.Compose([
        # T.RandomResizedCrop(crop_size),
        T.RandomHorizontalFlip(),  # Randomly flip horizontally
        T.RandomVerticalFlip(),    # Randomly flip vertically
    ])

    # Create datasets
    train_x, train_y, val_x, val_y, test_x, test_y, weights, in_channels, out_channels = preprocess(s2_path, dem_path, slope_path, lulc_path, burn_severity_path, patch_size, mode, threshold, _train_fraction, _test_fraction, _seed)

    # Set model
    decoder_channels = np.ones(encoder_depth) * dim     # Create decoder channels (based on largest "dim" size from args)
    decoder_channels = [int(decoder_channels[i+1] / 2**(i+1)) for i in range(len(decoder_channels)-1)]

    print("IN CHANNELS : ", in_channels)
    if args.enc_name == "None":
        model = SegmentationModel(arch=dec_name,loss=loss_function,encoder_depth=encoder_depth, decoder_channels=decoder_channels,in_channels=in_channels,n_classes=out_channels, lr=_lr)
    else:                
        model = SegmentationModel(arch=dec_name,loss=loss_function,enc_name=enc_name,encoder_depth=encoder_depth, decoder_channels=decoder_channels,in_channels=in_channels,n_classes=out_channels, lr=_lr)

    # Set directory to save
    _timename = "log_" + datetime.datetime.now().strftime('%Y%m%d')+"_"+enc_name+"_"+dec_name
    timename = str(_timename)
    if not os.path.exists(os.path.join('lightning_logs', timename)):    
        os.makedirs(os.path.join('lightning_logs', timename))

    checkpoint_callback = checkpoint_params(dirpath="checkpoints",filename=timename)
    logger, early_stopping_callback = log_params(dirpath="lightning_logs",logname=timename,patience=_patience)

    # Setup trainer
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=31,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=_epochs,
        accelerator="gpu",
        devices=gpu_devices,
        num_sanity_val_steps=0,
        strategy=strategy
        )

    # Create data module for PL
    data_module = SegmentationDataModule(train_x, train_y, val_x, val_y, test_x, test_y, augmentations=augmentations, seed=_seed, batch_size=_batch_size)
    data_module.setup()

    # Train
    trainer.fit(model, data_module)

    # Test
    trainer.test(model, data_module)

    # Plot
    metrics = pd.read_csv("./lightning_logs/" + timename + "/version_0/metrics.csv")
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    names = ['Loss', 'IoU', 'Accuracy', 'Precision', 'Recall', 'F1score']

    for axis, name in zip(axes, names):
        axis.plot(metrics[f'train/{name}'].dropna())
        axis.plot(metrics[f'val/{name}'].dropna())
        axis.set_title(f'{name}: Train/Val')
        axis.set_ylabel(name)
        axis.set_xlabel('Epoch')
        ax1.legend(['training', 'validation'], loc="upper right")