'''
Models
'''

# Model
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tm # Resnet

# device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(encoder=None, decoder=None, in_channels=None, n_classes=None, aux=None, device=None):
    
    # Set pooling, dropout, activation function head
    if aux is not None:
        aux_params=dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=n_classes+1,       # define number of output labels
        )

    # U-Net
    if decoder == "UNET":
        model = smp.Unet(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1)
        if aux is not None: 
            print(aux_params)
            model = smp.Unet(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1, aux_params=aux)
        return model

    # U-Net++
    if decoder == "UNETPLUSPLUS":
        model = smp.UnetPlusPlus(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1)
        if aux is not None: 
            model = smp.UnetPlusPlus(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1, aux_params=aux)
        return model

    # DeepLabV3
    if decoder == "DEEPLABV3":
        model = smp.DeepLabV3(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1)
        if aux is not None: 
            model = smp.DeepLabV3(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1, aux_params=aux)
        return model

    # DeepLabV3+
    if decoder == "DEEPLABV3PLUS":
        model = smp.DeepLabV3Plus(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1)
        if aux is not None: 
            model = smp.DeepLabV3Plus(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1, aux_params=aux)
        return model

    # MANet
    if decoder == "MANET":
        model = smp.MAnet(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1)
        if aux is not None: 
            model = smp.MAnet(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1, aux_params=aux)
        return model

    # FPN
    if decoder == "FPN":
        model = smp.FPN(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1)
        if aux is not None: 
            model = smp.FPN(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1, aux_params=aux)
        return model

    # PSPNet
    if decoder == "PSPNET":
        model = smp.PSPNet(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1)
        if aux is not None: 
            model = smp.PSPNet(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1, aux_params=aux)
        return model

    # LinkNet
    if decoder == "LINKNET":
        model = smp.Linknet(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1)
        if aux is not None: 
            model = smp.Linknet(encoder_name=encoder,encoder_weights=None,in_channels=in_channels,classes=n_classes+1, aux_params=aux)
        return model

### CNN
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, dims):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, dims, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dims)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(dims, dims*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dims*2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(dims*2, dims, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(dims)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(dims, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Decoder
        x = self.conv4(x)
        
        return x

class basic(nn.Module):
    def __init__(self, in_channels, num_classes, dims):
        super(basic, self).__init__()
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims, dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims, self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # nn.Dropout2d(0.5)  # Add dropout with a probability of 0.5
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ResNet50MultiChannel(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super(ResNet50MultiChannel, self).__init__()
        
        # Load the ResNet-50 model without pretrained weights
        resnet = tm.resnet50(pretrained=False)
        
        # Adjust the input layer to accept the desired number of channels
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, :num_input_channels, :, :])
        
        # Use the remaining layers from the original ResNet-50 model
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Modify the last layer to have the desired number of output classes
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_resnet(resnet=50, num_input_channels=None, num_classes=14):

    if resnet == 50:
        model = ResNet50MultiChannel(num_input_channels=num_input_channels, num_classes=num_classes+1)

    return model





