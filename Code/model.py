# * 1.0 IMPORTS ----------------------------------------------------------------------------------------------------------------------------------------------
import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict


# * 2.0 DEFINE HYDRANET ARCHITECTURE -------------------------------------------------------------------------------------------------------------------------
class HydraNet(nn.Module):
    def __init__(self, labels_type, backbone_name, use_depth, pretrained):
        """
        Initialize a HydraNet model for active face perception.

        Parameters:
            labels_type (str)  : The type of labels for the model. Can be 'soft' or 'hard'.
            backbone_name (str): The backbone architecture for the feature extraction. Supported options: 'resnet18',
                                 'resnet34', 'resnet50', 'mobilenet', 'efficientnet-b0', 'efficientnet-b1',
                                 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4'.
            use_depth (bool)   : Whether to include depth head in the model.

        Raises:
            ValueError: If the provided arguments are invalid.
        """

        super().__init__()

        self.labels_type   = labels_type
        self.backbone_name = backbone_name
        self.use_depth     = use_depth
        self.pretrained    = pretrained

        # Define the number of output neurons for the direction heads (horizontal and vertical)
        if   self.labels_type.lower() == 'soft':
            self.n_output_neurons = 2
        elif self.labels_type.lower() == 'hard':
            self.n_output_neurons = 3
        else:
            raise ValueError("Invalid labels type. Use 'soft', or 'hard'.")

        # Set up Backbone Net
        self.build_backbone()

        # Define Direction Heads
        self.net.head_horizontal = nn.Sequential(
            OrderedDict([('linear', nn.Linear(self.n_features,128)),
                         ('relu'  , nn.ReLU()),
                         ('final' , nn.Linear(128, self.n_output_neurons))]))

        self.net.head_vertical  = nn.Sequential(
            OrderedDict([('linear', nn.Linear(self.n_features,128)),
                         ('relu'  , nn.ReLU()),
                         ('final' , nn.Linear(128, self.n_output_neurons))]))

        # Depth Head
        if self.use_depth: # (this must be modified for other model)
            # Adding the depth head (decoder) here:
            self.net.head_depth = nn.Sequential(
                nn.ConvTranspose2d(1280, 256, kernel_size=4, stride=2, padding=1),  # Upsampling 1280 is selected specifically for the efficientnet-b0 architecture
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # Upsampling
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # Upsampling
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),     # Upsampling
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),      # Final layer to output 3 channels
                nn.Sigmoid()  # Sigmoid activation to limit output values between 0 and 1
            )

    def build_backbone(self):
        """
        Build the backbone network based on the specified architecture.
        """

        # ResNet family models
        if self.backbone_name.lower().startswith('resnet'):
            # Define backbone model
            if   self.backbone_name.lower() == 'resnet18':
                if self.pretrained:
                    self.net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                else:
                    self.net = models.resnet18() # By default, no pre-trained weights are used
            elif self.backbone_name.lower() == 'resnet34':
                if self.pretrained:
                    self.net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
                else:
                    self.net = models.resnet34() # By default, no pre-trained weights are used
            elif self.backbone_name.lower() == 'resnet50':
                if self.pretrained:
                    self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                else:
                    self.net = models.resnet50() # By default, no pre-trained weights are used  
            else:
                raise ValueError("Invalid ResNet backbone name. Use 'resnet18', 'resnet34', or 'resnet50'.")
            # Extract the number of outoput features
            self.n_features = self.net.fc.in_features
            self.net.fc = nn.Identity()
        
        # MobileNet
        elif self.backbone_name.lower() == 'mobilenet':
            # Define backbone model
            self.net = models.mobilenet_v2(pretrained=self.pretrained)
            # Extract the number of outoput features
            self.n_features   = self.net.classifier[1].in_features
            self.net.classifier[1] = nn.Identity()
        
        # EfficientNet family models
        elif self.backbone_name.lower().startswith('efficientnet-b'):
            if self.pretrained:
                # Define backbone model
                self.net = EfficientNet.from_pretrained(self.backbone_name.lower())
            else:
                # Define backbone model
                self.net = EfficientNet.from_name(self.backbone_name.lower())
            # Extract the number of outoput features
            self.n_features = self.net._fc.in_features
            self.net._fc    = nn.Identity()
        else:
            raise ValueError("Invalid backbone name. Use 'resnet18', 'resnet34', 'resnet50', 'mobilenet', \
                             'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', or 'efficientnet-b4'.")

    def register_hooks(self):
        """
        Register a forward hook to capture the output of the last convolutional layer.
        """

        # Forward Hook
        # The forward hook is used to capture the output of the last convolutional layer of the backbone network
        # This is necessary for the depth head in case 'use_depth' is True
        def forward_hook(module, input, output):
            self.last_conv_output = output

        # Register the forward hook on the last conv layer (this must be modified for other model)
        self.hook_handle = self.net._conv_head.register_forward_hook(forward_hook)

    def forward(self, x):
        """
        Forward pass of the HydraNet model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple of horizontal_output (torch.Tensor), vertical_output (torch.Tensor), and depth_output (torch.Tensor).
        """
        
        # Register hooks (if not already registered)
        if not hasattr(self, 'hook_handle'):
            self.register_hooks()

        # Feature extraction
        features = self.net(x)  # This will also update self.last_conv_output via the forward hook
        
        # Direction outputs
        horizontal_output = self.net.head_horizontal(features)
        vertical_output   = self.net.head_vertical(features)
        
        # Depthmap
        if self.use_depth:
            # Use the output of the last conv layer as input to the depth head
            depth_output = self.net.head_depth(self.last_conv_output)
        else:
            depth_output = None
        
        # Optionally, remove the hook if it's no longer needed
        # self.hook_handle.remove()
        
        return horizontal_output, vertical_output, depth_output
    
    

if __name__ == '__main__':
    pass
