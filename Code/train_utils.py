# * 1.0 IMPORTS ----------------------------------------------------------------------------------------------------------------------------------------------
import os
import json
import yaml

# PyTorch
import torch
import torch.nn    as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau


# * 2.0 LOAD CONFIG FILE -------------------------------------------------------------------------------------------------------------------------------------
def load_config_file(file_path):
    """
    This function reads a YAML configuration file from the specified path. It safely loads the 
    contents of the file into a Dictionary.

    Parameters:
        file_path (str): The path to the configuration file.

    Returns:
        dict: The contents of the configuration file as a dictionary. If the file is empty or 
              has invalid YAML, an empty dictionary is returned.
    """

    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Load the file content into a dictionary
        config = yaml.safe_load(file)

    return config


# * 3.0 SAVE MODEL AND PARAMETERS ----------------------------------------------------------------------------------------------------------------------------
def save_model_and_parameters(model, config, mode):
    """
    This function saves a given model along with its configuration parameters based on the
    specified mode. The mode determines the directory structure under which the model and
    configuration are saved. Supported modes include 'classification', 'regression',
    'multi_task_learning', and 'multi_task_learning_grid'.

    Parameters:
        model (torch.nn.Module): The model to be saved.
        config (dict)          : A dictionary containing configuration parameters for the model.
        mode (str)             : A string specifying the mode/type of the model. Accepted
                                 values are 'classification', 'regression', 'multi_task_learning',
                                 and 'multi_task_learning_grid'.
    """

    # Determine the directory based on the mode
    if mode == 'classification':
        save_dir = 'saved_models/classification_models'
    if mode == 'regression':
        save_dir = 'saved_models/regression_models'
    if mode == 'multi_task_learning':
        save_dir = 'saved_models/multi_task_learning'
    if mode == 'multi_task_learning_grid':
        save_dir = 'saved_models/MTL_Grid_Data'

    # Create the main save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a subdirectory for the model based on its backbone name
    model_dir = os.path.join(save_dir, config['backbone_name'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model to a file in the appropriate directory
    model_path = os.path.join(model_dir, f'{config["run_name"]}.pth')
    torch.save(model.state_dict(), model_path)

    # Save the configuration parameters to a YAML file
    config_path = os.path.join(model_dir, f'{config["run_name"]}.yaml')
    with open(config_path, 'w') as config_file:
        yaml.dump(config, config_file)


# * 4.0 CHOOSE TRAIN AND VALIDATION TRANSFORMATIONS ----------------------------------------------------------------------------------------------------------
def choose_tain_val_transformations(config):
    """
    This function generates two sets of transformations: one for training and another for 
    validation. The training transformations include various augmentations based on the 
    configuration parameters provided. The validation transformations are simpler,
    consisting mainly of resizing, tensor conversion, and normalization.

    Parameters:
        config (dict): A dictionary containing configuration parameters. The configuration 
                       should include keys like 'input_image_height', 'input_image_width', 
                       and flags for different augmentations.

    Returns:
        tuple: A tuple containing two elements:
               - train_transformations: A torchvision.transforms.Compose object for training.
               - val_transformations: A torchvision.transforms.Compose object for validation.
    """

    # Define the training transformations list
    transformation_list = [transforms.Resize((config['input_image_height'], config['input_image_width']))]

    # Augmentations (if routines are used to keep track when each augmentation method is used)
    if config['use_color_jitter']:
        transformation_list.append(transforms.ColorJitter(
            brightness = ( 0.000, 1.700) if config['use_jitter_brightness'] else None,
            contrast   = ( 0.000, 3.000) if config['use_jitter_contrast'  ] else None,
            saturation = ( 0.000, 2.500) if config['use_jitter_saturation'] else None,
            hue        = ( 0.085, 0.085) if config['use_jitter_hue'       ] else None
            ))
    if config['use_gaussian_blur']:
        transformation_list.append(transforms.GaussianBlur(
            kernel_size=(3, 3),
            sigma=(0.1, 5.0)
            ))
    if config['use_random_perspective']:
        transformation_list.append(transforms.RandomPerspective(
            distortion_scale=0.6,
            p=1.0
            ))
        
    # Standard transformations added to the list
    transformation_list.append(transforms.ToTensor())
    transformation_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                    std  = [0.229, 0.224, 0.225]
                                                    ))
    
    # Define the composed transformation for training
    train_transformations = transforms.Compose(transformation_list)

    # Define the validation transformations (No augmentation is used here)
    val_transformations = transforms.Compose([
        transforms.Resize((config['input_image_height'], config['input_image_width'])),  # Resize the image
        transforms.ToTensor(),                              # Convert the image to a tensor
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225])  # Normalize the image
    ])

    return train_transformations, val_transformations


# * 5.0 CHOOSE LOSS FUNCTION ---------------------------------------------------------------------------------------------------------------------------------
def choose_loss_function(loss_function_name):
    """
    This function takes the name of a loss function as input and returns the corresponding 
    PyTorch loss function object. Supported loss functions include L1 Loss, Mean Squared Error 
    Loss (MSELoss), Mean Absolute Error Loss (MAELoss, which is equivalent to L1Loss), and 
    Cross Entropy Loss.

    Parameters:
        loss_function_name (str): A string representing the name of the desired loss function. 
                                  Accepted values are 'L1Loss', 'MSELoss', 'MAELoss', and 
                                  'CrossEntropyLoss'.

    Returns:
        torch.nn.Module: An instance of the requested loss function.
    """

    # L1 Loss (Mean Absolute Error Loss)
    if (loss_function_name == 'L1Loss') or (loss_function_name == 'CrossEntropyLoss'):
        return nn.L1Loss()

    # Mean Squared Error Loss
    elif loss_function_name == 'MSELoss':
        return nn.MSELoss()
    
    # Cross Entropy Loss
    elif loss_function_name == 'MAELoss':
        return nn.L1Loss()
    
    else:
        raise ValueError(f"Unrecognized loss function name: {loss_function_name}")


# * 6.0 CHOOSE OPTIMIZER -------------------------------------------------------------------------------------------------------------------------------------
def choose_optimizer(model, optimizer_name, learning_rate):
    """
    This function takes a model, the name of an optimizer, and a learning rate as inputs. It 
    returns the corresponding PyTorch optimizer object initialized with the model's parameters 
    and the specified learning rate. Supported optimizers include Adam, Stochastic Gradient 
    Descent (SGD), RMSprop, and Adagrad.

    Parameters:
        model (torch.nn.Module): The model for which the optimizer will be used.
        optimizer_name (str)   : A string representing the name of the desired optimizer. 
                                 Accepted values are 'Adam', 'SGD', 'RMSprop', and 'Adagrad'.
        learning_rate (float)  : The learning rate to be used with the optimizer.

    Returns:
        torch.optim.Optimizer: An instance of the requested optimizer.
    """

    # Adam Optimizer
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate)

    # Stochastic Gradient Descent (SGD) Optimizer
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate)

    # RMSprop Optimizer
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate)

    # Adagrad Optimizer
    elif optimizer_name == 'Adagrad':
        return optim.Adagrad(model.parameters(), lr=learning_rate)
    
    else:
        raise ValueError(f"Unrecognized optimizer name: {optimizer_name}")

# * 7.0 CHOOSE LEARNING RATE SCHEDULER -----------------------------------------------------------------------------------------------------------------------
def choose_sceduler(optimizer, scheduler_name):
    """
    This function creates a scheduler for adjusting the learning rate based on the specified 
    name. The scheduler is initialized with the given optimizer. Supported schedulers include 
    'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', and 'ReduceLROnPlateau'.

    Parameters:
        optimizer (torch.optim.Optimizer): The optimizer to which the scheduler will be applied.
        scheduler_name (str)             : A string representing the name of the desired scheduler. 
                                           Accepted values are 'CosineAnnealingLR', 
                                           'CosineAnnealingWarmRestarts', and 'ReduceLROnPlateau'. 
                                           If None is passed, no scheduler is returned.

    Returns:
        torch.optim.lr_scheduler: An instance of the requested scheduler, or None if no 
                                  scheduler name is provided.
    """

    # No Scheduler
    if scheduler_name is None:
        return None
    
    # Cosine Annealing Learning Rate Scheduler
    elif scheduler_name == 'CosineAnnealing':
        return CosineAnnealingLR(optimizer, T_max=5, eta_min=0.000001)
    
    # Cosine Annealing with Warm Restarts Scheduler
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        return CosineAnnealingWarmRestarts(optimizer, T_max=5, T_mult=0.8, eta_min=0.000001)

    # Reduce Learning Rate on Plateau Scheduler
    elif scheduler_name == 'ReduceOnPlateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    else:
        raise ValueError(f"Unrecognized scheduler name: {scheduler_name}")



if __name__ == '__main__':
    pass
