# * 1.0 IMPORTS -----------------------------------------------------------------------------------------------------------------------------------
import os 
import wandb
from playsound import playsound
from tqdm import tqdm

# PyTorch
import torch
from torch.utils.data  import DataLoader

# Custom
from Code.data.dataset import OcclusionRemovalDataset
from Code.model        import HydraNet
from Code.train_utils  import load_config_file, save_model_and_parameters, \
                              choose_tain_val_transformations, choose_loss_function, \
                              choose_optimizer, choose_sceduler


# * 2.0 SET UP  ------------------------------------------------------------------------------------------------------------------------------------
# Config file for Hyperparameters
config = load_config_file('config.yaml')

# Base root for all datasets
dataset_root = '/media/mloufakis/Data/Datasets/Thesis_Datasets/'

train_data_dir = os.path.join(dataset_root, 'grid_simulation_train_samples/')
val_data_dir   = os.path.join(dataset_root, 'grid_simulation_test_samples/' )

# Chek if a correct labeling method was selected and define the label type
if   config['labeling_method'] in ['derivative_slope', 'integral_ratio']:
    config['label_type'] = 'hard'
elif config['labeling_method'] in ['symmetric_sampling', 'estimation_correction']:
    config['label_type'] = 'soft'
else:
    raise ValueError("Wrong labeling method was choosed, select from ['derivative_slope', \
                     'integral_ratio', 'symmetric_sampling', 'estimation_correction']")

# Save the dataset naem to the config file
dataset_name = f'Grid - {config["labeling_method"].replace("_"," ").title()}'
config['dataset_name'] = dataset_name

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device used {device}')
# Save the device to the config file
config['device'] = str(device)

# Log In W&B
wandb.login(key='ebdf1a8322ae518658148ba0182f45bd41656b87')
# Define the project name
wnb_project_name = f'MTL {dataset_name}'


# * 3.0 LOAD DATA ---------------------------------------------------------------------------------------------------------------------------------
# Define the training & validation transformations list
train_transform, val_transform = choose_tain_val_transformations(config)

# Create the training and validation pytorch Datasets
train_dataset = OcclusionRemovalDataset(data_dir                     = train_data_dir,
                                        labeling_method              = config['labeling_method'],
                                        accepted_verification_scores = config['accepted_verification_scores'],
                                        transform                    = train_transform,
                                        averaging_window             = config['averaging_window'],
                                        border_width                 = config['border_width'],
                                        derivative_threshold         = config['derivative_threshold'],
                                        integral_ratio_threshold     = config['integral_ratio_threshold'],
                                        gaussian_decay_window        = config['gaussian_decay_window'])

val_dataset   = OcclusionRemovalDataset(data_dir                     = val_data_dir,
                                        labeling_method              = config['labeling_method'],
                                        accepted_verification_scores = config['accepted_verification_scores'],
                                        transform                    = val_transform,
                                        averaging_window             = config['averaging_window'],
                                        border_width                 = config['border_width'],
                                        derivative_threshold         = config['derivative_threshold'],
                                        integral_ratio_threshold     = config['integral_ratio_threshold'],
                                        gaussian_decay_window        = config['gaussian_decay_window'])

# Save number of training and validation examples
config['training_samples'  ] = len(train_dataset)
config['validation_samples'] = len(val_dataset  )

print()
print(f'Total training   samples: {config["training_samples"  ]}')
print(f'Total validation samples: {config["validation_samples"]}')

# Construct the corresponding dataloaders
train_loader = DataLoader(dataset = train_dataset, batch_size = config['batch_size'], shuffle = True )
val_loader   = DataLoader(dataset = val_dataset  , batch_size = config['batch_size'], shuffle = False)


# * 4.0 DEFINE MODEL ------------------------------------------------------------------------------------------------------------------------------
print('Loading model...')

# Instantiate the model
model = HydraNet(labels_type   = config['label_type'        ],
                 backbone_name = config['backbone_name'     ],
                 use_depth     = config['use_depth_head'    ],
                 pretrained    = config['pretrained_weights'])

# Send the model to the device
model.to(device)

# Store the number of parameters the model contains
config['model_parameters'] = sum(p.numel() for p in model.parameters())

# Define Loss Functions for each head
depth_criterion      = choose_loss_function(config['depth_head_loss_function_name'     ]).cuda(device)
horizontal_criterion = choose_loss_function(config['horizontal_head_loss_function_name']).cuda(device)
vertical_criterion   = choose_loss_function(config['vertical_head_loss_function_name'  ]).cuda(device)

# Define Optimizer
optimizer = choose_optimizer(model, config['optimizer_name'], config['learning_rate'])

# Defeine Scheduler
scheduler = choose_sceduler(optimizer, config['sceduler_name'])


# * 5.0 START THIS RUN ----------------------------------------------------------------------------------------------------------------------------
# Initialize W&B run
wandb.init(project=wnb_project_name, config=config)
# Save the run name
config['run_name'] = wandb.run.name

# Lists to store training and validation metrics
train_depth_losses        = []
train_horizontal_losses   = []
train_vertical_losses     = []
train_total_losses        = []
val_depth_losses          = []
val_horizontal_losses     = []
val_vertical_losses       = []
val_total_losses          = []
# Store the learning rate values
learning_rates            = []

# Initialize EARLY STOPPING metrics
best_loss         = float('inf')  # Initialize with a large value
epochs_since_best = 0

# Train Network
print('Start Training:')
print()
for epoch in range(config['epochs']):


    # * 5.1 TRAINING PHASE ------------------------------------------------------------------------------------------------------------------------
    # Set model to training mode
    model.train()  
    # Initialize epoch metrics
    epoch_depth_loss      = 0.0
    epoch_horizontal_loss = 0.0
    epoch_vertical_loss   = 0.0
    
    # Iterate throught training batches
    for images, depth_maps, horizontal_labels, vertical_labels in tqdm(train_loader, desc=f"Epoch [{epoch + 1:03d}/{config['epochs']}] Training"):

        # Get data to cuda if possible
        images            = images           .to(device=device)
        if config['use_depth_head']:
            depth_maps    = depth_maps       .to(device=device)
        horizontal_labels = horizontal_labels.to(device=device)
        vertical_labels   = vertical_labels  .to(device=device)

        optimizer.zero_grad()

        # Get the models predictions
        horizontal_outputs, vertical_outputs, depth_outputs = model(images)

        # Calculate losses
        if config['use_depth_head']:
            depth_loss  = depth_criterion     (depth_outputs,      depth_maps)
        horizontal_loss = horizontal_criterion(horizontal_outputs, horizontal_labels.argmax(dim=1))
        vertical_loss   = vertical_criterion  (vertical_outputs,   vertical_labels  .argmax(dim=1))
        
        loss = horizontal_loss + vertical_loss
        if config['use_depth_head']:
            loss += depth_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate losses
        if config['use_depth_head']:
            epoch_depth_loss   += depth_loss     .item()
        epoch_horizontal_loss  += horizontal_loss.item()
        epoch_vertical_loss    += vertical_loss  .item()

    # Calculate average loss and accuracy
    train_depth_loss      = epoch_depth_loss      / len(train_loader)
    train_horizontal_loss = epoch_horizontal_loss / len(train_loader)
    train_vertical_loss   = epoch_vertical_loss   / len(train_loader)
    train_total_loss      = (train_horizontal_loss + train_vertical_loss) / 2               

    # Track metrics
    train_depth_losses     .append(train_depth_loss)
    train_horizontal_losses.append(train_horizontal_loss)
    train_vertical_losses  .append(train_vertical_loss  )
    train_total_losses     .append(train_total_loss     )


    # * 5.2 VALIDATION PHASE ----------------------------------------------------------------------------------------------------------------------
    # Set model to evaluation mode
    model.eval()  
    # Initialize epoch metrics
    epoch_depth_loss       = 0.0
    epoch_horizontal_loss  = 0.0
    epoch_vertical_loss    = 0.0
    
    with torch.no_grad():
        # Iterate throught validation batches
        for images, depth_maps, horizontal_labels, vertical_labels in tqdm(val_loader, desc=f"Epoch [{epoch + 1:03d}/{config['epochs']}] Validation"):

            # Get data to cuda if possible
            images            = images           .to(device=device)
            if config['use_depth_head']:
                depth_maps    = depth_maps       .to(device=device)
            horizontal_labels = horizontal_labels.to(device=device)
            vertical_labels   = vertical_labels  .to(device=device)
            
            # vertical output
            horizontal_outputs, vertical_outputs, depth_outputs = model(images)

            # Calculate losses
            if config['use_depth_head']:
                depth_loss  = depth_criterion     (depth_outputs,      depth_maps)
            horizontal_loss = horizontal_criterion(horizontal_outputs, horizontal_labels.argmax(dim=1))
            vertical_loss   = vertical_criterion  (vertical_outputs,   vertical_labels  .argmax(dim=1))
            loss = horizontal_loss + vertical_loss
            if config['use_depth_head']:
                loss += depth_loss

            # Accumulate losses
            if config['use_depth_head']:
                epoch_depth_loss  += depth_loss     .item()
            epoch_horizontal_loss += horizontal_loss.item()
            epoch_vertical_loss   += vertical_loss  .item()

    # Calculate average loss and accuracy
    val_depth_loss      = epoch_depth_loss      / len(val_loader)
    val_horizontal_loss = epoch_horizontal_loss / len(val_loader)
    val_vertical_loss   = epoch_vertical_loss   / len(val_loader)
    val_total_loss      = (val_horizontal_loss + val_vertical_loss) / 2

    # Track metrics
    val_depth_losses     .append(val_depth_loss     )
    val_horizontal_losses.append(val_horizontal_loss)
    val_vertical_losses  .append(val_vertical_loss  )
    val_total_losses     .append(val_total_loss     )

    # Update the learning rate at the end of each epoch
    if scheduler is not None:
        scheduler.step(val_total_loss)
        learning_rate = optimizer.param_groups[-1]['lr']
    learning_rates.append(learning_rate)

    # Print epoch information
    print(f"Training   Loss: {train_total_loss:.4f}")
    print(f"Validation Loss: {val_total_loss  :.4f}")
    print(f"Learning Rate  : {learning_rate  :.8f} ")
    print()

    # Log training and validation metrics to W&B
    if config['use_depth_head']:
        wandb.log({'train depth loss'      : train_depth_loss,
                   'train horizontal loss' : train_horizontal_loss,
                   'train vertical loss'   : train_vertical_loss,
                   'train total loss'      : train_total_loss,
                   'val depth loss'        : val_depth_loss,
                   'val horizontal loss'   : val_horizontal_loss,
                   'val vertical loss'     : val_vertical_loss,
                   'val total loss'        : val_total_loss,
                   'learning rate'         : learning_rate,
                   })
    else:
        wandb.log({'train horizontal loss' : train_horizontal_loss,
                   'train vertical loss'   : train_vertical_loss,
                   'train total loss'      : train_total_loss,
                   'val horizontal loss'   : val_horizontal_loss,
                   'val vertical loss'     : val_vertical_loss,
                   'val total loss'        : val_total_loss,
                   'learning rate'         : learning_rate,
                   })


    # * 5.3 EARLY STOPPING ------------------------------------------------------------------------------------------------------------------------
    if config['use_early_stop']:

        # Check if the current validation loss is better than the previous best loss
        if val_total_loss < best_loss:
            best_loss         = val_total_loss
            epochs_since_best = 0
            # Save the best model weights
            save_model_and_parameters(model  = model,
                                      config = config,
                                      mode   = 'multi_task_learning_grid')

        # If the validation loss does not improve, increase the counter
        else:
            epochs_since_best += 1

        # Check if early stopping condition is met (e.g., no improvement for a certain number of epochs)
        if epochs_since_best >= config['early_stop_patience']:
            print(f"Early stopping triggered. "
                  f"No improvement in validation loss for {config['early_stop_patience']} epochs.")
            break


# * 6.0 SAVE MODEL AND PARAMETERS -----------------------------------------------------------------------------------------------------------------
if not config['use_early_stop']:
    save_model_and_parameters(model  = model,
                              config = config,
                              mode   = 'multi_task_learning_grid')

# Close the W&B run
wandb.finish()

# Play a sound file when the script completes.
# This code is employed to ensure that upon the completion of
# a training session, a pleasant notification message is played,
# thereby informing me of the successful completion.
sound_file = "assets/complete_sound.wav"
playsound(sound_file)
