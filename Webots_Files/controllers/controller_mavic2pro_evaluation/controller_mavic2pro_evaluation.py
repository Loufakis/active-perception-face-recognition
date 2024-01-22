"""
Description:
    This script performs an evaluation of a deep learning model in a simulated environment of Webots using a
    Mavic 2 Pro drone. It involves spawning objects and humans in various configurations and using the model
    to decide the drone's movements based on face verification scores. The results of each iteration are
    stored and analyzed to evaluate the model's performance.

Webots World      : Occlusion_Removal
Run Controller On : Mavic 2 PRO
"""

# * 1.0 IMPORTS -----------------------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('..')                                      # Add parent directory to the system path
sys.path.append('/home/mloufakis/Desktop/MSc_Thesis/Code') # Here are my deep learning functions located

# Basics
import os
import yaml
import torch
import pandas  as pd
from PIL       import Image
from playsound import playsound

# Webots
from controller import Robot, Supervisor

# Custom
from Webots_Files.controllers.mavic2pro_actions   import *
from Webots_Files.controllers.spawn_controller     import *
from Webots_Files.controllers.mavic2pro_behaviors import *
from model       import HydraNet
from train_utils import choose_tain_val_transformations


# * 2.0 HYPERPARAMETERS ---------------------------------------------------------------------------------------------------------------------------
# Define essential parameters for the evaluation
seed_value       = 24    # Change every time (doesn't happen automatically)
maxEvalSteps     = 15
TotalItterations = 500
mode             = 'all' # train, test, all, real, over_the_soulder

# Choose a trained model to evaluate
run_name = 'rich-sound-19'

# E.g.
# Derivative Slope      : peachy-dew-33,   sandy-cosmos-42
# Integral Ratio        : lilac-glitter-5, stellar-violet-11 dry-disco-1
# Symmetric Sampling    : apricot-dawn-2,  pleasant-grass-6
# Estimation Correction : stoic-wood-2,    rich-sound-19


# * 3.0 HELPERS -----------------------------------------------------------------------------------------------------------------------------------
actions = {'nothing'  : 0,
           'go_left'  : 3,
           'go_right' : 4,
           'go_up'    : 7,
           'go_down'  : 8}


def choose_direction_to_move(horizontal, vertical, config):
    """
    Decides the drone's movement direction based on the model's output for horizontal and vertical predictions.

    The function takes into account the configuration setting ('soft' or 'hard') to interpret the model's output
    and determine the most suitable direction for the drone to move in both the horizontal (left/right) and 
    vertical (up/down) axes.

    Parameters:
        horizontal (list): A list containing model's predictions for horizontal movement.
        vertical   (list): A list containing model's predictions for vertical movement.
        config     (dict): A configuration dictionary that includes the 'label_type' key indicating 
                       the output interpretation method ('soft' or 'hard').

    Returns:
        tuple: A tuple containing two strings, the first indicating the horizontal move ('go_left', 'go_right', 
               or 'nothing') and the second indicating the vertical move ('go_up', 'go_down', or 'nothing').
    """

    if config['label_type'] == 'soft':
        # Horizontal movement
        if horizontal[0] > horizontal[1]:
            h_move = 'go_left'
        elif horizontal[0] < horizontal[1]:
            h_move = 'go_right'
        else:
            h_move = 'nothing'

        # Vertical movement
        if vertical[0] > vertical[1]:
            v_move = 'go_up'
        elif vertical[0] < vertical[1]:
            v_move = 'go_down'
        else:
            v_move = 'nothing'

    elif config['label_type'] == 'hard':
        # Horizontal movement
        # Find the maximum value in the list
        max_value = max(horizontal)
        # Find the index of the maximum value
        max_index = horizontal.index(max_value)
        if max_index == 0:
            h_move = 'go_left'
        elif max_index == 2:
            h_move = 'go_right'
        else:
            h_move = 'nothing'

        # Vertical movement
        # Find the maximum value in the list
        max_value = max(vertical)
        # Find the index of the maximum value
        max_index = vertical.index(max_value)
        if max_index == 0:
            v_move = 'go_up'
        elif max_index == 2:
            v_move = 'go_down'
        else:
            v_move = 'nothing'

    return h_move, v_move


def define_model(run_name):
    """
    Locates and returns the path to the model weights and configuration file based on a given run name.

    Iterates through directories in a specified parent directory to find a matching model run.
    Returns the paths to the model's saved weights and its configuration file if found.

    Parameters:
        run_name (str): The name of the model run to locate.

    Returns:
        tuple: A tuple containing three elements; the first is the path to the saved model weights (.pth file),
               the second is the path to the model's configuration file (.yaml), and the third is the name of 
               the folder (backbone name) where these files are found. If the model is not found, returns None 
               for all elements.
    """

    # Locate and return the model weights and config based on the run name
    parent_dir = '/home/mloufakis/Desktop/MSc_Thesis/Code/saved_models/'
    
    # Iterate through all folders in the parent directory
    for folder in os.listdir(parent_dir):
        # Construct the path to the current folder
        folder_path = os.path.join(parent_dir, folder)
        
        # Ensure the folder is indeed a directory (and not a file)
        if os.path.isdir(folder_path):
            # Iterate through all files in the current folder
            for file in os.listdir(folder_path):
                # Split the file name and extension
                file_name, file_extension = os.path.splitext(file)

                # Check if the file extension is .pth and the file name matches the run name
                if file_extension == '.pth' and file_name == run_name:
                    # Construct the path to the saved weights
                    weights_path = os.path.join(folder_path, file)

                    # Construct the path to the yaml file
                    yaml_path   = os.path.join(folder_path, f'{file_name}.yaml')
                    # Return the path and the backbone name (which is the folder name)
                    return weights_path, yaml_path, folder
    
    # Return None if the run name was not found in any folder
    print("Model couldn't be found")
    return None, None, None


# * 4.0 INITIALIZATIONS ---------------------------------------------------------------------------------------------------------------------------
# Create Robot instance and configure the model for evaluation
robot    = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Build the paths to the saved model and config
model_path, config_path, backbone_name = define_model(run_name)

# Set up the model to evaluate
# Load the config
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Instantiate the model
model = HydraNet(labels_type   = config['label_type'        ],
                 backbone_name = config['backbone_name'     ],
                 use_depth     = config['use_depth_head'    ],
                 pretrained    = config['pretrained_weights'])
# Load the model parameters
model.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
model.eval()  # Set the model to evaluation mode

# Get correct transformations
_, transform = choose_tain_val_transformations(config)

# Initialize various controllers for robot actions and spawning
action_applier = RobotAction(robot)
mavic_behavior = GridActivePerception(robot)
spawner        = SpawnController(robot, mode=mode, seed_value=seed_value)

# Flags and counters for controlling the flow of the evaluation
spawnPhase      = True
gridSapmlePhase = False
itteration      = 0

# DataFrame to store verification scores
columns_names = ['Human', 'Object', 'Initial']
for step in range(1, maxEvalSteps+1):
   columns_names.append(f'Step{step}') 
verification_scores = pd.DataFrame(columns=columns_names)


# * 5.0 SAMPLE GENERATION -------------------------------------------------------------------------------------------------------------------------
print(' ')
print('Start!')
print(' ')

# Main loop for running the evaluation
# Perform simulation steps until Webots will stop the controller
while robot.step(timestep) != -1:

    # Run the experiment for the given number of itterations
    if itteration < TotalItterations:

        # Handle the spawning phase
        if spawnPhase:
            # Spawn the next pair of Human-Object
            # Clear previous pair
            spawner.clear()

            # Spawn next pair
            spawner.spawn()

            # Initialize drone position (Set the drone position so
            # that the camera looks to the center of the human face)
            drone_init_x = -2
            drone_init_y = spawner.human_face_y
            drone_init_z = spawner.human_face_z

            # Apply the new position
            mavic_behavior.restart(drone_init_x=drone_init_x,
                                   drone_init_y=drone_init_y,
                                   drone_init_z=drone_init_z)
            
            # Determine witch human and object were selected
            human_name  = str(spawner.human_name )
            object_name = str(spawner.object_name)

            # Check if in this run the human is at least parialy visible (a face is detected)
            # Capture image
            confidence, image = action_applier.step(9) # Apply Capture and verification action
            if (confidence is None) and (image is None):
                # No face detected
                # Re-spawn
                pass
            else:
                # Face was detected, continue normaly
                print(f'Collecting sample {itteration}/{TotalItterations}...')
                print(f'Initial verification score {confidence}')
                # Change flag variables
                spawnPhase      = False
                gridSapmlePhase = True
                evalStep        = 1
                new_ver_scores  = {'Human'   : human_name ,
                                   'Object'  : object_name,
                                   'Initial' : confidence
                                   }

        # Handle the grid sample phase (model evaluation)
        elif gridSapmlePhase:

            # First make sure the image exists (else move to next Human-Object pair)
            if image is not None:
                # Transform correctly the image
                preprocessed_image = Image.fromarray((image * 255).astype('uint8'))  # Convert array to PIL Image
                preprocessed_image = transform(preprocessed_image).unsqueeze(0)
                preprocessed_image = preprocessed_image.to(device)

                # Apply model predictions to move the drone
                with torch.no_grad():
                    horizontal_output, vertical_output, _ = model(preprocessed_image) # Here we dont need the depth estimation

                # According to the model predictions, choose the best direction to move in both axis (h:horizontal, v:vertical)
                h_move, v_move = choose_direction_to_move(horizontal_output.cpu().tolist()[0],
                                                          vertical_output  .cpu().tolist()[0],
                                                          config)
                
                # Apply horizontal movement
                action_applier.step(actions[h_move])
                # Apply vertical movement
                action_applier.step(actions[v_move])

                ## Capture image and store new verification score (After the movment)
                confidence, image = action_applier.step(9) # Apply Capture and verification action
                print(f'New verification score: {confidence}')

                # Update scores
                new_ver_scores[f'Step{evalStep}'] = confidence

                # Print some informations for the current step (Verification impruval)
                print(f'Evaluation step: {evalStep}')
                print(f'horizontal move: {horizontal_output.cpu().tolist()[0]}\t{h_move}')
                print(f'vertical   move: {vertical_output.cpu().tolist()[0]}\t{v_move}')

                if evalStep >= maxEvalSteps:
                    # Store iteration scores and update flags
                    verification_scores = verification_scores.append(new_ver_scores, ignore_index=True)
                    print(f'Current Metrics:')
                    print(verification_scores.mean())

                    # Change flag variables
                    spawnPhase       = True
                    gridSapmlePhase  = False
                    itteration       += 1

                else:
                    evalStep += 1 

            # If in any step the image got None (No face was detected). This step is excluded
            # Therfore in this case the iteration variable is not updated. Because we retake a pair
            # for this itteration
            else:
               # Change flag variables
                spawnPhase       = True
                gridSapmlePhase  = False
        print()

    else:
        # Once all iterations are complete, save the results and play a sound

        # Specify the file path
        folder_path = f'/home/mloufakis/Desktop/MSc_Thesis/Code/evaluation_results/{config["labeling_method"]}/'
        csv_path    = os.path.join(folder_path, f'{run_name}_{mode}.csv')

        # Check if the folder already exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f'Path {folder_path} created.')
        else:
            print(f'Path {folder_path} already exists.')

        # Save DataFrame to CSV
        verification_scores.to_csv(csv_path, index=False)  # Set index=False to not write row indices

        # The collection process is completed
        # Play a sound file when the script completes
        sound_file = "/home/mloufakis/Desktop/MSc_Thesis/Code/mission-completed.wav"
        playsound(sound_file)
        break

