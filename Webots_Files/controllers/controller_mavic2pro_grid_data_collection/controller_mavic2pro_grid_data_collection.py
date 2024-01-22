"""
This script is designed to automate the collection of samples for a grid-based simulation
using a Mavic 2 Pro drone in the Webots simulation environment. It aims to generate a
dataset for training or testing purposes by capturing images from various grid positions.

Webots World      : Occlusion_Removal
Run Controller On : Mavic 2 PRO
"""


# * 1.0 IMPORTS -----------------------------------------------------------------------------------------------------------------------------------
import sys             # Add parent directory to the system path
sys.path.append('..')  # add parent directory to the system path

# Basics
import os
import time
import json

# Webots
from controller import Robot, Supervisor

# Custom
from Webots_Files.controllers.mavic2pro_actions   import *
from Webots_Files.controllers.spawn_controller     import *
from Webots_Files.controllers.mavic2pro_behaviors import *


# * 2.0 HYPERPARAMETERS ---------------------------------------------------------------------------------------------------------------------------
# Define hyperparameters for the sample collection
mode             = 'train' # Choose from 'train' / 'test'
SamplesToCollect = 1_500
seed_value       = 24      # Change every time (doesn't happen automatically)


# * 3.0 INITIALIZATIONS ---------------------------------------------------------------------------------------------------------------------------
robot    = Supervisor()  # Initialize the Webots Supervisor (robot controller)
timestep = int(robot.getBasicTimeStep())

# Initialize controllers for drone actions, grid behavior, and spawning entities
action_applier = RobotAction(robot)
mavic_behavior = GridActivePerception(robot)
spawner        = SpawnController(robot, mode=mode, seed_value=seed_value)

# Directory for saving collected samples
root_dir = f'/media/mloufakis/Data/Datasets/Thesis_Datasets/grid_simulation_{mode}_samples'

# Flags and counter for managing the sample collection process
spawnPhase      = True
gridSapmlePhase = False
sampleTracker   = 0


# * 4.0 SAMPLE GENERATION -------------------------------------------------------------------------------------------------------------------------
# Start the sample collection
print(' ')
print('Start!')
print(' ')

# Main simulation loop
# Perform simulation steps until Webots will stop the controller
while robot.step(timestep) != -1:

    if sampleTracker < SamplesToCollect:

        # Spawn a new human-object pair and initialize the drone's position
        if spawnPhase:

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
            
            # Check if the human is at least partially visible. If not, re-spawn
            # Capture Image
            confidence, image = action_applier.step(9) # Apply Capture and verification action
            if (confidence is None) and (image is None):
                # No face detected
                # Re-spawn
                pass
            else:
                # Face was detected, continue normaly
                print(f'Collecting sample {sampleTracker}/{SamplesToCollect}...')
                start = time.time()
                # Change flag variables
                spawnPhase      = False
                gridSapmlePhase = True
        
        # Move the drone to the next position in the grid and capture an image
        elif gridSapmlePhase:

            # Set drone next position
            mavic_behavior.move_to_next_grid_position()

            # Capture Image
            confidence, image = action_applier.step(9) # Apply Capture and verification action
            done = mavic_behavior.store_samples_temporary(confidence, image)

            # Check if all samples in the grid are collected. If so, store them locally
            if done:

                # Save smaples localy
                mavic_behavior.store_samples_locally(save_dir=root_dir, spawner=spawner)
                stop = time.time()
                print(f'Total elapse time: {(stop-start)/60} minutes')
                print()

                # Update the flags and tracker after storing the samples
                spawnPhase       = True
                gridSapmlePhase  = False
                sampleTracker   += 1

    else:
        # Once the desired number of samples is collected, exit the loop
        print('Sample collection completed.')
        break
    