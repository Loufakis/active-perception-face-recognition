# Basics
import os
import json
import time
import copy
import cv2
import pandas as pd

# Custom
from Webots_Files.controllers.mavic2pro_actions import *
from Webots_Files.controllers.grid_utils     import *


class ManualActivePerception:
    """
    A class for manual active perception using a robot, specifically designed for a Mavic 2 Pro drone.
    It is responsible for controlling the drone to perform face verification from different angles
    and distances by moving in various directions. This class is useful in scenarios where manual
    intervention is required for face verification tasks.

    Attributes:
        robot            (Robot): An instance of the robot class, used to interact with the simulation environment.
        mavic2pro        (Robot): A specific instance of Mavic 2 Pro drone obtained from the robot.
        rotation         (Field): The rotation field of the drone.
        position         (Field): The position field of the drone.
        timestep           (int): The basic time step of the robot's operation.
        done              (bool): A flag to indicate whether the perception task is completed.
        initial_drone_position (numpy.ndarray): Initial position of the drone.
        initial_tracker   (dict): A dictionary to track the initial state of the drone and its actions.
        tracker           (dict): A copy of the initial tracker to track the current state of the drone.
        actions           (dict): A dictionary mapping action names to their corresponding numeric codes.
        actions_names     (dict): A dictionary mapping numeric action codes to their corresponding names.
        next_action        (int): The next action to be taken by the drone.
        action_total_steps (int): The total number of steps to be taken for an action.
        last_verification  (str): The last verification action performed.

    Methods:
        restart(drone_init_x, drone_init_y, drone_init_z): Resets the drone to its initial position and state.
        store_confidence(confidence, image)              : Stores the confidence score and image from the verification.
        choose_action()                                  : Chooses the next action based on the current state
                                                           and verification results.
    """


    def __init__(self, robot):
        """
        Initializes the ManualActivePerception class with the provided robot instance.
        Sets up the drone's initial position, rotation, and other parameters for active perception.
        """

        # Robot parameters
        self.robot     = robot
        self.mavic2pro = self.robot.getSelf()
        self.rotation  = self.mavic2pro.getField('rotation')
        self.position  = self.mavic2pro.getField('translation')
        self.timestep  = int(self.robot.getBasicTimeStep())

        self.done = False

        # Initialization parameters
        self.initial_drone_position = np.array([-2, 0, 1.6])
        self.initial_tracker = {
            # Dictionary to track the drone's movement and verification steps
            'steps_done_towards_left'    : 0,
            'steps_done_towards_right'   : 0,
            'steps_done_towards_up'      : 0,
            'steps_done_towards_down'    : 0,

            'center_verification_done' : False,
            'left_verification_done'   : False,
            'right_verification_done'  : False,
            'up_verification_done'     : False,
            'down_verification_done'   : False,

            'center_verification_confidence': [],
            'left_verification_confidence'  : [],
            'right_verification_confidence' : [],
            'up_verification_confidence'    : [],
            'down_verification_confidence'  : [],

            'center_verification_images': [],
            'left_verification_images'  : [],
            'right_verification_images' : [],
            'up_verification_images'    : [],
            'down_verification_images'  : [],

            'steps_done_returning_from_left' : 0,
            'steps_done_returning_from_right': 0,
            'steps_done_returning_from_up'   : 0,
            'steps_done_returning_from_down' : 0,

            'returned_from_left' : False,
            'returned_from_right': False,
            'returned_from_up'   : False,
            'returned_from_down' : False,
            }

        self.tracker = copy.deepcopy(self.initial_tracker)
        
        # Action dictionary for communication with the drone control script
        self.actions            = {'nothing'          : 0,
                                   'go_left'          : 3,
                                   'go_right'         : 4,
                                   'go_up'            : 7,
                                   'go_down'          : 8,
                                   'face_verification': 9,
                                   }
        self.actions_names      = {0: 'Center',
                                   3: 'Left',
                                   4: 'Right',
                                   7: 'Up',
                                   8: 'Down',
                                   9: 'Face_Verification'
                                   }
        self.next_action        = None
        self.action_total_steps = 15
        self.last_verification  = None

    
    def restart(self, drone_init_x=-2, drone_init_y=0, drone_init_z=1.6):
        """
        Resets the drone to a specified initial position and reinitializes the tracker.

        Parameters:
            drone_init_x (float): Initial x-coordinate of the drone.
            drone_init_y (float): Initial y-coordinate of the drone.
            drone_init_z (float): Initial z-coordinate of the drone.
        """

        self.done = False

        # Initialize tracker
        self.tracker = copy.deepcopy(self.initial_tracker)

        # Initialize drone position
        self.initial_drone_position = np.array([drone_init_x, drone_init_y, drone_init_z])
        self.position.setSFVec3f(list(self.initial_drone_position))
        self.robot.step(self.timestep)

        # Initialize action parameters
        self.next_action       = None
        self.last_verification = None


    def store_confidence(self, confidence, image):
        """
        Stores the confidence score and corresponding image from a face verification attempt.

        Parameters:
            confidence (float): The confidence score from the face verification.
            image      (Image): The image used for the face verification.

        Returns:
            bool: Returns True if the process is to be terminated, False otherwise.
        """

        # if no confidence is detected
        if (confidence is None) and (self.last_verification == 'center'):
            # if no face is detected in the center verification, the process is interrupted
            # Because no comparison can be made to check if the score has improved
            self.done = True

        else:
            print(f'{self.last_verification}\tverification score: {confidence}')
            self.tracker[f'{self.last_verification}_verification_confidence'].append(confidence)
            self.tracker[f'{self.last_verification}_verification_images'    ].append(image     )

        return self.done


    def choose_action(self):
        """
        Determines and sets the next action for the drone based on the current state and verification results.
        This method follows a sequence of steps to verify from different angles and distances.

        Returns:
            tuple: A tuple containing the next action to be taken and a flag indicating if the process is done.
        """
        
        # 1. Perform Center Verification
        if not self.tracker['center_verification_done']:
            # Run the Center Verification
            self.next_action = self.actions['face_verification']
            # Track the verification
            self.tracker['center_verification_done'] = True
            self.last_verification = 'center'

        # 2. Perform Left Verifications
        elif (not self.tracker['left_verification_done']) or (not self.tracker['returned_from_left']):
            # Moving towards Left
            if (not self.tracker['left_verification_done']) and \
               (self.tracker['steps_done_towards_left'] < self.action_total_steps * 2) and \
               (self.tracker['steps_done_towards_left'] % 2 == 0):
                # Move Left
                self.next_action = self.actions['go_left']
                # Track this move
                self.tracker['steps_done_towards_left'] += 1

            # Run the Left Verifications
            elif not self.tracker['left_verification_done']:
                # Verify
                if len(self.tracker['left_verification_confidence']) < self.action_total_steps:
                    self.next_action = self.actions['face_verification']
                    self.last_verification = 'left'
                    # Track the verification
                    self.tracker['steps_done_towards_left'] += 1
                # The left verifications are completed
                else:
                    self.next_action = self.actions['nothing']
                    self.tracker['left_verification_done'] = True

            # Return to Center
            elif (self.tracker['left_verification_done']) and \
                 (self.tracker['steps_done_returning_from_left'] < self.action_total_steps):
                # Move Right
                self.next_action = self.actions['go_right']
                # Track this move
                self.tracker['steps_done_returning_from_left'] += 1

            # Next Direction
            else:
                # Don't move till the next direction checking starts
                self.next_action = self.actions['nothing']
                # Complete this check
                self.tracker['returned_from_left'] = True

        # 3. Perform Right Verifications
        elif (not self.tracker['right_verification_done']) or (not self.tracker['returned_from_right']):
            # Moving towards Right
            if (not self.tracker['right_verification_done']) and \
               (self.tracker['steps_done_towards_right'] < self.action_total_steps * 2) and \
               (self.tracker['steps_done_towards_right'] % 2 == 0):
                # Move Right
                self.next_action = self.actions['go_right']
                # Track this move
                self.tracker['steps_done_towards_right'] += 1

            # Run the Right Verifications
            elif not self.tracker['right_verification_done']:
                # Verify
                if len(self.tracker['right_verification_confidence']) < self.action_total_steps:
                    self.next_action = self.actions['face_verification']
                    self.last_verification = 'right'
                    # Track the verification
                    self.tracker['steps_done_towards_right'] += 1
                # The right verifications are completed
                else:
                    self.next_action = self.actions['nothing']
                    self.tracker['right_verification_done'] = True

            # Return to Center
            elif (self.tracker['right_verification_done']) and \
                 (self.tracker['steps_done_returning_from_right'] < self.action_total_steps):
                # Move Left
                self.next_action = self.actions['go_left']
                # Track this move
                self.tracker['steps_done_returning_from_right'] += 1

            # Next Direction
            else:
                # Don't move till the next direction checking starts
                self.next_action = self.actions['nothing']
                # Complete this check
                self.tracker['returned_from_right'] = True

        # 4. Perform Up Verifications
        elif (not self.tracker['up_verification_done']) or (not self.tracker['returned_from_up']):
            # Moving towards Up
            if (not self.tracker['up_verification_done']) and \
               (self.tracker['steps_done_towards_up'] < self.action_total_steps * 2) and \
               (self.tracker['steps_done_towards_up'] % 2 == 0):
                # Move Up
                self.next_action = self.actions['go_up']
                # Track this move
                self.tracker['steps_done_towards_up'] += 1

            # Run the Up Verifications
            elif not self.tracker['up_verification_done']:
                # Verify
                if len(self.tracker['up_verification_confidence']) < self.action_total_steps:
                    self.next_action = self.actions['face_verification']
                    self.last_verification = 'up'
                    # Track the verification
                    self.tracker['steps_done_towards_up'] += 1
                # The up verifications are completed
                else:
                    self.next_action = self.actions['nothing']
                    self.tracker['up_verification_done'] = True

            # Return to Center
            elif (self.tracker['up_verification_done']) and \
                 (self.tracker['steps_done_returning_from_up'] < self.action_total_steps):
                # Move Down
                self.next_action = self.actions['go_down']
                # Track this move
                self.tracker['steps_done_returning_from_up'] += 1

            # Next Direction
            else:
                # Don't move till the next direction checking starts
                self.next_action = self.actions['nothing']
                # Complete this check
                self.tracker['returned_from_up'] = True

        # 5. Perform Down Verifications
        elif (not self.tracker['down_verification_done']) or (not self.tracker['returned_from_down']):
            # Moving towards Down
            if (not self.tracker['down_verification_done']) and \
               (self.tracker['steps_done_towards_down'] < self.action_total_steps * 2) and \
               (self.tracker['steps_done_towards_down'] % 2 == 0):
                # Move Down
                self.next_action = self.actions['go_down']
                # Track this move
                self.tracker['steps_done_towards_down'] += 1

            # Run the Down Verifications
            elif not self.tracker['down_verification_done']:
                # Verify
                if len(self.tracker['down_verification_confidence']) < self.action_total_steps:
                    self.next_action = self.actions['face_verification']
                    self.last_verification = 'down'
                    # Track the verification
                    self.tracker['steps_done_towards_down'] += 1
                # The down verifications are completed
                else:
                    self.next_action = self.actions['nothing']
                    self.tracker['down_verification_done'] = True

            # Return to Center
            elif (self.tracker['down_verification_done']) and \
                 (self.tracker['steps_done_returning_from_down'] < self.action_total_steps):
                # Move Left
                self.next_action = self.actions['go_up']
                # Track this move
                self.tracker['steps_done_returning_from_down'] += 1

            # Next Direction
            else:
                # Don't move till the next direction checking starts
                self.next_action = self.actions['nothing']
                # Complete this check
                self.tracker['returned_from_down'] = True

        # 6. Check if the behavior is completed
        else:
            # print(f"Center: {self.tracker['center_verification_confidence']}")
            # print(f"Left: {self.tracker['left_verification_confidence']}")
            # print(f"Right: {self.tracker['right_verification_confidence']}")
            # print(f"Up: {self.tracker['up_verification_confidence']}")
            # print(f"Down: {self.tracker['down_verification_confidence']}")

            self.next_action = self.actions['nothing']
            self.done        = True

        return self.next_action, self.done



class GridActivePerception():
    """
    This class is designed to perform data collection for active perception on a grid pattern
    using a robot, specifically tailored for a Mavic 2 Pro drone. It navigates through a
    predefined grid, performing face verification at each grid point and storing the results.

    Attributes:
        robot               (Robot): An instance of the robot class, used to interact with
                                     the simulation environment.
        mavic2pro           (Robot): A specific instance of Mavic 2 Pro drone obtained
                                     from the robot.
        rotation            (Field): The rotation field of the drone.
        position            (Field): The position field of the drone.
        timestep              (int): The basic time step of the robot's operation.
        done                 (bool): A flag to indicate whether the perception task is completed.
        initial_drone_position (numpy.ndarray): Initial position of the drone.
        grid_positions       (dict): A dictionary of grid positions to navigate.
        grid_template        (dict): A template for storing grid sample data.
        grid_samples         (dict): A dictionary to store samples from the grid.
        grid_keys (           list): A list of keys for grid positions.
        grid_initial_position (int): Initial index for grid navigation.
        grid_current_position (int): Current index in grid navigation.
        grid_next_position    (int): Next index to move to in the grid.

    Methods:
        restart(drone_init_x, drone_init_y, drone_init_z): Resets the drone to its initial position and state.
        store_samples_temporary(confidence, image)       : Temporarily stores the confidence score and image
                                                           for the current grid position.
        move_to_next_grid_position()                     : Moves the drone to the next position in the grid.
        store_samples_locally(save_dir, spawner)         : Stores the collected samples locally in the
                                                           specified directory.
    """


    def __init__(self, robot):
        """
        Initializes the GridActivePerception class with the provided robot instance.
        Sets up the drone's initial position, rotation, and grid parameters for active perception.
        """

        # Initialize robot parameters
        self.robot     = robot
        self.mavic2pro = self.robot.getSelf()
        self.rotation  = self.mavic2pro.getField('rotation')
        self.position  = self.mavic2pro.getField('translation')
        self.timestep  = int(self.robot.getBasicTimeStep())

        self.done = False

        # Initialization parameters
        self.initial_drone_position = np.array([-2, 0, 1.6])

        # All Grid positions 21x21. Each position contains the following coordinates [x, y, z] and
        # they are changing by a step of 0.02 in directions y and z
        self.grid_positions = copy.deepcopy(GRID_POSITIONS)
        self.grid_template  = copy.deepcopy(GRID_TEMPLATE )

        self.grid_samples = None
        self.grid_keys    = list(self.grid_positions)

        self.grid_initial_position = -1  # the index value to start on the grid
        self.grid_current_position = copy.deepcopy(self.grid_initial_position)      # the index value
        self.grid_next_position    = copy.deepcopy(self.grid_initial_position) + 1  # the index value


    def restart(self, drone_init_x=-2, drone_init_y=0, drone_init_z=1.6):
        """
        Resets the drone to a specified initial position and reinitializes grid tracking.

        Parameters:
            drone_init_x (float): Initial x-coordinate of the drone.
            drone_init_y (float): Initial y-coordinate of the drone.
            drone_init_z (float): Initial z-coordinate of the drone.
        """

        self.done = False

        # Initialize trackers
        self.grid_samples          = copy.deepcopy(self.grid_template        )
        self.grid_current_position = copy.deepcopy(self.grid_initial_position)
        self.grid_next_position    = copy.deepcopy(self.grid_initial_position) + 1 

        # Initialize drone position
        self.initial_drone_position = np.array([drone_init_x, drone_init_y, drone_init_z])
        self.position.setSFVec3f(list(self.initial_drone_position))
        self.robot.step(self.timestep)


    def store_samples_temporary(self, confidence, image):
        """
        Temporarily stores the confidence score and corresponding image from the current grid position.

        Parameters:
            confidence (float): The confidence score from the face verification.
            image (Image): The image used for the face verification.

        Returns:
            bool: Returns True if the grid navigation is complete, False otherwise.
        """

        self.grid_samples[self.grid_keys[self.grid_current_position]]['Verification_Score'] = confidence
        self.grid_samples[self.grid_keys[self.grid_current_position]]['Image']              = image

        # Check if the grid has reached the final position
        if self.grid_current_position == len(self.grid_positions)-1:
            self.done = True

        return self.done


    def move_to_next_grid_position(self):
        """
        Moves the drone to the next position in the grid and updates the current position tracker.
        """

        # Find new position
        self.grid_next_position = self.grid_current_position + 1

        # Move the drone to the new grid position
        grid_next_position_coordinates = [x + y for x, y in zip(list(self.initial_drone_position), self.grid_positions[self.grid_keys[self.grid_next_position]])]
        self.position.setSFVec3f(grid_next_position_coordinates)
        self.robot.step(self.timestep)
        
        # Update tracker
        self.grid_current_position = self.grid_next_position


    def store_samples_locally(self, save_dir, spawner):
        """
        Stores the collected samples locally, including verification scores, images, and other
        relevant information.

        Parameters:
            save_dir            (str): Directory where the samples will be saved.
            spawner (SpawnController): An instance of the SpawnController class to access spawn details.
        """

        # General informations
        json_data = {
                    'human_name'            : str  (spawner.human_name      ),
                    'human_translation_x'   : float(spawner.human_trans_x   ),
                    'human_translation_y'   : float(spawner.human_trans_y   ),
                    'human_translation_z'   : float(spawner.human_trans_z   ),
                    'human_rotation_x'      : float(spawner.human_rot_x     ),
                    'human_rotation_y'      : float(spawner.human_rot_y     ),
                    'human_rotation_z'      : float(spawner.human_rot_z     ),
                    'human_rotation_angle'  : float(spawner.human_rot_angle ),
                    'object_name'           : str  (spawner.object_name     ),
                    'object_translation_x'  : float(spawner.object_trans_x  ),
                    'object_translation_y'  : float(spawner.object_trans_y  ),
                    'object_translation_z'  : float(spawner.object_trans_z  ),
                    'object_rotation_x'     : float(spawner.object_rot_x    ),
                    'object_rotation_y'     : float(spawner.object_rot_y    ),
                    'object_rotation_z'     : float(spawner.object_rot_z    ),
                    'object_rotation_angle' : float(spawner.object_rot_angle),
                    'drone_initial_position': self.initial_drone_position.tolist()
                    }
        
        # Verification DataFrame
        # Create a 21x21 DataFrame filled with None values
        data = [[None] * 21 for _ in range(21)]
        # Set row names
        row_names    = ['row:0.20' , 'row:0.18' , 'row:0.16' , 'row:0.14' , 'row:0.12' , 'row:0.10',  'row:0.08' , 'row:0.06' , 'row:0.04' ,
                        'row:0.02' , 'row:0.00' , 'row:-0.02', 'row:-0.04', 'row:-0.06', 'row:-0.08', 'row:-0.10', 'row:-0.12', 'row:-0.14',
                        'row:-0.16', 'row:-0.18', 'row:-0.20',]
        # Set column names
        column_names = ['col:0.20' , 'col:0.18' , 'col:0.16' , 'col:0.14' , 'col:0.12' , 'col:0.10',  'col:0.08' , 'col:0.06' , 'col:0.04' ,
                        'col:0.02' , 'col:0.00' , 'col:-0.02', 'col:-0.04', 'col:-0.06', 'col:-0.08', 'col:-0.10', 'col:-0.12', 'col:-0.14',
                        'col:-0.16', 'col:-0.18', 'col:-0.20',]
        df = pd.DataFrame(data, columns=column_names, index=row_names)
        for key in self.grid_keys:
            row, col = key.split('_')
            df[col][row] = self.grid_samples[key]['Verification_Score']

        # Store Data ----------------------------------------------------------------------------------------
        # Find current timestamp
        current_time = time.time()
        # Create new directory
        sample_dir = os.path.join(save_dir, f'{current_time}')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        # Store JSON
        # Create path
        json_path = os.path.join(sample_dir, f'{current_time}_general_infos.json')
        # Store
        with open(json_path, "w") as file:
            json.dump(json_data, file)

        # Store Verification Scores
        csv_path = os.path.join(sample_dir, f'{current_time}_verification_scores.csv')
        # Save the DataFrame to a CSV file
        df.to_csv(csv_path, index=True)  # Set index=False to exclude the index column

        # Store Images
        # Choose direction
        for key in self.grid_keys:
            ver_score = self.grid_samples[key]['Verification_Score']
            image     = self.grid_samples[key]['Image'             ]
            
            if image is not None:
                # Save json
                row, col = key.split('_')
                img_json_path = os.path.join(sample_dir, f'{current_time}_{key}.json')
                image_json = {'position':{'row'   :row,
                                          'column':col},
                              'verification_score':ver_score
                              }
                with open(img_json_path, "w") as file:
                    json.dump(image_json, file)

                # save image
                img_path = os.path.join(sample_dir, f'{current_time}_{key}.jpg')
                cv2.imwrite(img_path, image)



if __name__ == '__main__':
    pass       
