import sys
sys.path.append('..')  # add parent directory to the system path

import random
from Webots_Files.controllers.spawn_utils import *  # Import all contents from the Catalogs module



class SpawnController:
    """
    A controller for spawning and managing human and object entities in a robotic simulation environment.

    The class allows for the dynamic placement of humans and objects within the simulation, 
    choosing from predefined sets based on the specified mode (training, testing, etc.).

    Attributes:
        robot            (Robot): An instance of the robot class, used to interact with the simulation environment.
        mode               (str): The mode of operation ('train', 'test', 'all', 'real', 'over_the_shoulder').
        seed_value         (int): Seed value for random number generation to ensure reproducibility.
        humans_to_choose  (list): A list of human entities available for spawning.
        objects_to_choose (list): A list of object entities available for spawning.
        human_catalog     (dict): A dictionary containing specifications for different human entities.
        object_catalog    (dict): A dictionary containing specifications for different object entities.

    Methods:
        spawn(): Spawns a human and an object in the simulation with randomly chosen positions and orientations.
        clear(): Removes the currently spawned human and object from the simulation.
    """

    def __init__(self, robot, mode='train', seed_value=2):
        """
        Initializes the SpawnController with a robot instance, operation mode, and seed value.
        
        Parameters:
            robot    (Robot): The robot instance to interact with the simulation environment.
            mode       (str): Mode of operation ('train', 'test', 'all', 'real', 'over_the_shoulder'). Default is 'train'.
            seed_value (int): Seed value for random number generation. Default is 2.
        """

        # Robot instance for interaction within the simulation.
        self.robot = robot

        ## Select appropriate catalogs based on the mode.
        if   mode == 'train':  # train mode
            self.humans_to_choose  = TRAIN_HUMANS
            self.objects_to_choose = TRAIN_OBJECTS
        elif mode == 'test':  # test mode
            self.humans_to_choose  = TEST_HUMANS
            self.objects_to_choose = TEST_OBJECTS
        elif mode == 'all':  # all mode
            self.humans_to_choose  = ALL_HUMANS
            self.objects_to_choose = ALL_OBJECTS
        elif mode == 'real': # real case
            self.humans_to_choose  = REAL_HUMANS
            self.objects_to_choose = REAL_OBJECTS
        elif mode == 'over_the_soulder': # over the soulder case
            self.humans_to_choose  = OTS_HUMANS
            self.objects_to_choose = OTS_OBJECTS    

        self.human_catalog  = HUMAN_CATALOG   # Catalog containing human entity specifications.
        self.object_catalog = OBJECT_CATALOG  # Catalog containing object entity specifications.

        # Set the random seed for reproducibility.
        self.seed_value = seed_value
        random.seed(self.seed_value)  

        # Initialize attributes to store the properties of the current human and object entities.
        # These will be set when the spawn method is called.
        self.human_name       = None
        self.human_face_x     = None
        self.human_face_y     = None
        self.human_face_z     = None
        self.human_trans_x    = None
        self.human_trans_y    = None
        self.human_trans_z    = None
        self.human_rot_x      = None
        self.human_rot_y      = None
        self.human_rot_z      = None
        self.human_rot_angle  = None

        self.object_name      = None
        self.object_trans_x   = None
        self.object_trans_y   = None
        self.object_trans_z   = None
        self.object_rot_x     = None
        self.object_rot_y     = None
        self.object_rot_z     = None
        self.object_rot_angle = None


    def __choose_valid_coordinate(self, interval, face_corr):
        """
        Chooses a valid coordinate for object placement relative to the human face.

        Parameters:
            interval (list or float): The interval or fixed value for choosing the coordinate.
            face_corr        (float): The coordinate of the human face to which the object's position is relative.

        Returns:
            float: The calculated coordinate for object placement.
        """

        # Determine whether the interval is a list or a single value.
        if isinstance(interval, list):
            # It is TWO Lists
            if isinstance(interval[0], list):
                # If interval is a list of lists, choose one list randomly.
                interval = random.choice(interval)
            # Choose a random number within the chosen interval.
            displacement = random.uniform(interval[0], interval[1])
        else:
            # If interval is a single value, use it directly.
            displacement = interval

        # Calculate the object's coordinate by adding the displacement to the face center.
        if displacement is None:
            object_corr = 0
        else:
            object_corr = face_corr + displacement

        return object_corr


    def spawn(self):
        """
        Spawns a human and an object in the simulation. 
        Randomly selects from predefined lists and calculates their positions and orientations.
        """

        # Randomly choose a human and an object from the respective lists.
        self.human_name  = random.choice(list(self.humans_to_choose))
        self.object_name = random.choice(list(self.objects_to_choose))

        # Set the properties for the human entity.
        self.human_face_x    = self.human_catalog[self.human_name]['face']['x']
        self.human_face_y    = self.human_catalog[self.human_name]['face']['y']
        self.human_face_z    = self.human_catalog[self.human_name]['face']['z']
        self.human_trans_x   = self.human_catalog[self.human_name]['translation']['x']
        self.human_trans_y   = self.human_catalog[self.human_name]['translation']['y']
        self.human_trans_z   = self.human_catalog[self.human_name]['translation']['z']
        self.human_rot_x     = self.human_catalog[self.human_name]['rotation']['x']
        self.human_rot_y     = self.human_catalog[self.human_name]['rotation']['y']
        self.human_rot_z     = self.human_catalog[self.human_name]['rotation']['z']
        self.human_rot_angle = self.human_catalog[self.human_name]['rotation']['angle']
        
        # Set the properties for the object entity.
        self.object_trans_x   = self.object_catalog[self.object_name]['translation']['x']
        self.object_trans_y   = self.object_catalog[self.object_name]['translation']['y']
        self.object_trans_z   = self.object_catalog[self.object_name]['translation']['z']
        self.object_rot_x     = self.object_catalog[self.object_name]['rotation']['x']
        self.object_rot_y     = self.object_catalog[self.object_name]['rotation']['y']
        self.object_rot_z     = self.object_catalog[self.object_name]['rotation']['z']
        self.object_rot_angle = self.object_catalog[self.object_name]['rotation']['angle']

        # Construct the strings to define the spawn positions and rotations for the human and object.

        # define x
        self.object_trans_x = self.__choose_valid_coordinate(self.object_trans_x, self.human_face_x)
        # define y
        self.object_trans_y = self.__choose_valid_coordinate(self.object_trans_y, self.human_face_y)
        # define z
        self.object_trans_z = self.__choose_valid_coordinate(self.object_trans_z, self.human_face_z)

        # Warning: here it's very important to also give a DEF value to the protos, so that I can delete them afterwords
        # Define human spawn position
        human_string  = f'DEF {self.human_name} {self.human_name} ' \
                        f'{{ translation {self.human_trans_x} {self.human_trans_y} {self.human_trans_z} ' \
                        f'rotation {self.human_rot_x} {self.human_rot_y} {self.human_rot_z} {self.human_rot_angle} }}'
        # Define object spawn position
        object_string = f'DEF {self.object_name} {self.object_name} ' \
                        f'{{ translation {self.object_trans_x} {self.object_trans_y} {self.object_trans_z} ' \
                        f'rotation {self.object_rot_x} {self.object_rot_y} {self.object_rot_z} {self.object_rot_angle} }}'

        # Spawn!
        # Import the human and object nodes into the simulation environment.
        children = self.robot.getRoot().getField('children')  # Chose the tree's children Node
        children.importMFNodeFromString(-1, human_string )
        children.importMFNodeFromString(-1, object_string)


    def clear(self):
        """
        Removes the currently spawned human and object entities from the simulation.
        """

        # Remove the human entity if it exists.
        human = self.robot.getFromDef(f'{self.human_name}')
        if human:
            human.remove()
        else:
            print(f'{human} not found')

        # Remove the object entity if it exists.
        object = self.robot.getFromDef(f'{self.object_name}')
        if object:
            object.remove()
        else:
            print(f'{object} not found')



if __name__ == '__main__':
    pass
