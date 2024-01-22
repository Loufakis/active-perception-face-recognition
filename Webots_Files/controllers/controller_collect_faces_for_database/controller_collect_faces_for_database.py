"""
Description:
    This script is designed to automatically generate a face database by spawning all the different 
    human models in a simulation environment of Webots and capturing their faces from various camera
    angles. The script utilizes the Webots Supervisor controller to manipulate the simulation and
    extract face data.

Webot World       : Create_Faces_Database
Run Controller On : Robot
"""

# * 1.0 IMPORTS -----------------------------------------------------------------------------------------------------------------------------------
import sys             # Add parent directory to the system path
sys.path.append('..')  # add parent directory to the system path

# Basics
import os
import time
import cv2
import numpy as np

# Webots
from controller import Robot, Supervisor, Node

# OpenDR Toolkit
from opendr.engine.data import Image
from opendr.perception.object_detection_2d import RetinaFaceLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes

# Custom
from Webots_Files.controllers.spawn_utils import HUMAN_CATALOG


# * 2.0 INITIALIZATIONS ---------------------------------------------------------------------------------------------------------------------------
# Database Directory: Specify where to store the collected face images
database_dir = '/media/mloufakis/Data/Datasets/Thesis_Datasets/faces_database/'

# Create Supervisor instance for controlling the simulation
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# Initialize and enable all cameras in the simulation
cameras = {'camera_center': None,
           'camera_up'    : None,
           'camera_down'  : None,
           'camera_left'  : None,
           'camera_right' : None,
           }
for camera_name in cameras.keys():
    # Get the camera device
    cameras[camera_name] = supervisor.getDevice(camera_name)
    # Enable the camera
    cameras[camera_name].enable(timestep)

# OpenCV window setup for displaying camera views
window_name = "Center Camera View"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Set up the face detection model (RetinaFace) with a specific backbone and load pretrained weights
learner = RetinaFaceLearner(backbone="mnet", device="cuda")
learner.download(".", mode="pretrained")
learner.load("./retinaface_{}".format("mnet"))
boxes   = None

# List of human models (protos) available in the simulation
protos_list = ['human_01_standing', 'human_02_standing', 'human_03_standing', 'human_04_standing', 'human_05_standing',
               'human_06_standing', 'human_07_standing', 'human_08_standing', 'human_09_standing', 'human_10_standing',
               'human_11_standing', 'human_12_standing', 'human_13_standing', 'human_14_standing', 'human_15_standing',
               'human_16_standing', 'human_17_standing', 'human_18_standing', 'human_19_standing', 'human_20_standing',
               'human_21_standing', 'human_22_standing', 'human_23_standing', 'human_24_standing', 'human_25_standing',
               'human_26_standing', 'human_27_standing', 'human_28_standing', 'human_29_standing', 'human_30_standing',
               'human_31_standing', 'human_32_standing', 'human_33_standing', 'human_34_standing', 'human_35_standing',
               'human_36_standing',
               ]


# * 3.0 HELPERS -----------------------------------------------------------------------------------------------------------------------------------
def update_camera_view(camera):
    """
    Captures and processes the current frame from the specified camera.
    Performs face detection on the frame using a pre-trained model and draws bounding boxes around detected faces.

    Parameters:
        camera (Camera): The camera device to capture the image from.

    Returns:
        tuple: Returns a tuple containing the processed image and a list of detected bounding boxes.
               If no data is retrieved from the camera, returns None for both elements.
    """

    # Reed camera data
    cameraData = camera.getImage()

    # Check if data where retrieved
    if cameraData:
        # Get current frame
        frame = np.frombuffer(cameraData, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        frame = frame[:, :, :3]
        image = Image(frame)

        # Perform inference
        boxes = learner.infer(image)  # a list of boxes

        # Display image
        image = image.opencv()
        image_with_boxes = image.copy()
        if boxes:
            # Inplace the current image with an image that has drawn the bounding boxes
            draw_bounding_boxes(image_with_boxes, boxes, class_names=learner.classes, line_thickness=2)

        # Plot the image (with the bounding boxes)
        cv2.imshow(window_name, image_with_boxes)

        # Wait for a key press to refresh the window
        key = cv2.waitKey(1)

        return image, boxes


def get_center_box(boxes, camera):
    """
    Identifies the bounding box closest to the center of the camera's field of view.
    This is useful for focusing on the most centrally located face in the image.

    Parameters:
        boxes    (list): A list of bounding boxes of detected faces in the image.
        camera (Camera): The camera device used for capturing the image. 
                         Needed to determine the center of the field of view.

    Returns:
        object: The bounding box closest to the center of the camera's field of view.
    """

    # Find the center of each box in the image
    boxes_centers = [box.left + box.width / 2 for box in boxes]
    # Calculate their distance from the image center (only width-wise)
    to_center_dist = [abs(center - camera.getWidth() / 2) for center in boxes_centers]
    # Locate the bos closest to the center
    central_box_index = np.argmin(to_center_dist)
    cntr_box = boxes[central_box_index]

    return cntr_box


def crop_and_store(img, box, person_name, camera_position):
    """
    Crops the part of the image defined by the bounding box and saves it as a separate image file.
    This function is used to extract and store individual faces from the larger camera frame.

    Parameters:
        img           (Image): The image from which to crop the face.
        box          (object): The bounding box defining the area to crop.
        person_name     (str): Name of the person, used to define the folder and filename for saving.
        camera_position (str): Position of the camera, used in naming the saved image file.

    Returns:
        None: This function does not return anything. It saves the cropped image to the filesystem.
    """

    # Get BBox bounds
    start_row = int(box.top)
    end_row   = int(box.top + box.height)
    start_col = int(box.left)
    end_col   = int(box.left + box.width)

    # Corp BBox part
    cropped_image = img[start_row:end_row, start_col:end_col]

    # Create folder
    folder_dir = os.path.join(database_dir, f'{person_name}')
    # Check if the folder already exists
    if not os.path.exists(folder_dir):
        # Create the folder
        os.makedirs(folder_dir)
    else:
        pass

    # Save the cropped image
    cv2.imwrite(os.path.join(folder_dir, f'{person_name}_face_{camera_position}.jpg'),
                cropped_image)


# * 4.0 COLLECT DATABASE IMAGES -------------------------------------------------------------------------------------------------------------------
# Main loop variables initialization
step      = 1
proto_idx = 0
isSpawn   = True

# Main simulation loop
# Perform simulation steps until Webots will stop the controller
while supervisor.step(timestep) != -1:

    # Iterate through all human models (protos)
    if proto_idx < len(protos_list):
        if step % 5 == 0:

            # Handling spawn and capture logic
            if isSpawn:
                # Spawn human model and adjust camera position

                print(f'Proto: {proto_idx}')
                # Choose proto
                proto = protos_list[proto_idx]
                # Spawn proto
                children = supervisor.getRoot().getField('children')
                # Warning here is very important to also give a DEF value to the proto.
                # so that I can delete him afterwords
                proto_string = f'DEF {proto} {proto} {{ translation ' \
                               f'{HUMAN_CATALOG[proto]["translation"]["x"]} '\
                               f'{HUMAN_CATALOG[proto]["translation"]["y"]} '\
                               f'{HUMAN_CATALOG[proto]["translation"]["z"]} rotation '\
                               f'{HUMAN_CATALOG[proto]["rotation"]["x"]} '\
                               f'{HUMAN_CATALOG[proto]["rotation"]["y"]} '\
                               f'{HUMAN_CATALOG[proto]["rotation"]["z"]} '\
                               f'{HUMAN_CATALOG[proto]["rotation"]["angle"]} }}'
                children.importMFNodeFromString(-1, proto_string)

                cameras_node = supervisor.getFromDef("Cameras")
                position     = cameras_node.getField('translation')
                position.setSFVec3f(list(np.array([0, -HUMAN_CATALOG[proto]["face"]["z"], 0])))
                supervisor.step(timestep)

                isSpawn = False

            elif not isSpawn:
                # For each camera, capture the image, detect and crop the face, and store it
                for camera_name in cameras.keys():
                    # Get camera data
                    image, boxes = update_camera_view(cameras[camera_name])
                    time.sleep(0.1)
                    # Find center box
                    central_box = get_center_box(boxes, cameras[camera_name])
                    # Crop and store proto face
                    crop_and_store(image, central_box, proto, camera_name.split('_')[-1])
                    # Delete the proto
                    node = supervisor.getFromDef(f'{proto}')

                # Remove the spawned human model from the simulation
                if node:
                    node.remove()
                else:
                    print(f'{node} not found')

                # Update the index for the next human model
                proto_idx += 1
                isSpawn = True

    else:
        # Once all human models have been processed, exit the simulation
        print('Database Created!')
        print('Exit Simulation')
        print(f'{proto_idx} faces registered')

        break

    # Increment the step counter
    step += 1
