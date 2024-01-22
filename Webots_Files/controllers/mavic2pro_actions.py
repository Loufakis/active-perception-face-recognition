# Basics
import numpy as np
import cv2
import time
from transforms3d import quaternions
from opendr.engine.data import Image

# Face Detection
from opendr.perception.object_detection_2d import RetinaFaceLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes

# Face Recognition
from opendr.perception.face_recognition import FaceRecognitionLearner


# This dictionary is a mapping of string keys to integer values, where each key represents a
# specific action command and its corresponding value is an integer used to uniquely identify that action.
Actions = {'forwards'    : 1,
           'backwards'   : 2,
           'left'        : 3,
           'right'       : 4,
           'rotate_left' : 5,
           'rotate_right': 6
           }


class RobotAction:
    """
    A class to represent performe the robots actions, specifically tailored for a drone (e.g., DJI Mavic 2 Pro).
    It includes functionalities for basic movement control, and facial recognition capabilities.

    Attributes:
        robot           (object): An instance of the robot (drone) being controlled.
        default_fov      (float): Default field of view of the drone's camera.
        mavic2pro       (object): A reference to the specific drone model being controlled.
        rotation        (object): The rotation field of the drone.
        position        (object): The position field of the drone.
        timestep           (int): The timestep for the robot's actions.
        camera          (object): The camera device of the drone.
        camera_width       (int): The width of the camera's viewport.
        camera_height      (int): The height of the camera's viewport.
        fov              (float): The field of view of the camera.
        face_detector   (object): An instance of a face detection model.
        boxes             (list): A list to hold detected bounding boxes of faces.
        face_recognizer (object): An instance of a face recognition model.

    Methods:
        set_view(self)           : Sets up an OpenCV window for displaying the camera view.
        update_view(self)        : Updates the camera view with the latest frame and detected faces.
        _central_box(self)       : Finds the bounding box closest to the center of the view.
        _crop_box(self, img, box): Crops the image to the area defined by the given box.
        step(self, action)       : Performs an action based on the given command, controlling the
                                   drone's movement and camera.
    """

    def __init__(self, robot, default_fov=0.8):
        """
        Initializes the RobotAction class with the provided robot instance, and the default field of view.

        Parameters:
            robot                (object): An instance of the robot (drone) being controlled.
            default_fov (float, optional): Default field of view of the drone's camera. Defaults to 0.8.
        """

        # Parameters
        self.default_fov = default_fov

        # Set up robot attributes
        self.robot     = robot                                   # The robot instance
        self.mavic2pro = self.robot.getSelf()                    # Reference to the Mavic 2 Pro drone
        self.rotation  = self.mavic2pro.getField('rotation')     # Get robot's rotation field
        self.position  = self.mavic2pro.getField('translation')  # Get robot's position field
        self.timestep  = int(self.robot.getBasicTimeStep())      # Time step for robot's actions

        # Set up camera attributes
        self.camera        = self.robot.getDevice('camera')  # Get camera device
        self.camera.enable(self.timestep)                    # Enable the camera with the given timestep
        self.camera_width  = self.camera.getWidth()          # Camera width in pixels
        self.camera_height = self.camera.getHeight()         # Camera height in pixels
        self.fov           = self.camera.getFov()            # Field of view of the camera

        # Initialize drone view by creating an OpenCV window
        self.set_view()

        # Set up detection model
        self.face_detector = RetinaFaceLearner(backbone="mnet", device="cuda")
        self.face_detector.download(".", mode="pretrained")
        self.face_detector.load("./retinaface_{}".format("mnet"))
        self.boxes   = None  # Placeholder for detected boxes

        # Set up face recognition model
        self.face_recognizer = FaceRecognitionLearner(device='cuda', backbone='mobilefacenet', mode='backbone_only')
        self.face_recognizer.download(path=".")
        self.face_recognizer.load(".")
        # Create a database of faces for recognition
        self.face_recognizer.fit_reference(path='/media/mloufakis/Data/Datasets/Thesis_Datasets/faces_database/',
                                           save_path='.',
                                           create_new=True)


    def set_view(self):
        """
        Sets up an OpenCV window to display the camera view from the drone.
        The window is named 'Mavic View' and is set to a normal window type.
        """

        # Create an OpenCV window to display the drone's camera view
        self.window_name = "Mavic View"  # Window name
        # Create OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)


    def update_view(self):
        """
        Captures the latest camera frame from the drone, performs face detection,
        and updates the OpenCV window with the frame.

        Returns:
            numpy.ndarray: The latest image frame captured from the camera.
        """

        # Update the camera view with the latest frame and detected faces
        cameraData = self.camera.getImage()  # Get the latest camera image

        # Check if data where retrieved
        if cameraData:
            # Get current frame data
            frame = np.frombuffer(cameraData, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
            frame = frame[:, :, :3]  # Extract RGB channels
            image = Image(frame)     # Convert frame to an Image object

            # Perform face detection
            boxes = self.face_detector.infer(image)  # Detect faces and get bounding boxes

            # Display the image with bounding boxes
            image_with_boxes = image.opencv().copy()
            if boxes:
                # Inplace the current image with an image that has drawn the bounding boxes
                draw_bounding_boxes(image_with_boxes, boxes, class_names=self.face_detector.classes, line_thickness=4)

            # Show the image in the OpenCV window
            cv2.imshow(self.window_name, image_with_boxes)

            # Wait for a key press to refresh the window
            key = cv2.waitKey(1)
            
            # Update the boxes attribute with detected faces
            self.boxes = boxes 

        return image

    
    def _central_box(self):
        """
        Identifies the bounding box of a detected face that is closest to the center of the camera view.

        Returns:
            object: The bounding box closest to the center of the view, or None if no boxes are detected.
        """

        # Find the bounding box that is closest to the center of the view
        if self.boxes:
            # Calculate the center of the camera view
            camera_center = (self.camera_width / 2, self.camera_height / 2)

            # Calculate the distance between each box's center point and the image center point
            distances = []
            for box in self.boxes:
                box_center = (box.left + box.width / 2, box.top + box.height / 2)
                distance = ((camera_center[0] - box_center[0]) ** 2 + (camera_center[1] - box_center[1]) ** 2) ** 0.5
                distances.append(distance)

            cntr_box_idx = np.argmin(distances)  # Find the index of the closest box
            cntr_box = self.boxes[cntr_box_idx]  # Get the closest box

            return cntr_box

        else:
            return None


    def _crop_box(self, img, box):
        """
        Crops the provided image to the area defined by the given bounding box.

        Parameters:
            img (numpy.ndarray): The image to be cropped.
            box        (object): The bounding box defining the area to crop.

        Returns:
            numpy.ndarray: The cropped section of the image.
        """

        if box:
            # Calculate the bounds of the box
            start_row = int(box.top)
            end_row   = int(box.top + box.height)
            start_col = int(box.left)
            end_col   = int(box.left + box.width)

            # Crop the image to the box
            cropped_image = img[start_row:end_row, start_col:end_col]

            return cropped_image

        else:
            return None


    def step(self, action):
        """
        Performs a specific action based on the given command. This includes controlling the drone's movement
        (forward, backward, left, right, up, down, rotate left, rotate right) and performing face recognition.

        Parameters:
            action (int): An integer representing the specific action to be performed.

        Returns:
            tuple: A tuple containing the confidence and the cropped face image if action 9 (face verification) is performed.
            Otherwise, returns None.
        """
 
        image = self.update_view()  # Update the camera view

        # Define drone movements based on action code
        if   action == 0:
            # DO NOTHING
            pass

        elif action == 1:
            # MOVE FORWARDS
            ori   = np.array(self.mavic2pro.getOrientation()).reshape([3, 3])
            b     = np.array([0.01, 0, 0])
            new_p = ori.dot(b) + self.position.getSFVec3f()
            self.position.setSFVec3f(list(new_p))
            self.robot.step(self.timestep)

        elif action == 2:
            # MOVE BACKWARDS
            ori   = np.array(self.mavic2pro.getOrientation()).reshape([3, 3])
            b     = np.array([-0.01, 0, 0])
            new_p = ori.dot(b) + self.position.getSFVec3f()
            self.position.setSFVec3f(list(new_p))
            self.robot.step(self.timestep)

        elif action == 3:
            # MOVE TO THE LEFT
            ori   = np.array(self.mavic2pro.getOrientation()).reshape([3, 3])
            b     = np.array([0, 0.01, 0])
            new_p = ori.dot(b) + self.position.getSFVec3f()
            self.position.setSFVec3f(list(new_p))
            self.robot.step(self.timestep)

        elif action == 4:
            # MOVE TO THE RIGHT
            ori   = np.array(self.mavic2pro.getOrientation()).reshape([3, 3])
            b     = np.array([0, -0.01, 0])
            new_p = ori.dot(b) + self.position.getSFVec3f()
            self.position.setSFVec3f(list(new_p))
            self.robot.step(self.timestep)

        elif action == 5:
            # ROTATE TO THE LEFT
            rotation = self.rotation.getSFRotation()
            q1 = quaternions.axangle2quat(rotation[0:3], rotation[3])
            q2 = quaternions.axangle2quat([0, 0, 1], -1 * 3.14 / 180)  # rotate 1°
            q  = quaternions.qmult(q1, q2)
            vec, angle   = quaternions.quat2axangle(q)
            new_rotation = [vec[0], vec[1], vec[2], angle]
            self.rotation.setSFRotation(new_rotation)
            self.robot.step(self.timestep)

        elif action == 6:
            # ROTATE TO THE RIGHT
            rotation = self.rotation.getSFRotation()
            q1 = quaternions.axangle2quat(rotation[0:3], rotation[3])
            q2 = quaternions.axangle2quat([0, 0, 1], 1 * 3.14 / 180)  # rotate 1°
            q  = quaternions.qmult(q1, q2)
            vec, angle   = quaternions.quat2axangle(q)
            new_rotation = [vec[0], vec[1], vec[2], angle]
            self.rotation.setSFRotation(new_rotation)
            self.robot.step(self.timestep)

        elif action == 7:
            # MOVE UP
            ori   = np.array(self.mavic2pro.getOrientation()).reshape([3, 3])
            b     = np.array([0.0, 0, 0.01])
            new_p = ori.dot(b) + self.position.getSFVec3f()
            self.position.setSFVec3f(list(new_p))
            self.robot.step(self.timestep)

        elif action == 8:
            # MOVE DOWN
            ori   = np.array(self.mavic2pro.getOrientation()).reshape([3, 3])
            b     = np.array([0, 0, -0.01])
            new_p = ori.dot(b) + self.position.getSFVec3f()
            self.position.setSFVec3f(list(new_p))
            self.robot.step(self.timestep)

        elif action == 9:
            # FACE VERIFICATION
            central_box    = self._central_box()                 # Find the central box (face)
            face_to_verify = self._crop_box(image, central_box)  # Crop the image to the face

            # Get verification score
            if isinstance(face_to_verify, np.ndarray):
                results = self.face_recognizer.infer(face_to_verify)
                return results.confidence, face_to_verify
            else:
                # print('No face detected')
                return None, None



if __name__ == '__main__':
    pass
