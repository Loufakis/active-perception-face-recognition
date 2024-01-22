# * 1.0 IMPORTS ----------------------------------------------------------------------------------------------------------------------------------------------
import os
import json
import numpy  as np
import pandas as pd
from tqdm import tqdm
from PIL  import Image

# PyTorch
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Custom
from Code.datasets.dataset_utils import DerivativeSlope, IntegralRatio, SymmetricSampling, EstimationCorrection


# * 2.0 DEFINE DATASET CLASS ---------------------------------------------------------------------------------------------------------------------------------
class OcclusionRemovalDataset(Dataset):
    """
    A custom dataset class for handling occlusion removal data. It loads images
    and computes labels based on four labeling methods such as derivative slope,
    integral ratio, symmetric sampling, or estimation correction.

    Parameters:
        data_dir                      (str): Directory containing the image data.
        labeling_method               (str): Method used for labeling. Options are
                                             'derivative_slope', 'integral_ratio',
                                             'symmetric_sampling', 'estimation_correction'.
        accepted_verification_scores (list): Range of verification scores accepted for
                                             inclusion in the dataset.
        transform      (callable, optional): Transformations to be applied to the images.
        averaging_window              (int): Window size used for averaging the scores.
        border_width                  (int): Width of the border where no computation is done.
        derivative_threshold        (float): Threshold used for derivative slope method.
        integral_ratio_threshold    (float): Threshold used for integral ratio method.
        gaussian_decay_window         (int): Window size used for gaussian decay in symmetric
                                             sampling and estimation correction.
    """

    def __init__(self, data_dir, labeling_method, # ['derivative_slope', 'integral_ratio', 'symmetric_sampling', 'estimation_correction']
                 accepted_verification_scores=[0.00,0.95],
                 transform=None, averaging_window=3,
                 border_width=3,
                 derivative_threshold=0.02,
                 integral_ratio_threshold=0.15,
                 gaussian_decay_window=5):
        """
        Initializes the dataset and load sample images and labels.
        Parameters are documented in the class docstring.
        """

        # * 2.1 SAVE LOCALY THE INPUT PARAMETERS AND INITIALIZE THE SAMPLES LISTS ----------------------------------------------------------------------------
        self.data_dir                 = data_dir
        self.labeling_method          = labeling_method
        self.valid_ver_scores         = accepted_verification_scores
        self.transform                = transform
        self.averaging_window         = averaging_window
        self.border_width             = border_width
        self.derivative_threshold     = derivative_threshold
        self.integral_ratio_threshold = integral_ratio_threshold
        self.gaussian_decay_window    = gaussian_decay_window
        self.images                   = []
        self.depth_maps               = []
        self.horizontal_labels        = []
        self.vertical_labels          = []
        self.names                    = []
        self.positions                = []

        # * 2.2 CREATE AND INDEX THE SAMPLES -----------------------------------------------------------------------------------------------------------------
        # Iterate throught folders (each simulation)
        for folder_name in tqdm(os.listdir(self.data_dir), total=len(os.listdir(self.data_dir)), desc='Loading data..'):
            folder_dir = os.path.join(self.data_dir, folder_name)

            # Read the corresponding verification file
            csv_path = os.path.join(folder_dir, f'{folder_name}_verification_scores.csv')
            scores = pd.read_csv(csv_path, index_col=0)

            # Compute vertical and horizontal samples
            imgs, depth, hor_lbls, ver_lbl, nms, pos = self._get_folder_samples(scores, folder_dir, folder_name)
            
            # Concut them to the dataset lists
            self.images           .extend(imgs    )
            self.depth_maps       .extend(depth   )
            self.horizontal_labels.extend(hor_lbls)
            self.vertical_labels  .extend(ver_lbl )
            self.names            .extend(nms     )
            self.positions        .extend(pos     )


    def __len__(self):
        """
        Overrides the default Dataset class method. Returns the total number
        of image pairs in the dataset.

        Returns:
        int: The length of the dataset.
        """

        # derivative_slope      : 216_128
        # integral_ratio        : 216_128
        # symmetric_sampling    : 216_128
        # estimation_correction : 216_128

        return len(self.images)  


    def __getitem__(self, index):
        """
        Overrides the default Dataset class method. Retrieves a specific sample
        (image pair and their labels) from the dataset based on the provided index.

        Parameters:
        index (int): The index of the desired sample in the dataset.

        Returns:
        tuple: A tuple containing the image, depth map, and their corresponding labels.
        """

        # Retrieve the stored data for the given index
        image_path       = self.images           [index]
        depth_map_path   = self.depth_maps       [index]
        horizontal_label = self.horizontal_labels[index]
        vertical_label   = self.vertical_labels  [index]

        # Load the images
        image     = Image.open(image_path)
        depth_map = Image.open(depth_map_path)

        # Check if transformations have been given
        if self.transform:
            transformed = self.transform({'image':image,
                                          'depth_map':depth_map}) # So that the same transformations are applyed to both images
            image       = transformed['image']
            depth_map   = transformed['depth_map']

        # Convert labels to tensors
        vertical_label   = torch.tensor(vertical_label  )
        horizontal_label = torch.tensor(horizontal_label)

        return image, depth_map, horizontal_label, vertical_label
    

    def _get_folder_samples(self, scores, folder_dir, folder_name):
        """
        Processes a single folder within the dataset to extract image paths and labels.

        This method prepares horizontal and vertical score DataFrames by averaging, constructs
        paths for image and depth map files, and determines labels for each image based on the
        specified labeling method. It ensures that the dataset is balanced by filtering out images
        with excessively high verification scores.

        Parameters:
            scores (DataFrame): The DataFrame containing scores for image occlusion.
            folder_dir   (str): The directory path of the current folder being processed.
            folder_name  (str): The name of the current folder.

        Returns:
            tuple: Contains lists of image paths, depth map paths, horizontal labels, vertical labels,
                   sample names, and sample positions for all images in the folder.
        """

        # * 2.3 PREPARE FOR DATASET CREATION -----------------------------------------------------------------------------------------------------------------

        # SEPERATE HORIZONTAL AND VERTICAL DIRECTIONS
        # Beacuse the next step is to average the scores, a direction must be choosen. Therefore,
        # two copies of the original scores are taken, so that this two ptions will be both be resolved. 
        horizontal_scores = scores.copy()
        vertical_scores   = scores.copy()

        # AVERAGING
        # Here the mean is computed either row-wise or column-wise, according to the axis value. Also, the center 
        # parameter to True is necceasery so that each time the window is applied, the current cell value will be at the
        # center of the window, in the defult case the cell value goes to the end. Also the period=1 alows the mean to be
        # calculaed even if 1 value is avaliable, so that this operetion will not shrink the gird cuse of the inability
        # to compute 3-leng window opoeretions close to the grid border.
        # Apply row-wise averaging
        horizontal_scores_avg = horizontal_scores.rolling(window=self.averaging_window, center=True, axis=1, min_periods=1).mean()
        # Apply column-wise averaging
        vertical_scores_avg   = vertical_scores  .rolling(window=self.averaging_window, center=True, axis=0, min_periods=1).mean()

        # * 2.4 CONSTUCT THE IMAGES PATHS DATAFRAME ----------------------------------------------------------------------------------------------------------
        # Initialize images_paths and depth_images_paths Dataframes that contains only None values and has the same row
        # and column names as the original sores Dataframe.
        images_paths     = pd.DataFrame(index=scores.index, columns=scores.columns)
        depth_maps_paths = pd.DataFrame(index=scores.index, columns=scores.columns)

        # Iterate throught the empty dataframe cells
        for row in list(scores.index):
            for column in list(scores.columns):
                # Determine the image path
                img_path = os.path.join(folder_dir, f'{folder_name}_{row}_{column}.jpg')
                depth_map_path = os.path.join(folder_dir, f'{folder_name}_{row}_{column}.jpg')
                # Make sure that there exists an image for this path
                if os.path.exists(img_path) and os.path.exists(depth_map_path):
                    # Fill the dataframe with the existing image path
                    images_paths    .loc[row, column] = img_path
                    depth_maps_paths.loc[row, column] = depth_map_path
                # If the image path does not exist move to the next cell
                else:
                    pass

        # * 2.5 CONSTUCT THE SAMPLES -------------------------------------------------------------------------------------------------------------------------
        # Initialize the samples lists
        sample_images            = []
        sample_depth_maps        = []
        sample_horizontal_labels = []
        sample_vertical_labels   = []
        sample_names             = []
        sample_positions         = []

        # COMPUTE THE LABELS
        # Choose labels construction method
        if self.labeling_method == 'derivative_slope':
            # Initialize label creator
            labelCreator = DerivativeSlope(images               = images_paths,
                                           horizontal_scores    = horizontal_scores_avg,
                                           vertical_scores      = vertical_scores_avg,
                                           border_width         = self.border_width,
                                           derivative_threshold = self.derivative_threshold
                                           )

        elif self.labeling_method == 'integral_ratio':
            # Initialize label creator
            labelCreator = IntegralRatio(images                   = images_paths,
                                         horizontal_scores        = horizontal_scores_avg,
                                         vertical_scores          = vertical_scores_avg,
                                         border_width             = self.border_width,
                                         integral_ratio_threshold = self.integral_ratio_threshold
                                         )

        elif self.labeling_method == 'symmetric_sampling':
            # Initialize label creator
            labelCreator = SymmetricSampling(images                = images_paths,
                                             horizontal_scores     = horizontal_scores_avg,
                                             vertical_scores       = vertical_scores_avg,
                                             border_width          = self.border_width,
                                             gaussian_decay_window = self.gaussian_decay_window
                                             )

        elif self.labeling_method == 'estimation_correction':
            # Initialize label creator
            labelCreator = EstimationCorrection(images                = images_paths,
                                                horizontal_scores     = horizontal_scores_avg,
                                                vertical_scores       = vertical_scores_avg,
                                                border_width          = self.border_width,
                                                gaussian_decay_window = self.gaussian_decay_window
                                                )
        
        # Compute the label Dataframes. Each position in this DataFrames has a 1-1 relation with the original
        # scores DataFrame. Each cell contains a label that consists either of 3 values for the hard labels
        # or of 2 values for the soft labels.
        horizontal_labels, vertical_labels = labelCreator.compute_dataframes()
        
        # COMPUTE EACH SAMPLE (IMAGE, HORIZONTAL LABEL AND VERTICAL LABEL)
        # Iterate throught dataframe cells
        for row in list(images_paths.index):
            for column in list(images_paths.columns):
                # Retrive the image path and depth image path
                image_path     = images_paths    .loc[row, column]
                depth_map_path = depth_maps_paths.loc[row, column]
                # Check if that image path is a string (it exists)
                if isinstance(image_path, str):

                    # To decrese the unbalance of label values. Meaning that in most cases, when the face is sliglitly, or not
                    # occluded the label is refrencing to the "stay center" behavior. Therefore, if the verification score is to 
                    # hight we exlude this sample in order to decrese the extrimly large amount of this knd of labels and therfore,
                    # end up with a more balanced dataset

                    # Cheking if the particular label is in the accepted verification range
                    if (scores.loc[row, column] >= self.valid_ver_scores[0]) and \
                       (scores.loc[row, column] <= self.valid_ver_scores[1]):

                        # Select each datapoint data
                        horizontal_label = horizontal_labels.loc[row, column]
                        vertical_label   = vertical_labels  .loc[row, column]
                        sample_name      = folder_name
                        sample_position  = f'{row}_{column}'

                        # Also make sure that the smples do not contain any nan values for the labels
                        # IF NEEDED, CHECK TRAIN TO BE SURE
                        if (self._isValidLabel(horizontal_label)) and (self._isValidLabel(vertical_label)):

                            # Append image path and the corresponding label
                            sample_images           .append(image_path      )
                            sample_depth_maps       .append(depth_map_path  )
                            sample_horizontal_labels.append(horizontal_label)
                            sample_vertical_labels  .append(vertical_label  )
                            sample_names            .append(sample_name     )
                            sample_positions        .append(sample_position )

        # Return the samples constructed for this folder        
        return sample_images, sample_depth_maps, sample_horizontal_labels, sample_vertical_labels, sample_names, sample_positions


    def _isValidLabel(self, label):
        """
        Checks if a given label is valid.

        This method validates a label by checking if it is not None and not NaN (Not a Number).
        It works for both single value labels and list-based labels. If the label is a list,
        it is considered valid. For non-list labels, it checks if the label is NaN.

        Parameters:
            label: The label to be validated. Can be a single value or a list.

        Returns:
            bool: True if the label is valid, False otherwise.
        """

        # Initialize the result as True, assuming the label is valid
        result = True

        # Check if the label is None, and set result to False if it is
        if label is None:
            result = False
        
        # Check if the label is not a list and is a NaN value,
        # set result to False if it is
        elif not isinstance(label, list):
            if np.isnan(label):
                result = False

        return result



if __name__ == '__main__':
    pass
