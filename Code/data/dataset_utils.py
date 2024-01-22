# * 1.0 imports --------------------------------------------------------------------------------------------------------------
import numpy as np
import scipy.stats as stats
import pandas as pd


# * 2.0 COMMON HELPERS -------------------------------------------------------------------------------------------------------
def calculate_area(y_values):
    """
    Calculates the area under a curve defined by a series of y values using the trapezoidal rule.

    Parameters:
    y_values (list): List of y-coordinates of the points on the curve.

    Returns:
    float: The area under the curve.

    Notes:
    - By subtracting the minimum height, the function estimates
      the deviation of the area above the minimum value.
    """

    x_values = list(range(len(y_values)))  # Generate x-values starting from 0
    min_height = min(y_values)             # Find the minimum height of the curve
    area = 0.0

    # Because for each iteration i compute the trapezoid area by using this point and the previus
    # I have to start from the second point, so that the previus will be the starting one
    for i in range(1, len(x_values)):
        x_0 = x_values[i-1]
        x_1 = x_values[ i ]

        # Avoid using None values
        if (y_values[i-1] is None) or (np.isnan(y_values[i-1])):
            y_0 = 0
        else:
            y_0 = y_values[i-1]

        if( y_values[i] is None) or (np.isnan(y_values[i])):
            y_1 = 0
        else:
            y_1 = y_values[ i ]

        dx             = x_1 - x_0   # Calculate the width of the current trapezoid
        height1        = y_0 #- min_height      # Calculate the height of the left side of the trapezoid
        height2        = y_1 #- min_height      # Calculate the height of the right side of the trapezoid
        trapezoid_area = (height1 + height2) * dx / 2.0  # Calculate the area of the trapezoid
        area += trapezoid_area
    
    # Return the total area
    return area


def apply_gaussian(lst, std, mean):
    """
    Applies a Gaussian decay to the elements of the input list based on the given standard deviation and mean.

    Parameters:
        lst   (list): The input list containing numerical elements.
        std  (float): The standard deviation of the Gaussian decay.
        mean (float): The mean position for the Gaussian decay.

    Returns:
        list: A new list with Gaussian decay applied to each element based on the specified parameters.
    """

    # Initialize a list to store the decayed values
    gaussian_decay_list = []

    # Iterate through each element in the input list
    for idx, element in enumerate(lst):
        if element is not None:
            # Compute the Gaussian decay factor for the current element
            gaussian_factor = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((idx - mean) ** 2) / (2 * std ** 2))
            
            # Apply the Gaussian decay factor to the element
            gaussian_decay_list.append(element * gaussian_factor)
        
        else:
            gaussian_decay_list.append(None)

    # Return the list of gaussian decayed values
    return gaussian_decay_list


def remaining_energy_gaussian(point, mean, std_deviation, direction="left"):
    """
    Calculates the remaining energy (CDF) of a Gaussian distribution up to the given value.
    
    Parameters:
        point         (float) : Value up to which the CDF is calculated.
        mean          (float) : Mean of the Gaussian distribution.
        std_deviation (float) : Standard deviation of the Gaussian distribution.
        direction       (str) : Direction to calculate the CDF ("left" or "right"). Default is "left".
    
    Returns:
        float: The remaining energy (CDF) up to the given value.
    """
    
    # Calculate the standardized value
    z = (point - mean) / std_deviation 
    
    # Calculate the cumulative distribution function (CDF) to the left
    if direction == "left":
        energy = stats.norm.cdf(z)
    # Calculate the CDF to the right (1 - CDF)
    elif direction == "right":
        energy = 1 - stats.norm.cdf(z)
    else:
        raise ValueError("Direction must be either 'left' or 'right'")
    
    # Return the CDF value
    return energy


# * 3.0 GRID HELPERS ---------------------------------------------------------------------------------------------------------
# * 3.1 DERIVATIVE SLOPE -----------------------------------------------------------------------------------------------------
class DerivativeSlope():
    def __init__(self,
                 images,
                 horizontal_scores, vertical_scores,
                 border_width,
                 derivative_threshold):
        """
        Initialize the DerivativeSlope class.

        This class is designed to compute the derivatives of given scores and determine
        the best action based on these derivatives. It is primarily used for processing
        image data and associated scores to determine directional movements.

        Parameters:
            images            (DataFrame): A pandas DataFrame containing image data.
            horizontal_scores (DataFrame): A DataFrame containing horizontal scores for each image.
            vertical_scores   (DataFrame): A DataFrame containing vertical scores for each image.
            border_width            (int): The width of the border where no computation is done.
            derivative_threshold  (float): The threshold for derivative values to determine the action.

        Attributes:
            horizontal_output (DataFrame): A DataFrame to store the computed horizontal output labels.
            vertical_output   (DataFrame): A DataFrame to store the computed vertical output labels.
        """

        # Basic input parameters
        self.images            = images
        self.horizontal_scores = horizontal_scores
        self.vertical_scores   = vertical_scores
        self.border_width      = border_width
        # Special parameters for this kind of dataset
        self.derivative_threshold = derivative_threshold

        # Initialize outputs with the same dimentions row names and column names
        # as the input scores (they contain only None values)
        self.horizontal_output = pd.DataFrame(index=horizontal_scores.index, columns=horizontal_scores.columns)
        self.vertical_output   = pd.DataFrame(index=vertical_scores  .index, columns=vertical_scores  .columns)


    def compute_dataframes(self):
        """
        Compute the derivative of scores and determine the best action based on the derivative values.

        Compute the first derivative for the given scores. Acording to the parameter axis, the derivatives
        are computed either row-wise or column-wise. Also it is importan to notice that the positive pepriod
        values start the calculations from the begining (idx=0) of the row/columns indexes and move twords the
        larger values, but the negative start from the larger values and end at the begining (idx=0).

        Returns:
            tuple: A tuple containing the horizontal and vertical output DataFrames.
        """
        
        # Compute the row-wise derivative:
        horizontal_scores_dev = self.horizontal_scores.diff(periods=1, axis=1)
        # Compute the column-wise derivative:
        vertical_scores_dev   = self.vertical_scores  .diff(periods=1, axis=0)

        # Set up the direction dictionary
        self.phases = {'horizontal':{'data'     : self.horizontal_scores,
                                     'dev_data' : horizontal_scores_dev ,
                                     'output'   : self.horizontal_output},
                       'vertical'  :{'data'     : self.horizontal_scores,
                                     'dev_data' : vertical_scores_dev   ,
                                     'output'   : self.vertical_output  }}

        for phase in ['horizontal', 'vertical']:
            # dataframe dimentions
            n_rows, n_columns = self.phases[phase]['data'].shape

            # Iterate throught dataframe cells
            for row in range(self.border_width, n_rows-self.border_width):
                for column in range(self.border_width, n_columns-self.border_width):
                    
                    # Check if there is an image found for this position
                    if isinstance(self.images.iloc[row, column], str):

                        dev_score = self.phases[phase]['dev_data'].iloc[row, column]

                        # Determine the sample label
                        # If the 1st derivative is close to zero, then the slope of the scores is also zero,
                        # and therfore the stay center is the best action
                        if (abs(dev_score) <= self.derivative_threshold) or (dev_score is None):
                            label = [0, 1, 0]  # [up, center, down] or [left, center, right]
                        
                        # If the 1st derivative is smaller than zero then the slope is descending, 
                        # and therfore the go left/up action is the best action
                        elif dev_score < -self.derivative_threshold:
                            label = [1, 0, 0]  # [up, center, down] or [left, center, right]

                        # If the 1st derivative is larger than zero then the slope is ascending, 
                        # and therfore the go right/down action is the best action
                        elif dev_score > self.derivative_threshold:
                            label = [0, 0, 1]  # [up, center, down] or [left, center, right]
                        
                        # This dev_score is propably None
                        else:
                            # cotinue to the next
                            label = [0, 1, 0]
                    
                        # store new value
                        self.phases[phase]['output'].iloc[row, column] = label

        return self.phases['horizontal']['output'], self.phases['vertical']['output']


# * 3.2 INTEGRAL RATIO -------------------------------------------------------------------------------------------------------
class IntegralRatio():
    def __init__(self,
                 images,
                 horizontal_scores, vertical_scores,
                 border_width,
                 integral_ratio_threshold):
        """
        Initialize the IntegralRatio class.

        This class computes the integral ratio of scores in horizontal and vertical directions 
        and determines the best action based on these ratios. It is used for processing image 
        data and associated scores to make directional decisions.

        Parameters:
            images               (DataFrame): A pandas DataFrame containing image data.
            horizontal_scores    (DataFrame): A DataFrame containing horizontal scores for each image.
            vertical_scores      (DataFrame): A DataFrame containing vertical scores for each image.
            border_width               (int): The width of the border where no computation is done.
            integral_ratio_threshold (float): The threshold for integral ratios to determine the action.

        Attributes:
            horizontal_output (DataFrame): A DataFrame to store computed horizontal output labels.
            vertical_output   (DataFrame): A DataFrame to store computed vertical output labels.
        """

        # Basic input parameters
        self.images            = images
        self.horizontal_scores = horizontal_scores
        self.vertical_scores   = vertical_scores
        self.border_width      = border_width
        # Special parameters for this kind of dataset
        self.integral_ratio_threshold = integral_ratio_threshold

        # Initialize outputs with the same dimentions row names and column names
        # as the input scores (they contain only None values)
        self.horizontal_output = pd.DataFrame(index=horizontal_scores.index, columns=horizontal_scores.columns)
        self.vertical_output   = pd.DataFrame(index=vertical_scores  .index, columns=vertical_scores  .columns)


    def compute_dataframes(self):
        """
        Compute the integral ratio of scores and determine the best action based on the ratio values.

        The method calculates the integral of the scores before and after a central point 
        in both horizontal and vertical directions. Based on the ratio of these integrals 
        and a set threshold, it determines the best action (label) for each score. 
        The actions are stored in the output DataFrames.

        Returns:
            tuple: A tuple containing the horizontal and vertical output DataFrames.
        """

        # Set up the direction dictionary
        self.phases = {'horizontal':{'data'     : self.horizontal_scores,
                                     'output'   : self.horizontal_output},
                       'vertical'  :{'data'     : self.horizontal_scores,
                                     'output'   : self.vertical_output  }}

        for phase in ['horizontal', 'vertical']:
            # dataframe dimentions
            n_rows, n_columns = self.phases[phase]['data'].shape

            # Iterate throught dataframe cells
            for row in range(self.border_width, n_rows-self.border_width):
                for column in range(self.border_width, n_columns-self.border_width):
                    
                    # Check if there is an image found for this position
                    if isinstance(self.images.iloc[row, column], str):

                        # Determine the sub-lists
                        if phase == 'horizontal':
                            prev_values = [self.phases[phase]['data'].iloc[row, column-j] for j in range(0, self.border_width + 1)][::-1]
                            next_values = [self.phases[phase]['data'].iloc[row, column+j] for j in range(0, self.border_width + 1)]
                        if phase == 'vertical':
                            prev_values = [self.phases[phase]['data'].iloc[row-j, column] for j in range(0, self.border_width + 1)][::-1]
                            next_values = [self.phases[phase]['data'].iloc[row+j, column] for j in range(0, self.border_width + 1)]

                        # Calculate integrals (areas) for the adjacent values
                        integral_before = calculate_area(prev_values)
                        integral_after  = calculate_area(next_values)

                        # Compute the ratio of integrals
                        integral_ratio = self._compute_ratio(integral_before = integral_before,
                                                            integral_after  = integral_after)
                        
                        # Determine the sample label
                        # If the ratio is close to 1 then the stay center is the best action,
                        # as the two integrals are almost the same
                        if abs(integral_ratio-1) <= self.integral_ratio_threshold:
                            label = [0, 1, 0]  # [up, center, down] or [left, center, right]

                        # If the ratio is smaller than one then the left/down integral is larger and therfore,
                        # the go left/down action is the best
                        elif integral_ratio      < 1-self.integral_ratio_threshold:
                            label = [1, 0, 0]  # [up, center, down] or [left, center, right]

                        # If the ratio is larger than one then the right/up integral is larger and therfore,
                        # the go right/up action is the best
                        elif integral_ratio      > 1+self.integral_ratio_threshold:
                            label = [0, 0, 1]  # [up, center, down] or [left, center, right]

                        # This integral_ratio is propably None
                        else:
                            # cotinue to the next
                            label = [0, 1, 0]

                        # store new value
                        self.phases[phase]['output'].iloc[row, column] = label

        return self.phases['horizontal']['output'], self.phases['vertical']['output']


    def _compute_ratio(self, integral_before, integral_after):
        """
        Compute the ratio of two integrals.

        Parameters:
            integral_before (float): Integral value computed for the preceding  elements.
            integral_after  (float): Integral value computed for the succeeding elements.

        Returns:
            float: Ratio of integral_after to integral_before. If either integral is None, returns None.
        """
        
        # Check if any of the two integrals is None
        if (integral_before is None) or (integral_after is None):
            return None
        # Check if both integrals are zero. In that case no direction gives the best answer
        elif (integral_before == 0) and (integral_after == 0):
            return 1.0
        # If the integral_before is zero then the best action is to go twords the after_integral
        elif integral_before == 0:
            return np.inf
        # For all the remaining (normal situations)
        else:
            denominator = integral_after + integral_before
            integral_after  = integral_after  / denominator
            integral_before = integral_before / denominator

            return integral_after / integral_before


# * 3.3 SYMETRIC SAMPLING ----------------------------------------------------------------------------------------------------
class SymmetricSampling():
    def __init__(self,
                 images,
                 horizontal_scores, vertical_scores,
                 border_width,
                 gaussian_decay_window):
        """
        Initialize the SymmetricSampling class.

        This class is designed to apply symmetric sampling using a Gaussian decay window to
        horizontal and vertical scores of images, to determine the best action based on these
        scores. It is primarily used for processing image data and associated scores to make
        directional decisions.

        Parameters:
            images            (DataFrame): A pandas DataFrame containing image data.
            horizontal_scores (DataFrame): A DataFrame containing horizontal scores for each image.
            vertical_scores   (DataFrame): A DataFrame containing vertical scores for each image.
            border_width            (int): The width of the border where no computation is done.
            gaussian_decay_window   (int): The size of the Gaussian decay window for sampling.

        Attributes:
            horizontal_output (DataFrame): A DataFrame to store computed horizontal output labels.
            vertical_output   (DataFrame): A DataFrame to store computed vertical output labels.
        """

        # Basic input parameters
        self.images            = images
        self.horizontal_scores = horizontal_scores
        self.vertical_scores   = vertical_scores
        self.border_width      = border_width
        # Special parameters for this kind of dataset
        self.gaussian_decay_window = gaussian_decay_window

        # Initialize outputs with the same dimentions row names and column names
        # as the input scores (they contain only None values)
        self.horizontal_output = pd.DataFrame(index=horizontal_scores.index, columns=horizontal_scores.columns)
        self.vertical_output   = pd.DataFrame(index=vertical_scores  .index, columns=vertical_scores  .columns)


    def compute_dataframes(self):
        """
        Compute symmetric sampling using a Gaussian decay window for the scores and determine the action.

        The method applies a Gaussian decay window to the scores in both horizontal and vertical
        directions, calculates areas under the curve on each side of the center, and determines
        the action based on these areas.

        Returns:
            tuple: A tuple containing the horizontal and vertical output DataFrames.
        """

        # Set up the direction dictionary 
        self.phases = {'horizontal':{'axis_step': None, # placeholder for row values
                                     'axis_max' : self.horizontal_scores.shape[1],
                                     'data'     : self.horizontal_scores,
                                     'output'   : self.horizontal_output},
                       'vertical'  :{'axis_step': None, # placeholder for column values
                                     'axis_max' : self.vertical_scores.shape[0],
                                     'data'     : self.horizontal_scores,
                                     'output'   : self.vertical_output  }}

        for phase in ['horizontal', 'vertical']:
            # dataframe dimentions
            n_rows, n_columns = self.phases[phase]['data'].shape

            # Iterate throught dataframe cells
            for row in range(self.border_width, n_rows-self.border_width):
                for column in range(self.border_width, n_columns-self.border_width):

                    # Check if there is an image found for this position
                    if isinstance(self.images.iloc[row, column], str):

                        # Assign the corect step values to generalaize the process for each direction
                        self.phases['horizontal']['axis_step'] = column
                        self.phases['vertical'  ]['axis_step'] = row

                        # Check if in both directions the complete gaussian is available
                        if (self.phases[phase]['axis_step'] >= self.gaussian_decay_window) and \
                        (self.phases[phase]['axis_step'] < self.phases[phase]['axis_max']-self.gaussian_decay_window):
                            window_size = self.gaussian_decay_window
                        # If in a direction the gaussian center is cloose to the edge, define a smaler window size   
                        else:
                            window_size = min(self.phases[phase]['axis_step'], self.phases[phase]['axis_max']-self.phases[phase]['axis_step']-1)

                        # Determine the sub-list
                        if   phase == 'horizontal':
                            values = [self.phases[phase]['data'].iloc[row, c] for c in range(column-window_size,
                                                                                            column+window_size+1)]
                        elif phase == 'vertical':
                            values = [self.phases[phase]['data'].iloc[r, column] for r in range(row-window_size,
                                                                                                row+window_size+1)]

                        gaussian_mean   = window_size # To be set at the midle of the sublist
                        gaussian_std    = self.gaussian_decay_window/3
                        gaussian_scores = apply_gaussian(lst=values, std=gaussian_std, mean=gaussian_mean)
                        
                        # sample the windows of interest
                        left_serie  = gaussian_scores[0:window_size+1]
                        right_serie = gaussian_scores[window_size:2*window_size + 1]

                        # Compute the corresponding integrals
                        left_area  = calculate_area(left_serie )
                        right_area = calculate_area(right_serie)

                        # Calculate denominator
                        denominator = left_area + right_area

                        # check if the denominater is not zero
                        if denominator != 0:
                            # Normilize the labels
                            left_area  = left_area /denominator  # Left  / Up
                            right_area = right_area/denominator  # Right / Down

                        label = [left_area, right_area] # [Left/Up, Right/Down]
                        
                        # store new value
                        self.phases[phase]['output'].iloc[row, column] = label

        return self.phases['horizontal']['output'], self.phases['vertical']['output']


# * 3.3 ESTIMATION CORRECTION ------------------------------------------------------------------------------------------------
class EstimationCorrection():
    def __init__(self,
                 images,
                 horizontal_scores, vertical_scores,
                 border_width, gaussian_decay_window):
        """
        Initialize the EstimationCorrection class.

        This class applies estimation correction using a Gaussian decay window to
        horizontal and vertical scores of images. It is designed to refine the estimation
        process by correcting the areas under the Gaussian curve based on the proximity
        to the edges of the data.

        Parameters:
            images            (DataFrame): A pandas DataFrame containing image data.
            horizontal_scores (DataFrame): A DataFrame containing horizontal scores for each image.
            vertical_scores   (DataFrame): A DataFrame containing vertical scores for each image.
            border_width            (int): The width of the border where no computation is done.
            gaussian_decay_window   (int): The size of the Gaussian decay window for sampling.

        Attributes:
            horizontal_output (DataFrame): A DataFrame to store computed horizontal output labels.
            vertical_output   (DataFrame): A DataFrame to store computed vertical output labels.
        """

        # Basic input parameters
        self.images            = images
        self.horizontal_scores = horizontal_scores
        self.vertical_scores   = vertical_scores
        self.border_width      = border_width
        # Special parameters for this kind of dataset
        self.gaussian_decay_window = gaussian_decay_window

        # Initialize outputs with the same dimentions row names and column names
        # as the input scores (they contain only None values)
        self.horizontal_output = pd.DataFrame(index=horizontal_scores.index, columns=horizontal_scores.columns)
        self.vertical_output   = pd.DataFrame(index=vertical_scores  .index, columns=vertical_scores  .columns)
        self.row    = None
        self.column = None

        self.phases = {'horizontal':{'axis_step': None, # placeholder for row values
                                     'axis_max' : horizontal_scores.shape[1],
                                     'data'     : horizontal_scores,
                                     'output'   : self.horizontal_output},
                       'vertical'  :{'axis_step': None, # placeholder for column values
                                     'axis_max' : vertical_scores.shape[0],
                                     'data'     : horizontal_scores,
                                     'output'   : self.vertical_output }}
    

    def compute_dataframes(self):
        """
        Compute and correct the estimation of areas using a Gaussian decay window for the scores.

        The method applies a Gaussian decay window to the scores in both horizontal and vertical
        directions, calculates areas under the curve on each side of the center, and corrects
        these areas based on their distance to the edges of the data.

        Returns:
            tuple: A tuple containing the horizontal and vertical output DataFrames.
        """

        for phase in ['horizontal', 'vertical']:
            # dataframe dimentions
            n_rows, n_columns = self.phases[phase]['data'].shape

            # Iterate throught dataframe cells
            for row in range(self.border_width, n_rows-self.border_width):
                for column in range(self.border_width, n_columns-self.border_width):

                    # Check if there is an image found for this position
                    if isinstance(self.images.iloc[row, column], str):

                        # Assign the corect step values to generalaize the process for each direction
                        self.phases['horizontal']['axis_step'] = column
                        self.phases['vertical'  ]['axis_step'] = row

                        # Choose LEFT side window size
                        if self.phases[phase]['axis_step'] >= self.gaussian_decay_window:
                            left_window_size = self.gaussian_decay_window
                        else:
                            left_window_size = self.phases[phase]['axis_step']

                        # Choose RIGHT side window size
                        if self.phases[phase]['axis_step'] < self.phases[phase]['axis_max']-self.gaussian_decay_window:
                            right_window_size = self.gaussian_decay_window
                        else:
                            right_window_size = self.phases[phase]['axis_max'] - self.phases[phase]['axis_step']-1

                        # Determine gaussian scores
                        if   phase == 'horizontal':
                            values = [self.phases[phase]['data'].iloc[row, c] for c in range(column-left_window_size,
                                                                                            column+right_window_size+1)]
                        elif phase == 'vertical': 
                            values = [self.phases[phase]['data'].iloc[r, column] for r in range(row-left_window_size,
                                                                                                row+right_window_size+1)]
                        gaussian_mean   = left_window_size
                        gaussian_std    = self.gaussian_decay_window/3
                        gaussian_scores = apply_gaussian(lst=values, std=gaussian_std, mean=gaussian_mean)

                        # Deal with the LEFT side area
                        left_serie = gaussian_scores[0:left_window_size+1]
                        left_area  = calculate_area(left_serie)
                        if self.phases[phase]['axis_step'] < self.gaussian_decay_window:
                            remaining = remaining_energy_gaussian(point=0, mean=gaussian_mean,
                                                                std_deviation=gaussian_std, direction="left")
                            # Correct left area by deviding with the respected energy
                            left_area = (left_area * 0.5) / (0.5-remaining)
                        
                        # Deal with the RIGHT side area
                        right_serie = gaussian_scores[left_window_size:(left_window_size+right_window_size)+1]
                        right_area  = calculate_area(right_serie)
                        if self.phases[phase]['axis_step'] >= self.phases[phase]['axis_max']-self.gaussian_decay_window:
                            remaining = remaining_energy_gaussian(point=len(gaussian_scores)-1, mean=gaussian_mean,
                                                                std_deviation=gaussian_std, direction="right")
                            # Correct left area by deviding with the respected energy
                            right_area = (right_area * 0.5) / (0.5-remaining)

                        # Calculate denominator
                        denominator = left_area + right_area

                        # check if the denominater is not zero
                        if denominator != 0:
                            # Normilize the labels
                            left_area  = left_area /denominator  # Left  / Up
                            right_area = right_area/denominator  # Right / Down
                    
                        label = [left_area, right_area] # [Left/Up, Right/Down]
                        
                        self.phases[phase]['output'].iloc[row, column] = label

        return self.phases['horizontal']['output'], self.phases['vertical']['output']



if __name__ == '__main__':
    pass
