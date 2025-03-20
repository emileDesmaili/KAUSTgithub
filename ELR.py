import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from sklearn.linear_model import LogisticRegression
from typing import Tuple

class ELR:
    """
    ELR class for Extended Logistic Regression (ELR) modeling.
    
    This class implements a rolling window approach to create quantile-based 
    labels for time series data, using logistic regression models fitted to each grid point 
    of a spatial-temporal dataset. It performs both the fitting of the model 
    using training data and predictions on new data.

    Attributes:
        window (int): The window size for rolling labeler.
        labeler (function): The labeler function used for quantile labeling.
        models (dict): A dictionary to store logistic regression models for each grid point.
    """
    
    def __init__(self, window=1):
        """
        Initialize the ELR class with a specified rolling window size.

        Args:
            window (int): The window size for quantile-based rolling labeler. Default is 1.
        """
        self.window = window
        self.labeler = None
        self.models = {}

    def fit(self, X: xr.DataArray, y: xr.DataArray) -> None:
        """
        Fit the ELR model using the provided training data.
        
        This method prepares the data by applying quantile-based labeling, then fits 
        an ELR model to each grid point in the spatial-temporal data.

        Args:
            X (xr.DataArray): The input data with shape (T, Y, X, M).
            y (xr.DataArray): The label data with shape (T, Y, X).
            
        Returns:
            None
        """
        X = X.transpose('T', 'Y', 'X', 'M')
        y = y.transpose('T', 'Y', 'X')

        # Initialize the labeler for creating quantile labels
        self.labeler = self.rolling_labeler_ELR(y, window=self.window)
        
        # Preprocess and label the training data
        edges, y_elr = self.labeler(y)

        # Prepare the ensemble of X data (average along the M dimension)
        X_ens = X.mean('M')
        X_ens = np.tile(X_ens, reps=(2, 1, 1))
        X_ens = xr.DataArray(X_ens, dims=('QT', 'Y', 'X'), coords={'Y': X.Y.values, 'X': X.X.values})

        y_elr = y_elr.stack(QT=('quantile', 'T')).transpose('QT', 'Y', 'X')
        edges_qt = edges.stack(QT=('quantile', 'T')).transpose('QT', 'Y', 'X')
        
        # Loop over each spatial grid point and fit logistic regression
        self.models = {}
        for i in range(len(X.Y)):
            for j in range(len(X.X)):
                # Extract data for the grid point
                y_grid_raw = y.isel(X=j, Y=i).values
                if np.isnan(y_grid_raw).any():
                    continue

                x_grid = X_ens.sel(X=X.X.values[j], Y=X.Y.values[i]).values
                edges_qt_grid = edges_qt.sel(X=X.X.values[j], Y=X.Y.values[i]).values

                # Label quantile edges
                total_elements = edges_qt_grid.shape[0]
                midpoint = total_elements // 2
                edges_qt_grid[:midpoint] = 33
                edges_qt_grid[midpoint:] = 66

                y_grid = y_elr.sel(X=X.X.values[j], Y=X.Y.values[i]).values
                
                if np.isnan(y_grid).all():
                    continue
                
                # Remove NaN entries and prepare data for logistic regression
                valid_indices = ~np.isnan(y_grid)
                X_qt_grid = np.stack([x_grid, edges_qt_grid], axis=1)
                y_grid = y_grid[valid_indices]
                X_qt_grid = X_qt_grid[valid_indices]

                # Fit the logistic regression model for this grid point
                model = sm.GLM(y_grid, sm.add_constant(X_qt_grid, has_constant='add'), family=sm.families.Binomial())
                self.models[(i, j)] = model.fit()

    def predict(self, X: xr.DataArray, y: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Make predictions using the trained ELR model.

        This method uses the trained extendded logistic regression models for each grid point 
        to predict the probabilities of each quantile category ('below', 'normal', 'above') 
        based on the test data.

        Args:
            X (xr.DataArray): The input test data with shape (T, Y, X, M).
            y (xr.DataArray): The test label data with shape (T, Y, X).
        
        Returns:
            Tuple[xr.DataArray, xr.DataArray]: 
                - Predicted probabilities for each quantile ('below', 'normal', 'above') as an xarray.
        """
        X = X.transpose('T', 'Y', 'X', 'M')
        y = y.transpose('T', 'Y', 'X')

        # Preprocess and label the testing data
        edges, y_elr = self.labeler(y)

        # Prepare the ensemble of X data (average along the M dimension)
        X_ens = X.mean('M')
        X_ens = np.tile(X_ens, reps=(2, 1, 1))
        X_ens = xr.DataArray(X_ens, dims=('QT', 'Y', 'X'), coords={'Y': X.Y.values, 'X': X.X.values})

        y_elr = y_elr.stack(QT=('quantile', 'T')).transpose('QT', 'Y', 'X')
        edges_qt = edges.stack(QT=('quantile', 'T')).transpose('QT', 'Y', 'X')
        
        # Initialize the storage for prediction results
        predictions_storage = np.full((len(X['T']), len(X.Y), len(X.X), 3), np.nan)
        
        # Loop over each grid point and make predictions
        for i in range(len(X.Y)):
            for j in range(len(X.X)):

                y_grid_raw = y.isel(X=j, Y=i).values
                if np.isnan(y_grid_raw).any():
                    continue

                x_grid = X_ens.sel(X=X.X.values[j], Y=X.Y.values[i]).values
                edges_qt_grid = edges_qt.sel(X=X.X.values[j], Y=X.Y.values[i]).values

                # Label quantile edges
                total_elements = edges_qt_grid.shape[0]
                midpoint = total_elements // 2
                edges_qt_grid[:midpoint] = 33
                edges_qt_grid[midpoint:] = 67

                y_grid = y_elr.sel(X=X.X.values[j], Y=X.Y.values[i]).values
                
                if np.isnan(y_grid).all():
                    continue
                
                # Remove NaN entries and prepare data for logistic regression prediction
                valid_indices = ~np.isnan(y_grid)
                valid_indices_half = valid_indices[:len(valid_indices)//2]
        
                X_qt_grid = np.stack([x_grid, edges_qt_grid], axis=1)
                y_grid = y_grid[valid_indices]
                X_qt_grid = X_qt_grid[valid_indices]

                model = self.models[(i, j)]
                predictions = model.predict(sm.add_constant(X_qt_grid, has_constant='add'))

                # Store predictions for each quantile
                p_below = predictions[:len(predictions)//2]
                p_above = predictions[len(predictions)//2:]

                predictions_storage[valid_indices_half, i, j, 0] = p_below
                predictions_storage[valid_indices_half, i, j, 1] = p_above - p_below
                predictions_storage[valid_indices_half, i, j, 2] = 1 - p_above

                # Fill the rest of indices with equal probabilities for each class
                predictions_storage[:, i, j, :] = np.nan_to_num(predictions_storage[:, i, j, :], nan=1/3)

        # Return predictions as xarray
        return xr.DataArray(predictions_storage, dims=('T', 'Y', 'X', 'category'), coords={'category': ['below', 'normal', 'above'],
                                                                                            'T': X['T'], 'Y': X['Y'], 'X': X['X']})

    def rolling_labeler_ELR(self, observations,window=1):

        '''
        Rolling labeler for ELR model
        Args:
            
            observations (xr.DataArray): The input data with shape (T, Y, X).
            window (int): The window size for rolling labeler. Default is 1.
                
        Returns:
            labeler (function): The labeler function used for quantile labeling.
                
                '''
        observations['T'] = pd.to_datetime(observations['T'].values)
        # Extract week numbers from 'T'
        week_values = observations['T'].dt.isocalendar().week.values

        # Add week as a coordinate variable
        observations = observations.assign_coords(week=('T', week_values))

        edges_list = []

        # Iterate over each unique week to compute quantiles
        for week in np.unique(week_values):
            # Select data for the specific week, 3 weeks before and 3 weeks after (6-week window)
            week_window = [(week + i) % 53 or 53 for i in range(-window, 1 + window)]
            observations_weekly = observations.sel(T=observations['week'].isin(week_window))

            # Compute quantiles for the selected week window
            edges = observations_weekly.quantile([1/3, 2/3], dim='T')
            
            # Store the quantiles with 'week' as a dimension
            edges = edges.assign_coords(week=week)
            edges_list.append(edges)

        # Concatenate edges across all weeks along the 'week' dimension
        edges = xr.concat(edges_list, dim='week')


        def labeler(y):
            # Ensure the 'T' coordinate is a datetime index
            y['T'] = pd.to_datetime(y['T'].values)
            
            # Extract week numbers from the 'T' dimension
            week_values = y['T'].dt.isocalendar().week.values
            y = y.assign_coords(week=('T', week_values))
            
            edges_list_t = []
            y_elr_list = []
            
            # Iterate over each unique week
            for week in np.unique(week_values):
                # Select the corresponding edges for the current week
                edges_weekly = edges.sel(week=week)
                
                # Select the observations for the current week
                y_weekly = y.sel(T=y['week'].isin(week))

                edges_weekly_t = edges_weekly.expand_dims(T=y_weekly['T'])
                edges_weekly_t = edges_weekly_t.assign_coords(T=y_weekly['T'])
                edges_list_t.append(edges_weekly_t)

                mask = (
                    edges_weekly.isnull().any('quantile') |
                    # Points where quantile 0 equals quantile 1 (degenerate case)
                    (edges_weekly.isel(quantile=0) == 0) | 
                    (edges_weekly.isel(quantile=0) == edges_weekly.isel(quantile=1))
                )

                y_below_33 = xr.where( (y_weekly <= edges_weekly.isel(quantile=0)), 1,0)
                y_below_66 = xr.where( (y_weekly <= edges_weekly.isel(quantile=1)), 1,0)
                #put on top of each other
                y_weekly_elr = xr.concat([y_below_33,y_below_66],dim='quantile').where(np.logical_not(mask))

                y_elr_list.append(y_weekly_elr)
                
            # Concatenate the labeled data across all weeks
            edges_t = xr.concat(edges_list_t, dim='T')
            y_elr = xr.concat(y_elr_list, dim='T')
            return edges_t.sortby('T'), y_elr.sortby('T')
        
        return labeler
