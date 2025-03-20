

from matplotlib.patches import Patch, Polygon
from operator import sub
import matplotlib.pyplot as plt
import numpy as np



def compute_reliability_score(y_true, y_pred, num_bins=10):
    """
    Compute the reliability score for probabilistic forecasts.

    Parameters:
    y_true (numpy array): Array of true binary outcomes (0 or 1).
    y_pred (numpy array): Array of predicted probabilities (between 0 and 1).
    num_bins (int): Number of bins to use for grouping predicted probabilities.

    Returns:
    float: The reliability score.
    """
    
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Initialize arrays to hold the average predicted probabilities and observed frequencies for each bin
    bin_avg_pred = np.zeros(num_bins)
    bin_avg_true = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    # Bin edges
    bin_edges = np.linspace(0, 1, num_bins + 1)
    
    # Assign each prediction to a bin
    bin_indices = np.digitize(y_pred, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Ensure indices are within valid range
    
    # Calculate bin statistics
    for i in range(num_bins):
        # Indices of predictions in the current bin
        bin_mask = (bin_indices == i)
        bin_count = np.sum(bin_mask)
        
        if bin_count > 0:
            # Average predicted probability for the current bin
            bin_avg_pred[i] = np.mean(y_pred[bin_mask])
            # Observed frequency (empirical probability) for the current bin
            bin_avg_true[i] = np.mean(y_true[bin_mask])
            bin_counts[i] = bin_count
    
    # Calculate the reliability score
    reliability_score = np.nansum((bin_avg_pred - bin_avg_true)**2 * bin_counts) / np.sum(bin_counts)
    
    return reliability_score

def compute_brier_skill_score(y_pred, t):
    """
    Compute the Brier Skill Score (BSS) for probabilistic forecasts.

    Parameters:
    y_pred (numpy array): Predicted probabilities.
    t (numpy array): True binary outcomes (0 or 1).

    Returns:
    float: The Brier Skill Score (BSS).
    """
    # Adjust y_pred to avoid edge cases
    y_pred = y_pred * 0.9999999999999
    
    # Mask to remove NaNs
    msk = np.where(~np.isnan(y_pred + t))
    y_pred = y_pred[msk]
    t = t[msk]
    
    # Calculate the base rate (climatological mean)
    base_rate = np.nanmean(t)
    base_rate = np.ones_like(np.nanmean(t)) * 1/3
    
    # Calculate the Brier Score (BS)
    bs = np.nanmean((y_pred - t)**2)
    
    # Calculate the Brier Score for the reference forecast (BR)
    br = np.nanmean((base_rate - t)**2)
    
    # Calculate the Brier Skill Score (BSS)
    bss = 1 - (bs / br)
    
    return bss

def compute_resolution_score(predictions, observations, num_bins=10):
    """
    Compute the resolution score for probabilistic forecasts.

    Parameters:
    predictions (numpy array): Predicted probabilities.
    observations (numpy array): True binary outcomes (0 or 1).
    num_bins (int): Number of bins to use for grouping predicted probabilities.

    Returns:
    float: The resolution score.
    """
    # Adjust predictions to avoid edge cases
    predictions = predictions * 0.9999999999999
    
    # Mask to remove NaNs
    mask = np.where(~np.isnan(predictions + observations))
    predictions = predictions[mask]
    observations = observations[mask]
    
    # Calculate the base rate (climatological mean)
    base_rate = np.nanmean(observations)
    
    # Initialize arrays to hold the average observed frequencies and counts for each bin
    bin_obs_freq = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    # Bin edges
    bin_edges = np.linspace(0, 1, num_bins + 1)
    
    # Assign each prediction to a bin
    bin_indices = np.digitize(predictions, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Ensure indices are within valid range
    
    # Calculate bin statistics
    for i in range(num_bins):
        bin_mask = (bin_indices == i)
        bin_counts[i] = bin_mask.sum()
        
        if bin_counts[i] > 0:
            bin_obs_freq[i] = observations[bin_mask].mean()
    
    # Calculate the resolution score
    resolution = np.nansum(bin_counts * (bin_obs_freq - base_rate)**2) / np.sum(bin_counts)
    
    return resolution

def reliability_diagram(ypred, t, title=None, perfect_reliability_line=True, plot_hist=True, fig=None, ax=None, bin_minimum_pct=0.01, tercile_skill_area=True, scores=True):
    """
    Compute and plot a reliability diagram (calibration curve) with a normalized histogram of the forecast probabilities.

    Parameters:
    ypred (numpy array): Predicted probabilities.
    t (numpy array): True binary outcomes (0 or 1).
    title (str): Title for the plot.
    perfect_reliability_line (bool): Whether to plot the perfect reliability line.
    plot_hist (bool): Whether to plot the histogram of forecast probabilities.
    fig (matplotlib Figure): Figure object to plot on.
    ax (matplotlib Axes): Axes object to plot on.
    bin_minimum_pct (float): Minimum percentage of samples in a bin to include it in the plot.
    """


    ypred = ypred * 0.9999999999999  # Avoids edge cases with digitize
    assert ypred.shape == t.shape, 'Inconsistent shapes between ypred and t - {} vs {}'.format(ypred.shape, t.shape)

    # Mask to remove NaNs
    msk = np.where(~np.isnan(ypred + t))
    ypred = ypred[msk]
    t = t[msk]

    # Total number of non-NaN samples
    countnonnan = np.ones_like(ypred).sum()

    # Initialize arrays
    bin_avg_pred = np.zeros(10)
    bin_obs_freq = np.zeros(10)
    bin_counts = np.zeros(10)

    # Iterate over bins
    for i in range(10):
        bin_mask = (ypred >= i / 10.0) & (ypred < (i / 10.0 + 0.1))
        bin_counts[i] = bin_mask.sum()
        
        if bin_counts[i] > 0:
            bin_avg_pred[i] = ypred[bin_mask].mean()
            bin_obs_freq[i] = t[bin_mask].mean()

    # Compute bin centers
    bin_centers = (np.arange(10) + 0.5) / 10.0

    # Mask bins with insufficient data
    valid_bins = bin_counts / countnonnan >= bin_minimum_pct
    bin_centers = bin_centers[valid_bins]
    bin_avg_pred = bin_avg_pred[valid_bins]
    bin_obs_freq = bin_obs_freq[valid_bins]
    bin_counts = bin_counts[valid_bins]

    # Normalize bin counts for the histogram
    bin_counts = bin_counts / countnonnan

    if ax is None:
        fig, ax = plt.subplots()

    b1, t1 = ax.set_ylim(0, 1)
    l, r = ax.set_xlim(0, 1)

    #plt.hist(epoelm_xval[:, 0], bins=11)
    if tercile_skill_area:
        ur = Polygon([[0.33, 0.33 ], [0.33, 1], [1,1], [1, 1.33/2.0]], facecolor='gray', alpha=0.25)
        bl = Polygon([[0.33, 0.33 ], [0.33, 0], [0,0], [0, 0.33/2.0]], facecolor='gray', alpha=0.25)
        ax.add_patch(ur)
        ax.add_patch(bl)

        ax.text(0.66, 0.28, 'No Resolution')
        noresolution = ax.plot([0, 1], [0.33,0.33], lw=0.5, linestyle='dotted')

        noskill = ax.plot([0, 1], [0.33/2.0,1.33/2.0], lw=0.5, linestyle='dotted')
        figW, figH = ax.get_figure().get_size_inches()
        _, _, w, h = ax.get_position().bounds
        disp_ratio = (figH * h) / (figW * w)
        data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
        angle = (180.0/np.pi)*np.arctan(disp_ratio / data_ratio)
        ax.text(0.66, 0.45, 'No Skill', rotation=angle*0.5)

        ax.plot([0.33, 0.33], [0,1], lw=0.5, linestyle='dotted')

    # Plot the reliability diagram
    ax.plot(bin_centers, bin_obs_freq, marker='o', linestyle='-', color='red', label='Observed Frequency')

    # Plot the perfect reliability line
    if perfect_reliability_line:
        ax.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Perfect Reliability')

    # Plot normalized histogram
    if plot_hist:
        ax.bar(bin_centers, bin_counts, width=0.1, color='violet', alpha=0.5, label='Normalized Counts')

    if scores:
        bss = compute_brier_skill_score(ypred, t)
        rel = compute_reliability_score(t, ypred)
        res = compute_resolution_score(ypred, t)
        ax.text( 0.7, 0.11, 'BSS: {:0.04f}'.format(bss))
        ax.text( 0.7, 0.06, 'REL: {:0.04f}'.format(rel))
        ax.text( 0.7, 0.01, 'RES: {:0.04f}'.format(res))

    ax.set_xlabel('Forecast Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    if title is not None:
        ax.set_title(title)
    ax.legend(loc='upper left')


    plt.show()
