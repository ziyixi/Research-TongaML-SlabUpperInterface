import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_surface(data, sigma=10.0):
    # Apply Gaussian filter for smoothing
    return gaussian_filter(data, sigma=sigma)


def smooth_across_sublists(slab_interfaces, smoothing_sigma=10.0):
    # Convert to NumPy array
    data_array = np.array(slab_interfaces).T  # Transpose to process along the columns

    # Apply smoothing
    smoothed_data = smooth_surface(data_array, sigma=smoothing_sigma)

    # Transpose back to match the original structure and convert to list of lists
    return smoothed_data.T.tolist()
