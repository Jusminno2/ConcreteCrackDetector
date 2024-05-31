import numpy as np

def kittler_threshold(image):
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Calculate cumulative sums
    cumsum = np.cumsum(histogram)
    cumsum_back = cumsum[-1] - cumsum

    # Calculate cumulative means
    cumsum_mean = np.cumsum(histogram * bin_centers)
    cumsum_mean_back = cumsum_mean[-1] - cumsum_mean

    # Calculate variances
    var_back = np.zeros_like(histogram, dtype=float)
    var_fore = np.zeros_like(histogram, dtype=float)

    valid_mask = cumsum > 0
    var_back[valid_mask] = (cumsum_mean[valid_mask] ** 2) / cumsum[valid_mask]
    valid_mask = cumsum_back > 0
    var_fore[valid_mask] = (cumsum_mean_back[valid_mask] ** 2) / cumsum_back[valid_mask]

    # Calculate the minimum error threshold
    sigma_b_squared = (cumsum_mean ** 2) / cumsum - (cumsum_mean_back ** 2) / cumsum_back
    min_sigma_b_squared_index = np.nanargmin(sigma_b_squared)
    threshold = bin_centers[min_sigma_b_squared_index]

    return threshold
