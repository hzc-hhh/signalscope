"""
Medical semantic mapping.

Map learned latent representations to known physiological concepts.
"""


import numpy as np
from scipy import stats


def latent_correlation(
    latent_vectors: np.ndarray,
    physiological_labels: dict[str, np.ndarray],
    method: str = "pearson",
) -> dict[str, dict[int, float]]:
    """
    Correlate each latent dimension with known physiological variables.

    Helps interpret what the model has learned: which dimensions
    correspond to heart rate, respiration rate, motion artifacts, etc.

    Parameters
    ----------
    latent_vectors : np.ndarray, shape (n_samples, n_latent_dims)
        Encoded latent representations.
    physiological_labels : dict
        {'heart_rate': array, 'respiration_rate': array, ...}
    method : str
        'pearson' or 'spearman' correlation.

    Returns
    -------
    dict
        {label_name: {dim_index: correlation_coefficient}}
    """
    corr_fn = stats.pearsonr if method == "pearson" else stats.spearmanr

    results = {}
    for label_name, label_values in physiological_labels.items():
        dim_corrs = {}
        for dim in range(latent_vectors.shape[1]):
            latent_dim = latent_vectors[:, dim]
            valid = ~(np.isnan(latent_dim) | np.isnan(label_values))
            if valid.sum() > 2:
                r, p = corr_fn(latent_dim[valid], label_values[valid])
                dim_corrs[dim] = float(r)
        results[label_name] = dim_corrs

    return results
