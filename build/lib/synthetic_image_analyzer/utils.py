import numpy as np
from sklearn.metrics import pairwise_distances

feature_metrics = {'discrete': 'hamming', 'numeric': 'euclidean'}


def compute_pairwise_distance(data_x, data_y=None, distance_type=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
        distance_type: the type for distance computing. choose from 'hamming' or 'euclidean'
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = pairwise_distances(data_x, data_y, metric=distance_type, n_jobs=8)  # 'euclidean' or 'hamming'
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k, feature_type):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features, distance_type=feature_metrics[feature_type])
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_dmin(input_features, nearest_k=0):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Mean distances of all k nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_fidelity_privacy(real_features, fake_features, nearest_k, feature_type,
                             mode_collapse_threshold):
    """
    Computes precision, precision with privacy considered, and privacy.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'.format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k, feature_type)
    # fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
    #     fake_features, nearest_k, feature_type)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features, distance_type=feature_metrics[feature_type])

    real_dmins = compute_nearest_neighbour_distances(
        real_features, 1, feature_type)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    if np.mean(np.std(fake_features, axis=0)) > mode_collapse_threshold:
        ## if there are no mode collapse
        privacy = 1 - (
                distance_real_fake <
                np.expand_dims(real_dmins, axis=1)
        ).any(axis=0).mean()
        precisionv2 = (
                (
                        distance_real_fake <
                        np.expand_dims(real_nearest_neighbour_distances, axis=1)
                ).any(axis=0)
                *
                (1 - (
                        distance_real_fake <
                        np.expand_dims(real_dmins, axis=1)
                ).any(axis=0))
        ).mean()
    else:
        ## if there are mode collapse
        privacy = 1 - (
                distance_real_fake <
                np.expand_dims(real_dmins, axis=1) + mode_collapse_threshold
        ).any(axis=0).mean()
        precisionv2 = (
                (
                        distance_real_fake <
                        np.expand_dims(real_nearest_neighbour_distances, axis=1)
                ).any(axis=0)
                *
                (1 - (
                        distance_real_fake <
                        np.expand_dims(real_dmins, axis=1) + mode_collapse_threshold
                ).any(axis=0))
        ).mean()

    return dict(precision=precision, precision_privacy=precisionv2, privacy=privacy)
