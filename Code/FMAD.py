import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler


def FMAD(datas, types, K, Gamma, T_max=100, Epsilon=1e-6):
    """
    FMAD: Feature Routing-Based Multi-Fuzzy Neighborhood Anomaly Detection for Heterogeneous Data

    Parameters
    ----------
    datas : ndarray, shape (n_samples, n_features)
        Input data matrix.
    types : ndarray, shape (1, n_features)
        Feature type: 0 = numeric, 1 = categorical.
    K : int
        Number of nearest neighbors, ranging from [1, 100], with a step size of 1.
    Gamma : float
       Balance factor between fsr and fssr, ranging from [0, 1], with a step size of 0.2.
    T_max : int, default=100
        Max iterations for convergence.
    Epsilon : float, default=1e-6
        Convergence tolerance (L1 norm).

    Returns
    -------
    AS : ndarray, shape (n_samples, 1)
        Anomaly score per object (higher = more anomalous).
    """
    num_fsr, cat_fsr = FSR(datas, types)

    AF_list = []
    if num_fsr is not None:
        AF_list.append(Calculate_AF(num_fsr, K=K, Gamma=Gamma, T_max=T_max, Epsilon=Epsilon))

    if cat_fsr is not None:
        AF_list.append(Calculate_AF(cat_fsr, K=K, Gamma=Gamma, T_max=0, Epsilon=Epsilon))

    scaler = MinMaxScaler()
    AS = np.sum([scaler.fit_transform(AF[:, None]) ** 2 for AF in AF_list], axis=0)
    return AS


def Calculate_AF(fsr, K, Gamma, T_max=100, Epsilon=1e-6):
    """
   Compute anomaly factor (AF) from a fuzzy similarity matrix.

   Parameters
   ----------
   fsr : ndarray, shape (n_samples, n_samples)
       Fuzzy similarity matrix (values in [0,1]).
   K : int
       Number of nearest neighbors.
   Gamma : float
       Balance factor between fsr and fssr.
   T_max : int, default=100
       Max iterations for convergence.
   Epsilon : float, default=1e-6
        Convergence tolerance (L1 norm).

   Returns
   -------
   AF : ndarray, shape (n_samples,)
       Anomaly factor per object.
   """
    kth_fsr = -np.partition(-fsr, K, axis=1)[:, K]
    FkNN = np.where(fsr >= kth_fsr[:, None], fsr, 0)
    FkNN_Card = np.sum(FkNN, axis=1)

    FMNN = np.minimum(FkNN, FkNN.T)

    temp_SUM = np.sum(FMNN, axis=1)
    temp_DIFF = squareform(pdist(FMNN, metric='cityblock'))
    fssr = (temp_SUM[:, None] + temp_SUM[None, :] - temp_DIFF) / (temp_SUM[:, None] + temp_SUM[None, :] + temp_DIFF)
    fssr[np.isnan(fssr)] = 0.0

    kth_fssr = -np.partition(-fssr, K, axis=1)[:, K]
    FSkNN = np.where(fssr >= kth_fssr[:, None], fssr, 0)
    FSkNN_Card = np.sum(FSkNN, axis=1)

    fhsr = 0.5 * (1 - Gamma) * fsr + 0.5 * Gamma * np.exp(fssr - 1)

    kth_fhsr = -np.partition(-fhsr, K, axis=1)[:, K]
    FHkNN = np.where(fhsr >= kth_fhsr[:, None], fhsr, 0)
    FHkNN_Card = np.sum(FHkNN, axis=1)

    frs = np.minimum(FHkNN, kth_fhsr)
    den = np.sum(frs, axis=1) / FHkNN_Card

    AF = np.sum(np.where(FkNN > 0, den, 0), axis=1) / (den * FkNN_Card)
    W = FSkNN / FSkNN_Card[:, None]
    t = 0
    while True:
        temp_AF = W @ AF
        if np.linalg.norm(AF - temp_AF, ord=1) <= Epsilon or t >= T_max:
            break
        else:
            t += 1
            AF = temp_AF

    return AF


def FSR(datas, types):
    """
   Separate features by type and compute fuzzy similarity matrices.

   Parameters
   ----------
   datas : ndarray, shape (n_samples, n_features)
       Input data.
   types : ndarray, shape (1, n_features)
       Type per feature: 0 = numeric, 1 = categorical.

   Returns
   -------
   num_fsr : ndarray, shape (n_samples, n_features) or None
       Fuzzy similarity matrix for numeric features (if any).
   cat_fsr : ndarray, shape (n_samples, n_features) or None
       Fuzzy similarity matrix for categorical features (if any).
       Both are in [0,1].
    """
    n, m = datas.shape

    num_fsr = None
    cat_fsr = None

    num_fea = types[0] == 0
    cat_fea = types[0] == 1

    num_dis = squareform(pdist(datas[:, num_fea], metric="cityblock")) if num_fea.any() else np.zeros((n, n))
    cat_dis = squareform(pdist(datas[:, cat_fea], metric='hamming')) if cat_fea.any() else np.zeros((n, n))

    if num_fea.any():
        num_fsr = 1 - num_dis / np.sum(num_fea)

    if cat_fea.any():
        cat_fsr = 1 - cat_dis

    return num_fsr, cat_fsr
