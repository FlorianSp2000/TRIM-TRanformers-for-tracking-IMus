"""
Evaluation metrics for trajectory prediction. Taken from https://github.com/Sachini/ronin
"""

import numpy as np



def compute_absolute_trajectory_error(est, gt):
    return np.sqrt(np.mean((est - gt) ** 2))

def compute_relative_trajectory_error(est, gt, delta, max_delta=-1):
    """
        delta: fixed window size. If set to -1, the average of all RTE up to max_delta will be computed.
        max_delta: maximum delta. If -1 is provided, it will be set to the length of trajectories.
    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """
    if max_delta == -1:
        max_delta = est.shape[0]
    deltas = np.array([delta]) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    rtes = np.zeros(deltas.shape[0])
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[deltas[i]:] + gt[:-deltas[i]] - est[:-deltas[i]] - gt[deltas[i]:]
        rtes[i] = np.sqrt(np.mean(err ** 2))

    # The average of RTE of all window sized is returned.
    return np.mean(rtes)

def compute_ate_rte(est, gt, pred_per_min=12000):
    print(f"est.shape: {est.shape}, gt.shape: {gt.shape}")
    shape_diff = gt.shape[0] - est.shape[0]

    latest_est_entry = est[-1]
    # Step 2: Append latest_entry to g 88 times
    missing_entries = np.tile(latest_est_entry, (shape_diff, 1))
    new_est = np.concatenate((est, missing_entries), axis=0)

    ate = compute_absolute_trajectory_error(new_est, gt)
    if new_est.shape[0] < pred_per_min:
        ratio = pred_per_min / new_est.shape[0]
        rte = compute_relative_trajectory_error(new_est, gt, delta=new_est.shape[0] - 1) * ratio
    else:
        rte = compute_relative_trajectory_error(new_est, gt, delta=pred_per_min)

    return ate, rte