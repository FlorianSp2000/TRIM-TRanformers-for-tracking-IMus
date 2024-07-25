import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle

def plot_pred_and_tgt(pred, target):
    """
    Plots 3d trajectory of the predicted and target values with dimensions (Batch, Seq_len, 3)
    """    
    num_samples = min(pred.shape[0], 8)  # Limit to 8 samples or less
    num_cols = min(num_samples, 4)  # Maximum of 4 columns
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    fig = plt.figure(figsize=(15, 5 * num_rows))
    for i in range(num_samples):
        x_pred = pred[i*2, 1:, 0]
        y_pred = pred[i*2, 1:, 1]
        z_pred = pred[i*2, 1:, 2]

        x_target = target[i*2][:, 0]
        y_target = target[i*2][:, 1]
        z_target = target[i*2][:, 2]

        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection='3d')
        ax.plot(x_pred, y_pred, z_pred, color='blue', label='Pred', marker='o')
        ax.plot(x_target, y_target, z_target, color='red', label='Target', marker='o')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_title(f'Sample {i*2 + 1}')

        ax.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()


def plot_pred_and_tgt_without_batch(pred, target):
    """
    Plots 3d trajectory of the predicted and target values with dimensions (Seq_len, 3)
    """    

    fig = plt.figure(figsize=(15, 5))
    
    x_pred = pred[ 1:, 0]
    y_pred = pred[ 1:, 1]
    z_pred = pred[ 1:, 2]
    
    x_target = target[:, 0] - target[0, 0]
    y_target = target[:, 1] - target[0, 1]
    z_target = target[:, 2] - target[0, 2]

    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_pred, y_pred, z_pred, color='blue', label='Pred')
    ax.plot(x_target, y_target, z_target, color='red', label='Target')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    ax.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()


def plot_results_per_axis(pred, target):
    """
    Plots the predicted and target values per axis (x, y, z)
    """
    fig, ax = plt.subplots(1,3, figsize=(10,5))

    x_pred = pred[ 1:, 0]
    y_pred = pred[ 1:, 1]
    z_pred = pred[ 1:, 2]
    
    x_target = target[:, 0] - target[0, 0]
    y_target = target[:, 1] - target[0, 1]
    z_target = target[:, 2] - target[0, 2]

    ax[0].plot(x_target, label='target')
    ax[0].plot(x_pred, label='pred')
    # ax[0].plot(np.cumsum(np.array(x_naive_int)[:,0]))

    ax[1].plot(y_target, label='target')
    ax[1].plot(y_pred, label='pred')

    ax[2].plot(z_target, label='target')
    ax[2].plot(z_pred, label='pred')

    plt.legend()
    plt.show()