import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import os.path as osp


def EstimateCorrespondence(X, Y, t, R, dmax):

    x_transformed = SE3_transform(X, R, t)          

    target_tree = KDTree(Y)

    dists, nearest_indices = target_tree.query(x_transformed)
    mask = dists < dmax
    src_indices = np.where(mask)[0]
    tgt_indices = nearest_indices[mask]

    C = np.column_stack((src_indices, tgt_indices))
    return C


def ComputeOptimalRigidRegistration(X, Y, C):

    if len(C) == 0:
        d = X.shape[1]
        return np.eye(d), np.zeros((d, 1))

    x_matched = X[C[:, 0]]   
    y_matched = Y[C[:, 1]] 

    mu_x = x_matched.mean(axis=0)   
    mu_y = y_matched.mean(axis=0)

    x_c = x_matched - mu_x
    y_c = y_matched - mu_y

    W = y_c.T @ x_c

    try:
        U, _, Vt = np.linalg.svd(W)
    except np.linalg.LinAlgError:
        d = X.shape[1]
        return np.eye(d), np.zeros((d, 1))

    det_check = np.linalg.det(U @ Vt)
    d = X.shape[1]
    diag_correction = np.diag(np.ones(d))
    diag_correction[-1, -1] = det_check
    R = U @ diag_correction @ Vt

    t = (mu_y - R @ mu_x).reshape(-1, 1)

    return R, t


def SE3_transform(X, R, t):
    return (R @ X.T).T + t.T


def RMSE(X, Y, C):
    return np.sqrt(np.linalg.norm(X[C[:, 0]] - Y[C[:, 1]], axis=1).mean())


def ICP(X, Y, t0, R0, dmax, num_ICP_iters):

    t = t0.copy()
    R = R0.copy()

    C = np.empty((0, 2), dtype=int)

    for iteration in range(num_ICP_iters):
        C = EstimateCorrespondence(X, Y, t, R, dmax)

        if len(C) == 0:
            print(f"  Iter {iteration+1:3d}")
            break

        R, t = ComputeOptimalRigidRegistration(X, Y, C)

        rmse_val = RMSE(SE3_transform(X, R, t), Y, C)
        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"  Iter {iteration+1:3d}/{num_ICP_iters} | "
                  f"Correspondences: {len(C):5d} | RMSE: {rmse_val:.6f}")

    return R, t, C

if __name__ == "__main__":

    pt_cld_X = pd.read_csv(osp.join("data", "pclX.txt"),
                            header=None, names=["X", "Y", "Z"],
                            delimiter=" ").values
    pt_cld_Y = pd.read_csv(osp.join("data", "pclY.txt"),
                            header=None, names=["X", "Y", "Z"],
                            delimiter=" ").values


    d     = pt_cld_X.shape[1]
    R0    = np.eye(d)
    t0    = np.zeros((d, 1))
    dmax  = 0.25
    n_iter = 30

    R_est, t_est, C_final = ICP(pt_cld_X, pt_cld_Y, t0, R0, dmax, n_iter)

    RtR   = R_est.T @ R_est
    det_R = np.linalg.det(R_est)

    X_aligned   = SE3_transform(pt_cld_X, R_est, t_est)
    final_rmse  = RMSE(X_aligned, pt_cld_Y, C_final)

    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(pt_cld_Y[:, 0], pt_cld_Y[:, 1], pt_cld_Y[:, 2],
               c='red', s=1, alpha=0.5, label='Target (Y)')
    ax.scatter(X_aligned[:, 0], X_aligned[:, 1], X_aligned[:, 2],
               c='blue',  s=2, alpha=0.3, label='Aligned Source (X)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('ICP Point Cloud')
    ax.legend(markerscale=4)

    plt.tight_layout()
    plt.savefig("icp.png")
    plt.show()