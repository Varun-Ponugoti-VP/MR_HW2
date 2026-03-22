import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from scipy.linalg import expm
import os.path as osp


class EKF(object):

    def __init__(self, dt, x0, P0, R, Q, landmarks):
        self.dt = dt
        self.R = R
        self.Q = Q
        self.I = np.eye(2)
        self.landmarks = [np.array(l).reshape(-1, 1) for l in landmarks]
        self.n = len(self.landmarks)
        self.x = x0.copy()
        self.P = P0.copy()

    def predict(self, v):

        v = np.array(v).reshape(-1, 1)
        self.x = self.x + self.dt * v
        self.P = self.P + self.R  

    def update(self, z):

        z = np.array(z).reshape(-1, 1)

        h = np.array([
            np.linalg.norm(self.x - lm)
            for lm in self.landmarks
        ]).reshape(-1, 1)

        H = EKF._H(self.x, self.landmarks)

        y = z - h
        S = H @ self.P @ H.T + self.Q
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y

        self.P = (self.I - K @ H) @ self.P

    @staticmethod
    def _H(x, landmarks):
        x = np.array(x).reshape(-1, 1)
        H = []
        for lm in landmarks:
            diff = x - lm 
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                H.append([0.0, 0.0])
            else:
                H.append([diff[0, 0] / dist, diff[1, 0] / dist])
        return np.array(H)


def measurement(x, landmarks, Q):

    x = np.array(x).reshape(-1, 1)
    z_true = np.array([
        np.linalg.norm(x - np.array(lm).reshape(-1, 1))
        for lm in landmarks
    ])
    noise = np.random.multivariate_normal(np.zeros(len(landmarks)), Q)
    return (z_true + noise).reshape(-1, 1)


def transition(x, v, dt, R):
 
    x = np.array(x).reshape(-1, 1)
    v = np.array(v).reshape(-1, 1)
    noise = np.random.multivariate_normal(np.zeros(2), R).reshape(-1, 1)
    return x + dt * v + noise


def plot_confidence_ellipse(ax, mean, cov, n_sigma=np.sqrt(7.378), **kwargs):
    mean = np.asarray(mean).ravel()
    cov = np.asarray(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    width  = 2 * n_sigma * np.sqrt(eigvals[0])
    height = 2 * n_sigma * np.sqrt(eigvals[1])
    angle  = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    ellipse = Ellipse(xy=mean, width=width, height=height,
                      angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


if __name__ == "__main__":

    l1 = np.array([5,  5]).reshape(-1, 1)
    l2 = np.array([-5, 5]).reshape(-1, 1)
    landmarks = [l1, l2]

    R  = 0.1 * np.eye(2)   
    Q  = 0.5 * np.eye(2)   
    dt = 0.5

    x   = np.array([0, 0]).reshape(-1, 1)         
    x0  = np.random.multivariate_normal(            
              np.zeros(2), np.eye(2)).reshape(-1, 1)
    P0  = np.eye(2)

    v1 = np.array([ 1,  0]).reshape(-1, 1)
    v2 = np.array([ 0, -1]).reshape(-1, 1)
    v3 = np.array([-1,  0]).reshape(-1, 1)
    v4 = np.array([ 0,  1]).reshape(-1, 1)
    vs = [v1, v2, v3, v4]

    ts = [
        np.arange(0,        10 + dt, dt),
        np.arange(10 + dt,  20 + dt, dt),
        np.arange(20 + dt,  30 + dt, dt),
        np.arange(30 + dt,  40 + dt, dt),
    ]

    ekf = EKF(dt=dt, x0=x0, P0=P0, R=R, Q=Q, landmarks=landmarks)

    true_path = [x.copy()]
    est_path  = [ekf.x.copy()]
    cov_hist  = [ekf.P.copy()]
    time_hist = [0.0]

    for v, t_array in zip(vs, ts):
        for t in t_array:
            x = transition(x, v, dt, R)
            ekf.predict(v)

            z = measurement(x, landmarks, Q)

            ekf.update(z)

            true_path.append(x.copy())
            est_path.append(ekf.x.copy())
            cov_hist.append(ekf.P.copy())
            time_hist.append(t + dt)

    true_path = np.array([p.ravel() for p in true_path])   
    est_path  = np.array([p.ravel() for p in est_path])    
    time_hist = np.array(time_hist)

    pos_err = np.linalg.norm(true_path - est_path, axis=1)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(true_path[:, 0], true_path[:, 1],
             'b.-', lw=1.2, label='True Path')
    ax1.plot(est_path[:, 0],  est_path[:, 1],
             'g.--', lw=1.2, label='EKF Estimate')

    for i in range(0, len(cov_hist), 10):
        plot_confidence_ellipse(
            ax1,
            mean=est_path[i],
            cov=cov_hist[i],
            n_sigma=np.sqrt(7.378), 
            edgecolor='orange', facecolor='none',
            linewidth=0.8, alpha=0.7, zorder=3
        )

    lm_x = [l.ravel()[0] for l in landmarks]
    lm_y = [l.ravel()[1] for l in landmarks]
    ax1.plot(lm_x, lm_y, 'r*', markersize=14, label='Landmarks')
    ax1.plot(0, 0, 'ko', markersize=8, label='Start')
    ax1.set_title('True vs Estimated Trajectories')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    ax1.legend()
    ax1.grid(alpha=0.4)

    ax2.plot(time_hist, pos_err, 'r-', lw=2)
    ax2.set_title('Position Error over Time')
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Error')
    ax2.grid(alpha=0.4)

    plt.savefig("ekf.png")
    plt.show()

