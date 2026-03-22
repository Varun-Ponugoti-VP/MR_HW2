import numpy as np
from scipy.linalg import expm
import matplotlib 
import matplotlib.pyplot as plt

class DifferentialDriveParticleFilter:

    def __init__(self, dt, wheel_radius, track_width,
                 cmd_left, cmd_right,
                 sigma_left, sigma_right,
                 sigma_pos, n_particles):
        self.dt = dt
        self.r = float(wheel_radius)
        self.w = float(track_width)
        self.cmd_left = float(cmd_left)
        self.cmd_right = float(cmd_right)
        self.sigma_left = float(sigma_left)
        self.sigma_right = float(sigma_right)
        self.sigma_pos = float(sigma_pos)
        self.n_particles = n_particles
        self.weights = np.ones(n_particles) / n_particles

    def _sample_noisy_wheel_speeds(self):

        wl = np.random.normal(self.cmd_left,  self.sigma_left)
        wr = np.random.normal(self.cmd_right, self.sigma_right)
        return wl, wr

    @staticmethod
    def _hat_map(wl, wr, r, w):

        return np.array([[ 0, -(r / w) * (wr - wl),  (r / 2) * (wr + wl)], [ (r / w) * (wr - wl), 0, 0 ], [ 0, 0, 0 ]], dtype=float)

    @staticmethod
    def _extract_position(pose):

        return pose[:2, -1]

    # ------------------------------------------------------------------
    # (c) Propagation
    # ------------------------------------------------------------------

    def propagate(self, particles):

        new_particles = []
        for x_t in particles:
            wl, wr = self._sample_noisy_wheel_speeds()
            lie_element = self._hat_map(wl, wr, self.r, self.w)
            x_t_plus = x_t @ expm(self.dt * lie_element)
            new_particles.append(x_t_plus)
        return new_particles

    # ------------------------------------------------------------------
    # (d) Measurement update
    # ------------------------------------------------------------------

    def update(self, particles, z):

        weights = np.zeros(self.n_particles)

        for i, x_t in enumerate(particles):
            l_t = self._extract_position(x_t)          
            residual = z - l_t
            exponent = -0.5 * np.dot(residual, residual) / (self.sigma_pos ** 2)
            weights[i] = np.exp(exponent)              

        weights /= weights.sum()

        indices = np.random.choice(self.n_particles,
                                   size=self.n_particles,
                                   replace=True,
                                   p=weights)
        resampled = [particles[idx] for idx in indices]
        return resampled

    @staticmethod
    def positions_array(particles):
        return np.array([p[:2, -1] for p in particles])

    @staticmethod
    def print_statistics(positions):
        mean = positions.mean(axis=0)
        cov  = np.cov(positions.T)
        fmt  = {"float": "{:10.6f}".format}
        print("Mean:")
        print(np.array2string(mean, precision=6,
                              suppress_small=True, formatter=fmt))
        print("\nCovariance:")
        print(np.array2string(cov, precision=6,
                              suppress_small=True, formatter=fmt))

if __name__ == "__main__":
    cmd_left   = 1.5
    cmd_right  = 2.0
    r          = 0.25
    w          = 0.5
    sigma_l    = 0.05
    sigma_r    = 0.05
    sigma_p    = 0.10
    n_particles = 1000
    dt_pf      = 5.0

    def identity_particles(n):
        return [np.eye(3) for _ in range(n)]

    pf = DifferentialDriveParticleFilter(
        dt=dt_pf, wheel_radius=r, track_width=w,
        cmd_left=cmd_left, cmd_right=cmd_right,
        sigma_left=sigma_l, sigma_right=sigma_r,
        sigma_pos=sigma_p, n_particles=n_particles
    )

    # ------------------------------------------------------------------
    # Part (e): single propagation step, dt=10 => two steps of dt=5
    # ------------------------------------------------------------------

    pf_e = DifferentialDriveParticleFilter(
        dt=dt_pf, wheel_radius=r, track_width=w,
        cmd_left=cmd_left, cmd_right=cmd_right,
        sigma_left=sigma_l, sigma_right=sigma_r,
        sigma_pos=sigma_p, n_particles=n_particles
    )

    particles_e = identity_particles(n_particles)
    particles_e = pf_e.propagate(particles_e) 
    particles_e = pf_e.propagate(particles_e) 

    positions_e = pf_e.positions_array(particles_e)
    pf_e.print_statistics(positions_e)

    fig_e, ax_e = plt.subplots(figsize=(6, 5))
    ax_e.scatter(positions_e[:, 0], positions_e[:, 1],
                 s=4, alpha=0.5, color='black')
    ax_e.set_title("Particle positions at t=10")
    ax_e.set_xlabel("x")
    ax_e.set_ylabel("y")
    ax_e.set_aspect("equal")
    ax_e.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.savefig("./e.png", dpi=150)
    plt.show()
    print()

    # ------------------------------------------------------------------
    # Part (f): dead reckoning, 4 propagation steps (t=5,10,15,20)
    # ------------------------------------------------------------------

    print("Dead reckoning")

    pf_f = DifferentialDriveParticleFilter(
        dt=dt_pf, wheel_radius=r, track_width=w,
        cmd_left=cmd_left, cmd_right=cmd_right,
        sigma_left=sigma_l, sigma_right=sigma_r,
        sigma_pos=sigma_p, n_particles=n_particles
    )

    colors_f   = ['blue', 'green', 'black', 'red']
    times_f    = [5, 10, 15, 20]
    particles_f = identity_particles(n_particles)
    fig_f, ax_f = plt.subplots(figsize=(7, 6))

    for step, t in enumerate(times_f):
        particles_f = pf_f.propagate(particles_f)
        positions_f = pf_f.positions_array(particles_f)

        print(f"\n  t = {t}:")
        pf_f.print_statistics(positions_f)

        ax_f.scatter(positions_f[:, 0], positions_f[:, 1],
                     s=4, alpha=0.45, color=colors_f[step], label=f"t={t}")

    ax_f.set_title("Dead-reckoning")
    ax_f.set_xlabel("x")
    ax_f.set_ylabel("y")
    ax_f.set_aspect("equal")
    ax_f.legend(markerscale=3)
    ax_f.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.savefig("./f.png", dpi=150)
    plt.show()
    print()

    # ------------------------------------------------------------------
    # Part (g): propagation + measurement update, t in {5,10,15,20}
    # ------------------------------------------------------------------

    print("Propagate+update:")

    measurements = [
        np.array([1.6561,  1.2847]),
        np.array([1.0505,  3.1059]),
        np.array([-0.9875, 3.2118]),
        np.array([-1.6450, 1.1978]),
    ]

    pf_g = DifferentialDriveParticleFilter(
        dt=dt_pf, wheel_radius=r, track_width=w,
        cmd_left=cmd_left, cmd_right=cmd_right,
        sigma_left=sigma_l, sigma_right=sigma_r,
        sigma_pos=sigma_p, n_particles=n_particles
    )

    colors    = ['blue', 'red', 'green', 'yellow']
    time_steps   = [5, 10, 15, 20]
    particles_n = identity_particles(n_particles)
    fig_g, ax_g = plt.subplots(figsize=(7, 6))

    for step, (t, z) in enumerate(zip(time_steps, measurements)):
        particles_n = pf_g.propagate(particles_n)
        particles_n = pf_g.update(particles_n, z)
        positions_n = pf_g.positions_array(particles_n)

        print(f"\n  t = {t}  (measurement z = {z}):")
        pf_g.print_statistics(positions_n)

        ax_g.scatter(positions_n[:, 0], positions_n[:, 1],
                     s=4, alpha=0.5, color=colors[step], label=f"t={t}")
        ax_g.scatter(*z, marker='*', s=120, color=colors[step],
                     edgecolors='k', linewidths=0.5, zorder=5)

    ax_g.set_title("Positions of particles at {}")
    ax_g.set_xlabel("x (m)")
    ax_g.set_ylabel("y (m)")
    ax_g.set_aspect("equal")
    ax_g.legend(markerscale=3)
    ax_g.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.savefig("./g.png", dpi=150)
    plt.show()