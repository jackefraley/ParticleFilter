import math
from operator import pos
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def initialize_particles(N):
    x = np.random.uniform(-5, 5, N)
    y = np.random.uniform(-5, 5, N)
    z = np.random.uniform(-5, 5, N)

    dirs = np.random.uniform(-1, 1, (N, 3))

    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    return np.column_stack((x, y, z, dirs))

def plot_particles(particles, true_p=None, true_d=None):
    x = particles[:, 0]
    y = particles[:, 1] 
    z = particles[:, 2]

    dx = particles[:, 3]
    dy = particles[:, 4]
    dz = particles[:, 5]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(0, 0, 0, s=120, color='black', marker='x')

    # Particles
    ax.scatter(x, y, z, s=5, alpha=0.5)
    scale = 5
    ax.quiver(x, y, z, dx*scale, dy*scale, dz*scale, length=0.2, normalize=False, alpha=0.3)

    # --- TRUE WIRE ---
    if true_p is not None and true_d is not None:
        true_p = np.array(true_p)
        true_d = np.array(true_d)
        true_d = true_d / np.linalg.norm(true_d)

        # plot a long line for the true wire
        t = np.linspace(-5, 5, 100)
        line = true_p[None, :] + t[:, None] * true_d[None, :]

        ax.plot(line[:, 0], line[:, 1], line[:, 2],
                color='red', linewidth=3, label='True wire')

        # also mark the closest point to origin
        ax.scatter(true_p[0], true_p[1], true_p[2],
                   color='red', s=50)
        
    # --- MEAN POSITION VECTOR ---
    mean_pos = np.mean(particles[:, 0:3], axis=0)

    ax.quiver(0, 0, 0,
            mean_pos[0], mean_pos[1], mean_pos[2],
            color='blue', linewidth=2, label='Mean position')

    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if true_p is not None:
        ax.legend()



    plt.show()

def update_particles(particles, weights, V_meas):
    I = 0.3
    N = particles.shape[0]

    d = particles[:, 3:6]
    p = particles[:, 0:3]

    r = -p
    dot_r = np.sum(r * d, axis=1, keepdims=True)
    r_perp = r - dot_r * d

    distance = np.linalg.norm(r_perp, axis=1) + 1e-12

    B_magnitude = I / distance
    B_direction = np.cross(d, r_perp)
    B_direction /= np.linalg.norm(B_direction, axis=1, keepdims=True) + 1e-12

    V_pred = B_magnitude[:, None] * B_direction

    # Normalize the predicted voltage 
    V_pred = V_pred / (np.linalg.norm(V_pred, axis=1, keepdims=True) + 1e-12)

    V_meas = np.array(V_meas)
    V_meas = V_meas / np.linalg.norm(V_meas)

    errors = 1 - np.sum(V_pred * V_meas, axis=1)
    mean_error = np.mean(errors)

    sigma = 0.05

    weights = np.exp(-errors / (2 * sigma**2))
    weights += 1e-12
    weights /= np.sum(weights)

    # Resample particles based on weights, higher weighted particles are selected more frequently
    idx = np.random.choice(N, N, p=weights)
    particles = particles[idx]

    particles[:, 0:3] += np.random.normal(0, 0.1, particles[:, 0:3].shape)
    particles[:, 3:6] += np.random.normal(0, 0.05, particles[:, 3:6].shape)
    particles[:, 3:6] /= np.linalg.norm(particles[:, 3:6], axis=1, keepdims=True)

    p = particles[:, 0:3]
    d = particles[:, 3:6]
    dot_pd = np.sum(p * d, axis=1, keepdims=True)
    particles[:, 0:3] = p - dot_pd * d

    return particles, mean_error

def generate_fake_measurement():
    p = np.random.uniform(-5, 5, 3)
    
    d = np.random.uniform(-1, 1, 3)
    d /= np.linalg.norm(d)

    # make p the closest point on the wire to the origin
    p = p - np.dot(p, d) * d

    # vector from wire to sensor at origin
    r = -p

    B_direction = np.cross(d, r)
    B_direction /= np.linalg.norm(B_direction) + 1e-12

    return B_direction, d, p


def main():

    N = 2000
    errors_history = []
    V_meas, d_true, r_true = generate_fake_measurement()
    particles = initialize_particles(N)
    weights = np.ones(N) / N
    plot_particles(particles, r_true, d_true)

    for i in range(50):
        noise = np.random.normal(0, 0.01, 3)
        V_meas_noisy = V_meas + noise
        V_meas_noisy /= np.linalg.norm(V_meas_noisy)

        particles, mean_error = update_particles(particles, weights, V_meas_noisy)
        errors_history.append(mean_error)

        if i % 5 == 0:
            plot_particles(particles, r_true, d_true)

    plt.figure()
    plt.plot(errors_history)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Error")
    plt.title("Particle Filter Convergence")
    plt.grid(True)
    plt.show()

    return 0    

if __name__ == "__main__":
    main()

