import math
from operator import pos
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

GAIN = 1000.0

def canonicalize_particles(particles):
    d = particles[:, 3:6]
    p = particles[:, 0:3]
    particles[:, 0:3] = p - np.sum(p * d, axis=1, keepdims=True) * d
    return particles

###############################################################################
# Initialize particles with random positions and directions
# Each particle is represented as [x, y, z, dx, dy, dz]
# x, y, z are the coordinates of the particle
# dx, dy, dz are the components of the direction vector (normalized)
##############################################################################
def initialize_particles(N):
    # Create N particles with random coordinates and directions
    x = np.random.uniform(-10, 10, N)
    y = np.random.uniform(-10, 10, N)
    z = np.random.uniform(-10, 10, N)

    # Random directions on the unit sphere
    dirs = np.random.uniform(-1, 1, (N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # Return particles as an Nx6 array: [x, y, z, dx, dy, dz]
    particles = np.column_stack((x, y, z, dirs))
    return canonicalize_particles(particles)

################################################################################
# Visualize particles in 3D space, showing their positions and directions
# Optionally, also plot the true wire position and direction if provided
################################################################################
def plot_particles(particles, true_p=None, true_d=None, sensor_pos1=None, sensor_pos2=None):
    # Extract positions from particles
    x = particles[:, 0]
    y = particles[:, 1] 
    z = particles[:, 2]

    # Extract directions from particles
    dx = particles[:, 3]
    dy = particles[:, 4]
    dz = particles[:, 5]

    # Create a new 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Mark the sensor position at the origin
    ax.scatter(sensor_pos1[0], sensor_pos1[1], sensor_pos1[2], s=120, color='black', marker='x')
    ax.scatter(sensor_pos2[0], sensor_pos2[1], sensor_pos2[2], s=120, color='black', marker='x')

    # Plot particles as points and their directions as arrows
    # Adjust scale and alpha for better visibility
    ax.scatter(x, y, z, s=5, alpha=0.5)
    scale = 5
    ax.quiver(x, y, z, dx*scale, dy*scale, dz*scale, length=0.2, normalize=False, alpha=0.3)

    # Plot the true wire if provided
    if true_p is not None and true_d is not None:

        # Convert to numpy arrays and normalize direction
        true_p = np.array(true_p)
        true_d = np.array(true_d)
        true_d = true_d / np.linalg.norm(true_d)

        # plot a long line for the true wire
        t = np.linspace(-10, 10, 100)
        line = true_p[None, :] + t[:, None] * true_d[None, :]
        ax.plot(line[:, 0], line[:, 1], line[:, 2],
                color='red', linewidth=3, label='True wire')

        # also mark the closest point to origin
        ax.scatter(true_p[0], true_p[1], true_p[2],
                   color='red', s=50)
        
    # Plot the mean position of the particles as a blue arrow from the origin
    mean_pos = np.mean(particles[:, 0:3], axis=0)
    direction = mean_pos - sensor_pos1

    ax.quiver(sensor_pos1[0], sensor_pos1[1], sensor_pos1[2],
            direction[0], direction[1], direction[2],
            color='blue', linewidth=2, label='Mean position')
    
    ax.plot(
        sensor_pos1[0],
        sensor_pos1[1],
        sensor_pos1[2],
        color='red',
        linewidth=2,
        label='Sensor Path'
)

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1,1,1])

    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add legend if true wire is plotted
    if true_p is not None:
        ax.legend()

    # Show plot
    plt.show()

#################################################################################
# Compute the predicted magnetic field vector for every particle at one sensor
#################################################################################
def predict_field(particles, sensor_pos, current):
    d = particles[:, 3:6]
    p = particles[:, 0:3]

    # Distance from each particle to the sensor
    r = sensor_pos - p

    # Project r onto the direction d to find the perpendicular component
    dot_r = np.sum(r * d, axis=1, keepdims=True)
    r_perp = r - dot_r * d
    distance = np.linalg.norm(r_perp, axis=1) + 1e-12

    return (current * np.cross(d, r_perp) / (distance[:, None]**2 + 1e-12)) * GAIN

#################################################################################
# Update particles using the fixed two-sensor measurement snapshot
#################################################################################
def update_particles(particles, V_meas1, V_meas2, sensor_pos1, sensor_pos2, current, sigma, position_noise, direction_noise):

    # Number of particles
    N = particles.shape[0]

    B1_pred = predict_field(particles, sensor_pos1, current)
    B2_pred = predict_field(particles, sensor_pos2, current)

    errors1 = np.sum((B1_pred - V_meas1)**2, axis=1)
    errors2 = np.sum((B2_pred - V_meas2)**2, axis=1)
    total_error = errors1 + errors2
    mean_error = np.mean(total_error)
    best_idx = np.argmin(total_error)
    best_particle = particles[best_idx].copy()
    best_error = total_error[best_idx]

    # Recompute weights from scratch each round so the same static measurement
    # snapshot is used for refinement, not compounded as new evidence.
    log_weights = -total_error / (2 * sigma**2)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weight_sum = np.sum(weights)
    if not np.isfinite(weight_sum) or weight_sum == 0:
        weights = np.ones(N) / N
    else:
        weights /= weight_sum

    # Resample particles based on weights, higher weighted particles are selected more frequently
    idx = np.random.choice(N, N, p=weights)
    particles = particles[idx]

    particles[:, 0:3] += np.random.normal(0, position_noise, particles[:, 0:3].shape)
    particles[:, 3:6] += np.random.normal(0, direction_noise, particles[:, 3:6].shape)
    particles[:, 3:6] /= np.linalg.norm(particles[:, 3:6], axis=1, keepdims=True)
    particles = canonicalize_particles(particles)

    return particles, mean_error, best_error, best_particle

#################################################################################
# Generates a fake measurement of the magnetic field vector
#################################################################################
def generate_fake_measurement(sensor_pos, p_true, d_true, current):

    r = sensor_pos - p_true
    dot_r = np.dot(r, d_true)
    r_perp = r - dot_r * d_true

    B = current * np.cross(d_true, r_perp) / (np.linalg.norm(r_perp)**2 + 1e-12)
    return B * GAIN, r_perp


def main():

    # Number of particles
    N = 10000
    current = 2
    rounds = 50

    sensor_pos1 = np.array([0, 0, 0])
    sensor_pos2 = np.array([2, 0, 0])

    # Store the history of static refinement for plotting convergence
    errors_history = []
    best_error_history = []
    sigma_history = []
    particles_history = []

    # Randomly generate a wire position and direction
    p_true = np.random.uniform(-10, 10, 3)
    d_true = np.random.uniform(-1, 1, 3)
    d_true /= np.linalg.norm(d_true)
    p_true = p_true - np.dot(p_true, d_true) * d_true

    V_meas1, _ = generate_fake_measurement(sensor_pos1, p_true, d_true, current)
    V_meas2, _ = generate_fake_measurement(sensor_pos2, p_true, d_true, current)

    # Initialize particles once, then repeatedly refine them using the same
    # fixed two-sensor snapshot.
    particles = initialize_particles(N)
    particles_history.append(particles.copy())

    plot_particles(particles, p_true, d_true, sensor_pos1, sensor_pos2)

    for i in range(rounds):
        alpha = i / max(rounds - 1, 1)
        sigma = (0.2 - 0.18 * alpha) * GAIN
        position_noise = 0.4 - 0.35 * alpha
        direction_noise = 0.05 - 0.04 * alpha

        particles, mean_error, best_error, _ = update_particles(
            particles,
            V_meas1,
            V_meas2,
            sensor_pos1,
            sensor_pos2,
            current=current,
            sigma=sigma,
            position_noise=position_noise,
            direction_noise=direction_noise,
        )
        errors_history.append(mean_error)
        best_error_history.append(best_error)
        sigma_history.append(sigma)
        particles_history.append(particles.copy())

    # Final plot of particles after convergence, showing true wire position/direction
    plot_particles(particles, p_true, d_true, sensor_pos1, sensor_pos2)

    plt.figure()
    plt.plot(errors_history, label="Mean Error")
    plt.plot(best_error_history, label="Best Error")
    plt.plot(sigma_history, label="Sigma")
    plt.xlabel("Refinement Round")
    plt.ylabel("Static Search Score")
    plt.title("Static Two-Sensor Particle Refinement")
    plt.legend()
    plt.grid(True)
    plt.show()

    t = np.linspace(-10, 10, 100)
    line = p_true[None, :] + t[:, None] * d_true[None, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()

        particles = particles_history[frame]

        x = particles[:, 0]
        y = particles[:, 1]
        z = particles[:, 2]

        dx = particles[:, 3]
        dy = particles[:, 4]
        dz = particles[:, 5]

        ax.scatter(x, y, z, s=5, alpha=0.5)
        ax.quiver(x, y, z, dx, dy, dz, length=0.2, alpha=0.3)

        # sensor
        ax.scatter(*sensor_pos1, color='black', s=50, marker='x')
        ax.scatter(*sensor_pos2, color='black', s=50, marker='x')

        # mean arrow (fixed version)
        mean_pos = np.mean(particles[:, 0:3], axis=0)
        direction = mean_pos - sensor_pos1
        ax.quiver(*sensor_pos1, *direction, color='blue')

        ax.plot(line[:, 0], line[:, 1], line[:, 2],
            color='red', linewidth=3)
        
        ax.scatter(*p_true, color='red', s=50)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        ax.set_box_aspect([1,1,1])

    ani = FuncAnimation(fig, update, frames=len(particles_history), interval=100)
    #ani.save("pf.mp4", fps=10)

    plt.show()

    return 0    

if __name__ == "__main__":
    main()
