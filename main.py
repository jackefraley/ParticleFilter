import math
from operator import pos
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

###############################################################################
# Initialize particles with random positions and directions
# Each particle is represented as [x, y, z, dx, dy, dz]
# x, y, z are the coordinates of the particle
# dx, dy, dz are the components of the direction vector (normalized)
##############################################################################
def initialize_particles(N):
    # Create N particles with random coordinates and directions
    x = np.random.uniform(-5, 5, N)
    y = np.random.uniform(-5, 5, N)
    z = np.random.uniform(-5, 5, N)

    # Random directions on the unit sphere
    dirs = np.random.uniform(-1, 1, (N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # Return particles as an Nx6 array: [x, y, z, dx, dy, dz]
    return np.column_stack((x, y, z, dirs))

################################################################################
# Visualize particles in 3D space, showing their positions and directions
# Optionally, also plot the true wire position and direction if provided
################################################################################
def plot_particles(particles, true_p=None, true_d=None, sensor_positions=None, sensor_pos=None):
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
    ax.scatter(sensor_pos[0], sensor_pos[1], sensor_pos[2], s=120, color='black', marker='x')

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
        t = np.linspace(-5, 5, 100)
        line = true_p[None, :] + t[:, None] * true_d[None, :]
        ax.plot(line[:, 0], line[:, 1], line[:, 2],
                color='red', linewidth=3, label='True wire')

        # also mark the closest point to origin
        ax.scatter(true_p[0], true_p[1], true_p[2],
                   color='red', s=50)
        
    # Plot the mean position of the particles as a blue arrow from the origin
    mean_pos = np.mean(particles[:, 0:3], axis=0)
    direction = mean_pos - sensor_pos

    ax.quiver(sensor_pos[0], sensor_pos[1], sensor_pos[2],
            direction[0], direction[1], direction[2],
            color='blue', linewidth=2, label='Mean position')
    
    ax.plot(
        sensor_positions[:, 0],
        sensor_positions[:, 1],
        sensor_positions[:, 2],
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
# Update particles based on the measured voltage vector V_meas
# This function computes the predicted voltage for each particle,
# compares it to the measured voltage, and updates particle weights accordingly
#################################################################################
def update_particles(particles, weights, V_meas, sensor_pos):
    # Current of the wire mA
    I = 0.3

    # Number of particles
    N = particles.shape[0]

    # Extract direction and position from particles
    d = particles[:, 3:6]
    p = particles[:, 0:3]

    # Vector from wire to sensor at origin assuming sensor is at origin
    r = sensor_pos - p

    # Compute component of r perpendicular to d
    dot_r = np.sum(r * d, axis=1, keepdims=True)
    r_perp = r - dot_r * d

    # Compute the magnitude of perpendicular vector
    distance = np.linalg.norm(r_perp, axis=1) + 1e-12

    # Compute the predicted magnetic field vector at the sensor using Biot-Savart law
    B_magnitude = I / distance
    B_direction = np.cross(d, r_perp)
    B_direction /= np.linalg.norm(B_direction, axis=1, keepdims=True) + 1e-12
    V_pred = B_magnitude[:, None] * B_direction

    # Normalize the predicted voltage 
    V_pred = V_pred / (np.linalg.norm(V_pred, axis=1, keepdims=True) + 1e-12)

    # Normalize the measured voltage
    V_meas = np.array(V_meas)
    V_meas = V_meas / np.linalg.norm(V_meas)

    # Compute error as 1 - cosine similarity between predicted and measured voltage vectors
    errors = 1 - np.abs(np.sum(V_pred * V_meas, axis=1))

    # Compute mean error for monitoring convergence
    mean_error = np.mean(errors)

    # Sigma controls how sharply the particles are weighted based on their error
    # larger sigma means more uniform weights
    sigma = 0.05

    # Assign new weights to particles based on their errors, using a Gaussian function
    weights = np.exp(-errors / (2 * sigma**2))
    weights += 1e-12
    weights /= np.sum(weights)

    # Resample particles based on weights, higher weighted particles are selected more frequently
    idx = np.random.choice(N, N, p=weights)
    particles = particles[idx]

    # Add small random noise to particles to maintain diversity and prevent collapse
    particles[:, 0:3] += np.random.normal(0, 0.1, particles[:, 0:3].shape)
    particles[:, 3:6] += np.random.normal(0, 0.05, particles[:, 3:6].shape)
    particles[:, 3:6] /= np.linalg.norm(particles[:, 3:6], axis=1, keepdims=True)

    # Project particles back onto the constraint that position must be perpendicular to direction
    p = particles[:, 0:3]
    d = particles[:, 3:6]
    dot_pd = np.sum(p * d, axis=1, keepdims=True)
    particles[:, 0:3] = p - dot_pd * d

    return particles, mean_error

#################################################################################
# Generates a fake measurement of the magnetic field vector
#################################################################################
def generate_fake_measurement(sensor_pos, p_true, d_true):

    # make p the closest point on the wire to the origin
    p_true = p_true - np.dot(p_true, d_true) * d_true

    # vector from wire to sensor at origin
    r_true = sensor_pos - p_true

    # Compute the predicted magnetic field vector at the sensor using Biot-Savart law
    B_direction = np.cross(d_true, r_true)
    B_direction /= np.linalg.norm(B_direction) + 1e-12

    return B_direction, r_true


def main():

    # Number of particles
    N = 2000

    # Store the history of mean errors for plotting convergence
    errors_history = []

    sensor_positions = []

    particles_history = []
    sensor_history = []

    # Randomly generate a wire position and direction
    p_true = np.random.uniform(-5, 5, 3)
    d_true = np.random.uniform(-1, 1, 3)
    d_true /= np.linalg.norm(d_true)
    p_true = p_true - np.dot(p_true, d_true) * d_true

    # Initialize particles and weights
    particles = initialize_particles(N)
    weights = np.ones(N) / N

    # Run the particle filter for a number of iterations
    for i in range(100):

        # Perform sensor sweep by moving the sensor along the x-axis
        theta = (i * 0.1) % (2 * np.pi)
        if theta > np.pi:
            theta = 2 * np.pi - theta

        # Arc path in three dimensions
        #sensor_pos = np.array([np.cos(theta) * 5, np.sin(theta) * 5, np.cos(theta) * 5])

        # Linear path along all three axes
        sensor_pos = np.array([theta, theta, theta])

        sensor_positions.append(sensor_pos.copy())

        V_meas, _ = generate_fake_measurement(sensor_pos, p_true, d_true)

        # Initial plot of particles and true wire position/direction
        if i == 0:
            plot_particles(particles, p_true, d_true, sensor_positions=np.array(sensor_positions), sensor_pos=sensor_pos)

        # Add small noise to the measurement to simulate real-world conditions
        noise = np.random.normal(0, 0.05, 3)
        V_meas_noisy = V_meas + noise
        V_meas_noisy /= np.linalg.norm(V_meas_noisy)

        # Update particles based on the noisy measurement and compute mean error
        particles, mean_error = update_particles(particles, weights, V_meas_noisy, sensor_pos)
        errors_history.append(mean_error)
        
        particles_history.append(particles.copy())
        sensor_history.append(sensor_pos.copy())

        if i % 10 == 0:
            plot_particles(particles, p_true, d_true, sensor_positions=np.array(sensor_positions), sensor_pos=sensor_pos)

    # Final plot of particles after convergence, showing true wire position/direction
    #plot_particles(particles, p_true, d_true, sensor_positions=np.array(sensor_positions), sensor_pos=sensor_pos)

    #plt.figure()
    #plt.plot(errors_history)
    #plt.xlabel("Iteration")
    #plt.ylabel("Mean Error")
    #plt.title("Particle Filter Convergence")
    #plt.grid(True)
    #plt.show()

    t = np.linspace(-5, 5, 100)
    line = p_true[None, :] + t[:, None] * d_true[None, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()

        particles = particles_history[frame]
        sensor_pos = sensor_history[frame]

        x = particles[:, 0]
        y = particles[:, 1]
        z = particles[:, 2]

        dx = particles[:, 3]
        dy = particles[:, 4]
        dz = particles[:, 5]

        ax.scatter(x, y, z, s=5, alpha=0.5)
        ax.quiver(x, y, z, dx, dy, dz, length=0.2, alpha=0.3)

        # sensor
        ax.scatter(*sensor_pos, color='black', s=50)

        # mean arrow (fixed version)
        mean_pos = np.mean(particles[:, 0:3], axis=0)
        direction = mean_pos - sensor_pos
        ax.quiver(*sensor_pos, *direction, color='blue')

        ax.plot(line[:, 0], line[:, 1], line[:, 2],
            color='red', linewidth=3)
        
        ax.scatter(*p_true, color='red', s=50)

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        ax.set_box_aspect([1,1,1])

    ani = FuncAnimation(fig, update, frames=len(particles_history), interval=100)

    plt.show()

    return 0    

if __name__ == "__main__":
    main()

