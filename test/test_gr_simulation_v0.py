import numpy as np
import matplotlib.pyplot as plt
import time


import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """Calculate the Lennard-Jones potential for a distance r."""
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def initialize_positions(N, L):
    """Initialize N particles in a cubic box of side L with random positions."""
    return np.random.rand(N, 3) * L

def compute_forces(positions, L, epsilon=1.0, sigma=1.0):
    """Compute the forces on each particle using the Lennard-Jones potential."""
    N = len(positions)
    forces = np.zeros((N, 3))
    for i in range(N):
        for j in range(i + 1, N):
            rij = positions[j] - positions[i]
            rij -= L * np.round(rij / L)  # Apply periodic boundary conditions
            r = np.linalg.norm(rij)
            if r < L / 2.0:  # Cutoff distance for Lennard-Jones potential
                F = 24 * epsilon * (2 * (sigma / r)**12 - (sigma / r)**6) / r**2
                force_ij = F * rij
                forces[i] += force_ij
                forces[j] -= force_ij
    return forces

def integrate(positions, velocities, forces, dt, L):
    """Integrate the equations of motion using the velocity-Verlet algorithm."""
    positions += velocities * dt + 0.5 * forces * dt**2
    positions = positions % L  # Apply periodic boundary conditions
    velocities += 0.5 * forces * dt
    new_forces = compute_forces(positions, L)
    velocities += 0.5 * new_forces * dt
    return positions, velocities, new_forces

def compute_g_r(positions, L, dr, r_max):
    """Compute the radial distribution function g(r)."""
    N = len(positions)
    g_r = np.zeros(int(r_max / dr))
    distances = np.zeros(N * (N - 1) // 2)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            rij = positions[j] - positions[i]
            rij -= L * np.round(rij / L)  # Apply periodic boundary conditions
            r = np.linalg.norm(rij)
            if r < r_max:
                distances[idx] = r
                idx += 1
    distances = distances[:idx]
    for r in distances:
        bin_index = int(r / dr)
        g_r[bin_index] += 2
    norm = (4 / 3) * np.pi * ((np.arange(1, len(g_r) + 1) * dr)**3 - (np.arange(len(g_r)) * dr)**3)
    g_r /= (N * (N - 1) / 2) * norm / L**3
    return g_r

if __name__ == '__main__':

    # Parameters
    N = 100           # Number of particles
    L = 10.0          # Size of the cubic box
    T = 1.0           # Temperature (not used here but required for thermostat)
    dt = 0.001        # Time step
    steps = 100      # Number of simulation steps
    epsilon = 1.0     # Lennard-Jones potential parameter
    sigma = 1.0       # Lennard-Jones potential parameter
    dr = 0.1          # Bin width for g(r)
    r_max = L / 2.0   # Maximum distance for g(r)

    # 100-5,







    # Initialization
    start_time = time.time()
    positions = initialize_positions(N, L)

    velocities = np.random.randn(N, 3)
    forces = compute_forces(positions, L)
    init_time = time.time()
    print(f"Initialization time: {init_time - start_time:.2f} seconds")

    # Plot initial positions
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])
    #ax.set_title('Initial Positions of Particles')
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    #plt.show()

    # Compute g(r)
    g_r0 = compute_g_r(positions, L, dr, r_max)

    # Plot g(r)
    r = np.arange(0, r_max, dr)
    plt.plot(r, g_r0)
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function at t=0')
    plt.show()



    # Molecular Dynamics Simulation
    for step in range(steps):
        positions, velocities, forces = integrate(positions, velocities, forces, dt, L)
    sim_time = time.time()
    print(f"Simulation time: {sim_time - init_time:.2f} seconds")

    # Plot final positions
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])
    ##ax.set_title('Final Positions of Particles')
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    #plt.show()

    # Compute g(r)
    g_r = compute_g_r(positions, L, dr, r_max)
    compute_time = time.time()
    print(f"g(r) computation time: {compute_time - sim_time:.2f} seconds")

    total_time = time.time()
    print(f"Total elapsed time: {total_time - start_time:.2f} seconds")

    # Plot g(r)
    r = np.arange(0, r_max, dr)
    plt.plot(r, g_r)
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function at t = %s'%(steps))
    plt.show()


    r = np.arange(0, r_max, dr)
    plt.plot(r, g_r)
    plt.plot(r, g_r0, '-r')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function at t = %s'%(steps))
    plt.legend(['Final g(r)', 'Starting g(r)'])

    plt.show()



