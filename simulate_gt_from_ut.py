import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d



# Apply periodic boundary conditions
def apply_periodic_boundary(positions, box_size):
    return positions % box_size

"""
# Define the radial distribution function g(r)
def g_r(r):
    #Define the radial distribution function g(r) with two peaks.
    # Example: Sum of two Gaussians
    peak1 = np.exp(-((r - 1.0)**2) / (2 * 0.1**2))
    peak2 = np.exp(-((r - 2.5)**2) / (2 * 0.1**2))
    return 1 + peak1 + peak2
"""

# Define the potential U(r) derived from g(r)
def U_r(r):
    """Potential function derived from g(r)."""
    g_r_value = g_r(r)
    # Use a threshold to prevent log of non-positive values
    g_r_value = np.clip(g_r_value, 1e-10, None)
    return -np.log(g_r_value)


# Compute pairwise distances and forces
def compute_forces(positions, box_size, r_cutoff):
    forces = np.zeros_like(positions)
    potential_energy = 0.0

    for i in range(num_particles):
        for j in range(i+1, num_particles):
            # Compute distance vector and apply periodic boundary conditions
            rij = positions[i] - positions[j]
            rij -= np.round(rij / box_size) * box_size
            r = np.linalg.norm(rij)

            if r < r_cutoff:
                # Calculate the force using the interpolated derivative of U(r)
                force_magnitude = -dU_dr_interp(r)

                # Update forces
                forces[i] += force_magnitude * rij / r
                forces[j] -= force_magnitude * rij / r

                # Compute potential energy
                potential_energy += U_r(r)

    return forces, potential_energy

# Integrate equations of motion (Verlet algorithm)
def integrate(positions, velocities, forces, dt):
    velocities += 0.5 * forces * dt
    positions += velocities * dt
    forces, potential_energy = compute_forces(positions, box_size, r_cutoff)
    velocities += 0.5 * forces * dt
    return positions, velocities, forces, potential_energy

# Compute radial distribution function g(r)
def compute_gr(positions, box_size, num_bins=100, r_max=None):
    if r_max is None:
        r_max = box_size / 2
    dr = r_max / num_bins
    g_r = np.zeros(num_bins)
    bin_edges = np.linspace(0, r_max, num_bins + 1)
    r_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    for i in range(num_particles):
        for j in range(i+1, num_particles):
            rij = positions[i] - positions[j]
            rij -= np.round(rij / box_size) * box_size
            r = np.linalg.norm(rij)
            if r < r_max:
                bin_index = int(r / dr)
                g_r[bin_index] += 2  # Each pair contributes twice

    # Normalize g(r)
    volume = (4/3) * np.pi * ((bin_edges[1:])**3 - (bin_edges[:-1])**3)
    number_density = num_particles / box_size**3
    g_r /= (number_density * volume * num_particles)

    return r_mid, g_r


# Given g(r) with two peaks
def g_r(r):
    """Radial distribution function with two peaks."""
    return 1 + 0.5 * np.exp(-(r - 1.0) ** 2 / (2 * 0.1 ** 2)) + 0.3 * np.exp(-(r - 2.5) ** 2 / (2 * 0.1 ** 2))



if __name__ == "__main__":

    # Simulation parameters
    num_particles = 1000   # Number of particles
    rho = 0.1             # Density
    box_size = (num_particles / rho) ** (1/3)  # Box size
    dt = 0.001            # Time step
    #num_steps = 3000      # Number of simulation steps
    num_steps = 500      # Number of simulation steps
    r_cutoff = 3.0        # Cut-off radius for potential calculation

    # Initialize particle positions randomly in the box
    positions = np.random.rand(num_particles, 3) * box_size

    # Initialize particle velocities randomly
    velocities = np.random.randn(num_particles, 3)


    # Pre-compute the derivative of U(r) over a range of r values
    r_values = np.linspace(0.01, r_cutoff, 500)  # Avoid r=0 to prevent division by zero
    U_r_values = U_r(r_values)
    g_r_values = g_r(r_values)
    dU_dr_values = np.gradient(U_r_values, r_values)


    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(r_values, g_r_values, label='g(r)', color='b')
    plt.xlabel('Distance r')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function with Two Peaks')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(r_values, U_r_values, label='U(r)', color='r')
    plt.xlabel('Distance r')
    plt.ylabel('Potential U(r)')
    plt.title('Intermolecular Potential U(r) from g(r)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



    # Interpolation function for dU/dr
    dU_dr_interp = interp1d(r_values, dU_dr_values, kind='linear', fill_value='extrapolate')


    # Simulation loop
    positions = apply_periodic_boundary(positions, box_size)
    forces, _ = compute_forces(positions, box_size, r_cutoff)
    potential_energies = []
    kinetic_energies = []

    for step in range(num_steps):
        positions, velocities, forces, potential_energy = integrate(positions, velocities, forces, dt)
        positions = apply_periodic_boundary(positions, box_size)

        # Calculate kinetic energy
        kinetic_energy = 0.5 * np.sum(velocities**2)
        potential_energies.append(potential_energy)
        kinetic_energies.append(kinetic_energy)

        if step % 100 == 0:
            print(f"Step {step}, Potential Energy: {potential_energy}, Kinetic Energy: {kinetic_energy}")

    # Compute g(r) after the simulation
    r_mid, g_r_simulated = compute_gr(positions, box_size)

    # Plot results
    plt.figure(figsize=(12, 5))

    # Plot energies
    plt.subplot(1, 2, 1)
    plt.plot(potential_energies, label='Potential Energy', color='r')
    plt.plot(kinetic_energies, label='Kinetic Energy', color='b')
    plt.xlabel('Time Step')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('Energy vs. Time')

    # Plot g(r)
    plt.subplot(1, 2, 2)
    plt.plot(r_mid, g_r_simulated, label='Simulated g(r)', color='g')
    plt.xlabel('Distance r')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
