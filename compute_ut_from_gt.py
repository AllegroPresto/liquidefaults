import numpy as np
import matplotlib.pyplot as plt











# Define the target g(r) as input
def target_g_r(r):
    """Example target g(r). Replace this with your data."""
    return 1 + np.exp(-(r - 1.0) ** 2 / 0.1)


# Initial guess for the potential using the potential of mean force
def initial_potential(r, g_r):
    """Compute initial potential U(r) from g(r)."""
    return -np.log(g_r + 1e-8) / beta


def compute_forces_from_potential(positions, U_r, r_values, box_length):
    """Compute forces on each particle from the guessed potential."""
    forces = np.zeros_like(positions)
    num_particles = positions.shape[0]

    # Compute the derivative of the potential
    dU_dr = np.gradient(U_r, r_values)  # dU/dr over entire r_values

    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            # Minimum image convention
            delta = positions[j] - positions[i]
            delta -= box_length * np.round(delta / box_length)
            r = np.linalg.norm(delta)

            if r < r_values[-1]:  # Only consider up to the max r value
                # Interpolate to find the derivative of the potential at distance r
                F_mag = -np.interp(r, r_values, dU_dr)  # Negative gradient as force
                f = F_mag * delta / r
                forces[i] += f
                forces[j] -= f

    return forces


def integrate_with_potential(positions, velocities, U_r, r_values, dt, box_length):
    """Integrate positions and velocities using the Verlet method and given potential."""
    forces = compute_forces_from_potential(positions, U_r, r_values, box_length)

    # Verlet integration
    positions += velocities * dt + 0.5 * forces * dt ** 2
    positions %= box_length  # Apply periodic boundary conditions

    new_forces = compute_forces_from_potential(positions, U_r, r_values, box_length)
    velocities += 0.5 * (forces + new_forces) * dt

    return positions, velocities, new_forces

def calculate_simulated_g_r(positions, box_length, num_bins=100, r_max=None):
    """Calculate the simulated radial distribution function g'(r)."""
    if r_max is None:
        r_max = box_length / 2

    num_particles = positions.shape[0]

    dr = r_max / num_bins
    rdf = np.zeros(num_bins)
    norm_factor = (4 / 3) * np.pi * ((np.arange(num_bins) + 1) ** 3 - np.arange(num_bins) ** 3) * dr ** 3
    norm_factor *= density * num_particles


    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            # Minimum image convention
            delta = positions[j] - positions[i]
            delta -= box_length * np.round(delta / box_length)
            r = np.linalg.norm(delta)
            if r < r_max:
                bin_index = int(r / dr)
                rdf[bin_index] += 2  # Count pairs

    rdf /= norm_factor  # Normalize by the expected density

    return np.arange(num_bins) * dr, rdf

def update_potential(U_r, g_r_simulated, g_r_target, r_values, beta=1.0):
    """Update potential using Iterative Boltzmann Inversion."""
    delta_U = -np.log(g_r_simulated / g_r_target + 1e-8) / beta  # Avoid log(0)
    return U_r + delta_U

# Analyze the potential to find its minimum and other parameters
def analyze_potential(r_values, U_r):
    """Analyze the potential U(r) to find key characteristics."""
    min_index = np.argmin(U_r)
    min_r = r_values[min_index]
    min_U = U_r[min_index]

    # Calculate well depth (difference from zero)
    well_depth = -min_U

    # Calculate effective range (r where U(r) is close to 0)
    effective_range_indices = np.where(U_r > 0.01 * min_U)[0]
    if len(effective_range_indices) > 0:
        effective_range = r_values[effective_range_indices[-1]]
    else:
        effective_range = r_values[-1]

    return min_r, min_U, well_depth, effective_range


if __name__ == "__main__":

    # Define constants
    num_bins = 100
    r_max = 3.0
    beta = 1.0  # Inverse temperature, 1/kT

    # Simulation parameters
    num_particles = 100
    density = 0.1
    box_length = (num_particles / density) ** (1 / 3)
    dt = 0.001
    #num_steps = 900
    num_steps = 300



    flag_g0 = True
    flag_r0 = True

    # Calculate g(r) for a range of r values
    r_values = np.linspace(0.1, r_max, num_bins)
    g_r_target = target_g_r(r_values)
    U_r_initial = initial_potential(r_values, g_r_target)


    if (flag_g0):
        # Plot the target g(r)
        plt.plot(r_values, g_r_target, label="Target $g(r)$")
        plt.xlabel('r')
        plt.ylabel('g(r)')
        plt.title('Target Radial Distribution Function')
        plt.grid(True)
        plt.legend()
        plt.show()

    if (flag_r0):

        # Plot the initial potential
        plt.plot(r_values, U_r_initial, label="Initial $U(r)$")
        plt.xlabel('r')
        plt.ylabel('Potential $U(r)$')
        plt.title('Initial Potential from Target $g(r)$')
        plt.grid(True)
        plt.legend()
        plt.show()


    # Initialize positions and velocities
    positions = np.random.rand(num_particles, 3) * box_length
    velocities = np.random.normal(0, np.sqrt(1.0), (num_particles, 3))
    velocities -= np.mean(velocities, axis=0)  # Zero total momentum

    # Run the simulation with the initial guessed potential
    for step in range(num_steps):
        if (step%20) ==0:
            print('step: ',step)
        positions, velocities, _ = integrate_with_potential(positions, velocities, U_r_initial, r_values, dt, box_length)

    # Calculate g'(r) after simulation
    r_simulated, g_r_simulated = calculate_simulated_g_r(positions, box_length)

    # Plot the simulated g'(r)
    plt.plot(r_values, g_r_simulated, label="Simulated $g(r)$")
    plt.plot(r_values, g_r_target, label="Target $g(r)$", linestyle='--')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title('Comparison of Simulated and Target $g(r)$')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Perform IBI iterations
    num_iterations = 20

    U_r = U_r_initial.copy()
    for iteration in range(num_iterations):
        # Simulate with current potential
        print('Iterations: ', iteration)
        for step in range(num_steps):
            positions, velocities, _ = integrate_with_potential(positions, velocities, U_r, r_values, dt, box_length)

        # Calculate the simulated g'(r)
        _, g_r_simulated = calculate_simulated_g_r(positions, box_length)

        # Update the potential
        U_r = update_potential(U_r, g_r_simulated, g_r_target, r_values)

        # Plot the results for each iteration
        plt.plot(r_values, U_r, label=f"Iteration {iteration + 1}")

    gt_save = False
    if (gt_save):
        file_out = r'ut_from_gt/ut_%s.csv' % (version_)

        g_t_df = pd.DataFrame({'Time (Years)': bin_cent_, 'Probability': hist_})
        g_t_df.to_csv(file_out, index=False)

    plt.xlabel('r')
    plt.ylabel('Potential $U(r)$')
    plt.title('Potential Updates Through IBI')
    plt.grid(True)
    plt.legend()
    plt.show()



    # Analyze the final potential
    min_r, min_U, well_depth, effective_range = analyze_potential(r_values, U_r)

    # Output the key parameters of the potential
    print(f"Position of Minimum (r_min): {min_r:.3f}")
    print(f"Minimum Potential (U_min): {min_U:.3f}")
    print(f"Potential Well Depth: {well_depth:.3f}")
    print(f"Effective Range of Potential: {effective_range:.3f}")

    # Plot the final potential with key points highlighted
    plt.plot(r_values, U_r, label="Final $U(r)$")
    plt.axvline(x=min_r, color='r', linestyle='--', label=f"Minimum at $r={min_r:.2f}$")
    plt.axhline(y=min_U, color='g', linestyle='--', label=f"Minimum $U={min_U:.2f}$")
    plt.axvline(x=effective_range, color='b', linestyle='--', label=f"Effective range $r")
    plt.show()

