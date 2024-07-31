import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


# Given g(r) with two peaks
def g_r(r):
    """Radial distribution function with two peaks."""
    return 1 + 0.5 * np.exp(-(r - 1.0) ** 2 / (2 * 0.1 ** 2)) + 0.3 * np.exp(-(r - 2.5) ** 2 / (2 * 0.1 ** 2))
    #return 1 + 0.5 * np.exp(-(r - 1.0) ** 2 / (2 * 0.1 ** 2))

def update_c_r(h_r_values, c_r_values, r_values, rho):
    """Update c(r) using Ornstein-Zernike equation."""
    h_r_new = np.zeros_like(h_r_values)

    # Solve the OZ equation iteratively
    for i, r in enumerate(r_values):
        # Integrate up to current point
        if i > 0:
            # Reverse h_r_values[:i+1] to match convolution behavior
            integrand = c_r_values[:i + 1] * h_r_values[i::-1]  # Corrected slicing
            integral = simps(integrand, r_values[:i + 1])
            h_r_new[i] = c_r_values[i] + rho * integral * 4 * np.pi * dr ** 2
        else:
            h_r_new[i] = c_r_values[i]  # At r=0, no contribution from integral

    return h_r_new


def update_U_r(g_r_values, h_r_values, c_r_values, beta=1.0):
    """Update U(r) using HNC closure."""
    return -np.log(g_r_values) / beta + h_r_values - c_r_values





if __name__ == "__main__":


    # Constants
    kT = 1.0  # Boltzmann constant times temperature (assuming kT=1 for simplicity)
    rho = 0.8  # Number density
    r_max = 5.0  # Maximum distance
    num_points = 500  # Number of discrete points

    # Radial distances
    r_values = np.linspace(0.01, r_max, num_points)
    dr = r_values[1] - r_values[0]




    g_r_values = g_r(r_values)
    h_r_values = g_r_values - 1  # Total correlation function

    # Initial guess for c(r): Direct correlation function
    c_r_values = np.zeros_like(r_values)



    # Iterative solution
    num_iterations = 100
    tolerance = 1e-5

    for iteration in range(num_iterations):
        # Update h(r)
        h_r_new = update_c_r(h_r_values, c_r_values, r_values, rho)

        # Update U(r) using HNC closure
        U_r_values = update_U_r(g_r_values, h_r_new, c_r_values)

        # Update c(r)
        c_r_values = h_r_new - h_r_values + c_r_values

        # Check for convergence
        if np.max(np.abs(h_r_new - h_r_values)) < tolerance:
            print(f'Converged after {iteration + 1} iterations.')
            break

        h_r_values = h_r_new

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
