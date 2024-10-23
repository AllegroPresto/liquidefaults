import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq


def potential_step(r, epsilon, r_c):
    """Step potential with positive value within cutoff."""
    return epsilon if r <= r_c else 0.0


def direct_correlation_function(r, u_r, beta, cutoff):
    """Calculate the direct correlation function using Percus-Yevick closure."""
    return -beta * u_r if r <= cutoff else 0.0


def solve_oz(c_r, rho, dr, r_max):
    """Solve Ornstein-Zernike equation numerically in Fourier space."""
    r = np.arange(0, r_max, dr)

    # Compute Fourier transform of c(r)
    c_k = fft(c_r)
    k = fftfreq(len(c_r), d=dr)

    # Prevent division by zero or invalid values in denominator
    denominator = 1 - rho * c_k
    epsilon_tolerance = 1e-10

    # Identify problematic values in the denominator and replace them with epsilon tolerance
    problematic_indices = np.abs(denominator) < epsilon_tolerance
    denominator[problematic_indices] = epsilon_tolerance

    # Compute h(k) = c(k) / (1 - rho * c(k))
    h_k = c_k / denominator

    # Inverse Fourier transform to get h(r)
    h_r = np.real(ifft(h_k))

    # Calculate g(r) from h(r)
    g_r = 1 + h_r

    # Replace any NaN or negative g(r) values with 0 as they are not physically meaningful
    g_r = np.where(np.isnan(g_r) | (g_r < 0), 0, g_r)

    return r, g_r


def main():
    # Parameters
    rho = 0.8  # Number density
    r_max = 10.0  # Maximum r value to consider
    dr = 0.1  # Step size in r
    epsilon = 1.0  # Step potential height
    r_c = 2.0  # Cutoff radius for the step potential
    beta = 1.0  # Inverse temperature (1/kT)

    # Create r array
    r = np.arange(0, r_max, dr)

    # Define the step potential and direct correlation function
    u_r = np.array([potential_step(ri, epsilon, r_c) for ri in r])
    c_r = np.array([direct_correlation_function(ri, u_r[i], beta, r_c) for i, ri in enumerate(r)])

    # Debugging: Ensure c_r does not contain NaN or excessively large values
    if np.any(np.isnan(c_r)):
        raise ValueError("c(r) contains NaN values. Check the parameters or potential function.")
    if np.any(np.abs(c_r) > 1e6):
        raise ValueError(
            "c(r) contains excessively large values. This may indicate an issue with the potential or density.")

    # Solve OZ equation numerically
    r, g_r = solve_oz(c_r, rho, dr, r_max)

    # Plot the results
    plt.plot(r, g_r, label='g(r) from OZ Equation (Positive Step Potential)')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function from Ornstein-Zernike Equation (Positive Step Potential)')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
