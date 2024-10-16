import numpy as np
import matplotlib.pyplot as plt
import time
import sys


def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()

"""
def double_well_potential(x, a=1.0, b=2.0, c=1.0):  # c shifts the potential upwards
    return a * (x**2 - b)**2 + c

def double_well_force(x, a=1.0, b=2.0, c=1.0):
    return -4 * a * x * (x**2 - b)  # c doesn't affect the force
"""

import numpy as np



def double_well_potential(r, epsilon = 1.5, sigma1 = 1.0, A=1.5, r0 = 2.5, w=0.2):
    lj_part = 4 * epsilon * ((sigma1 / r)**12 - (sigma1 / r)**6)
    gaussian_part = A * np.exp(-((r - r0) / w)**2)
    return lj_part - gaussian_part

# Define the force function
def double_well_force(r,epsilon = 1.0, sigma1 = 1.0, A=0.5, r0 = 1.5, w=0.2):
    lj_force = 24 * epsilon * ((2 * sigma1**12 / r**13) - (sigma1**6 / r**7))
    gaussian_force = -A * np.exp(-((r - r0) / w)**2) * (2 * (r - r0) / w**2)
    return lj_force - gaussian_force


"""
def double_well_morse(r, D_e, alpha, r_e, A, beta, r_0):
  
  Calculates the double well Morse potential.

  Args:
    r: Distance between particles.
    D_e: Dissociation energy.
    alpha: Width parameter.
    r_e: Equilibrium bond length.
    A: Amplitude of the perturbation.
    beta: Width of the perturbation.
    r_0: Center of the perturbation.

  Returns:
    Potential energy.
  
  return D_e * (1 - np.exp(-alpha * (r - r_e)))**2 + A * np.exp(-beta * (r - r_0)**2)
"""

"""
def double_well_potential(r, A, r1, r2, B):
  
  Calculates the potential energy between two particles.

  Args:
    r: Distance between particles.
    A: Amplitude of the double well.
    r1, r2: Positions of the minima.
    B: Strength of the repulsive core.

  Returns:
    Potential energy.
  
  return A * ((r - r1)**2 * (r - r2)**2) + B / (r**12)
"""

"""
def double_well_force(r, A, r1, r2, B):
  
  Calculates the force between two particles.

  Args:
    r: Distance between particles.
    A: Amplitude of the double well.
    r1, r2: Positions of the minima.
    B: Strength of the repulsive core.

  Returns:
    Force between particles.
  
  return -4 * A * r * (r - r1) * (r - r2) - 12 * B / (r**13)
"""

def calculate_forces(positions, epsilon = 1.0, sigma1 = 1.0, A=0.5, r0 = 1.5, w=0.2):
    # Assuming a one-dimensional system for simplicity
    N = len(positions)
    forces = np.zeros_like(positions)
    for i in range(N):
        forces[i] = double_well_force(positions[i],epsilon, sigma1, A, r0, w)
    return forces

"""
def lennard_jones_potential(r):
    sigma = 1.0
    epsilon = 1.0
    r6 = r**6
    r12 = r6 * r6
    return 4 * epsilon * (r12 - r6)
"""
"""
def lennard_jones_force(r):
    sigma = 1.0
    epsilon = 1.0
    r6 = r**6
    r12 = r6 * r6
    return 48 * epsilon * (r12 - 0.5 * r6) / r**7
"""

"""
def calculate_forces(positions):
    N = len(positions)
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(i+1, N):
            rij = positions[j] - positions[i]
            r = np.linalg.norm(rij)
            if r < 2.5:  # Cutoff distance
                force = lennard_jones_force(r)
                forces[i] += force * rij / r
                forces[j] -= force * rij / r
    return forces
"""

def integrate_verlet(positions, velocities, forces, dt, epsilon, sigma1, A, r0, w):
    N = len(positions)
    accelerations = np.zeros_like(forces)
    for i in range(N):
        accelerations[i] = forces[i] / np.sum(np.square(positions[i]), axis=0)
    new_positions = positions + velocities * dt + 0.5 * accelerations * dt**2
    new_velocities = velocities + 0.5 * (accelerations + calculate_forces(new_positions, epsilon, sigma1, A, r0, w) / np.array([np.sum(np.square(new_positions), axis=1)] * 3).T) * dt
    return new_positions, new_velocities


def calculate_gr(positions, box_size, bins):
    N = len(positions)
    hist, bin_edges = np.histogram([(np.linalg.norm(p1 - p2) % box_size) for p1 in positions for p2 in positions if p1 is not p2], bins=bins, range=(0, box_size/2))
    rho = N / box_size**3
    dr = bin_edges[1] - bin_edges[0]
    gr = hist / (4 * np.pi * rho * N * dr * bin_edges[:-1]**2)
    return bin_edges[:-1], gr

def md_simulation(num_particles, box_size, dt, steps, epsilon, sigma1, A, r0, w):
    positions = np.random.rand(num_particles, 3) * box_size
    velocities = np.random.randn(num_particles, 3)

    t0 = time.time()
    for step in range(steps):
        if (step%200 == 0):
            t1 = time.time()
            #print('T0:',  t0)
            #print('T1:',  t1)

            print('N. Steps: %s, delta time: %.2f'%(str(step), (t1 - t0)))
            t0 = t1

        forces = calculate_forces(positions, epsilon, sigma1, A, r0, w)
        positions, velocities = integrate_verlet(positions, velocities, forces, dt, epsilon, sigma1, A, r0, w)

    bins = np.linspace(0, box_size/2, 100)
    r, gr = calculate_gr(positions, box_size, bins)

    dr = r[1] - r[0]
    area = 0.0
    ln = len(gr)
    for i in range(1, ln):
        area = area + gr[i]*dr

    gr = gr[1:]/area
    #print('area: ', area)
    #FQ(77)
    # Calculate U(r) for plotting
    r_values = np.linspace(0.1, box_size/2, 500)
    u_values = double_well_potential(r_values)


    # Plot U(r) and g(r)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(r_values, u_values)
    plt.xlabel('r')
    plt.ylabel('U(r)')
    plt.title('Potential Energy')
    plt.ylim([-5,5])

    plt.subplot(1, 2, 2)
    plt.plot(r[1:], gr, '-o')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.ylim([-0.5, 0.5])
    plt.title('Radial Distribution Function')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_particles = 1000
    box_size = 10.0
    dt = 0.01
    steps = 5000

    #epsilon = 1.0
    #sigma1 = 1.0
    #A = 0.5
    #r0 = 1.5
    #w = 0.2

    epsilon = 5.5
    sigma1 = 1.2
    A = 5.5
    r0 = 3.0
    w = 0.2


    x_values = np.linspace(0.5, 10, 300)


    y_values = double_well_potential(x_values, epsilon, sigma1, A, r0, w)
    plt.plot(x_values, y_values)
    plt.xlim([0,5])
    plt.ylim([-10,10])
    plt.xlabel('x')
    plt.ylabel('Potential Energy')
    plt.show()


    md_simulation(num_particles, box_size, dt, steps, epsilon, sigma1, A, r0, w)
