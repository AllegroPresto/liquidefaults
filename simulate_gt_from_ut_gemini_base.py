import numpy as np
import matplotlib.pyplot as plt
import time

def lennard_jones_potential(r):
    sigma = 1.0
    epsilon = 1.0
    r6 = r**6
    r12 = r6 * r6
    return 4 * epsilon * (r12 - r6)

def lennard_jones_force(r):
    sigma = 1.0
    epsilon = 1.0
    r6 = r**6
    r12 = r6 * r6
    return 48 * epsilon * (r12 - 0.5 * r6) / r**7

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

def integrate_verlet(positions, velocities, forces, dt):
    N = len(positions)
    accelerations = np.zeros_like(forces)
    for i in range(N):
        accelerations[i] = forces[i] / np.sum(np.square(positions[i]), axis=0)
    new_positions = positions + velocities * dt + 0.5 * accelerations * dt**2
    new_velocities = velocities + 0.5 * (accelerations + calculate_forces(new_positions) / np.array([np.sum(np.square(new_positions), axis=1)] * 3).T) * dt
    return new_positions, new_velocities


def calculate_gr(positions, box_size, bins):
    N = len(positions)
    hist, bin_edges = np.histogram([(np.linalg.norm(p1 - p2) % box_size) for p1 in positions for p2 in positions if p1 is not p2], bins=bins, range=(0, box_size/2))
    rho = N / box_size**3
    dr = bin_edges[1] - bin_edges[0]
    gr = hist / (4 * np.pi * rho * N * dr * bin_edges[:-1]**2)
    return bin_edges[:-1], gr

def md_simulation(num_particles, box_size, dt, steps):
    positions = np.random.rand(num_particles, 3) * box_size
    velocities = np.random.randn(num_particles, 3)

    t0 = time.time()
    for step in range(steps):
        if (step%200 == 0):
            t1 = time.time()
            #print('T0:',  t0)
            #print('T1:',  t1)

            print('N. Steps: %s, time: %.2f'%(str(step), (t1 - t0)))
            t0 = t1

        forces = calculate_forces(positions)
        positions, velocities = integrate_verlet(positions, velocities, forces, dt)

    bins = np.linspace(0, box_size/2, 100)
    r, gr = calculate_gr(positions, box_size, bins)

    # Calculate U(r) for plotting
    r_values = np.linspace(0.1, box_size/2, 100)
    u_values = lennard_jones_potential(r_values)

    # Plot U(r) and g(r)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(r_values, u_values)
    plt.xlabel('r')
    plt.ylabel('U(r)')
    plt.title('Potential Energy')

    plt.subplot(1, 2, 2)
    plt.plot(r, gr)
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_particles = 100
    box_size = 10.0
    dt = 0.01
    steps = 5000

    md_simulation(num_particles, box_size, dt, steps)
