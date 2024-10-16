import numpy as np
import matplotlib.pyplot as plt
from potentials import u1, u2, U, Delta_U

# Set seed for reproducibility
np.random.seed(42)

# Global variables (initialization)
num_events = 10  # Number of events
Tmax = 5   # Maximum time for events (in months)

# Event density
rho = num_events / Tmax  

# Initialize random event times
event_times = np.random.rand(num_events) * Tmax  

# MC displacement step size
dt = Tmax / (2 * num_events)  

# Interaction potential parameters
e12 = 1     # Strength of two-body potential

# Initialize system energy
U0 = U(event_times)

def MC_step(i, event_times):
    """Perform one Monte Carlo move for event `i`."""
    global U0  # Declare U0 as global to modify it

    # Store old and new times for event `i`
    ti_old = event_times[i]
    ti_new = ti_old + (2 * np.random.rand() - 1) * dt  # Random displacement

    # Ensure new time is within [0, Tmax]
    ti_new = max(0, min(ti_new, Tmax))

    # Calculate change in energy using all events
    dU = Delta_U(i, ti_old, ti_new, event_times)

    # Metropolis-Hastings acceptance criterion
    if np.random.rand() < np.exp(-dU):
        event_times[i] = ti_new  # Accept new time
        U0 += dU  # Update the system energy

def run_simulation(num_iterations):
    """Run the Monte Carlo simulation for a given number of iterations."""
    # Initialize system energy
    U0 = U(event_times)

    # Debugging: Print initial setup
    print("Initial event times:", event_times)
    print("Initial system energy U0:", U0)

    for step in range(num_iterations):
        # Pick a random event to move
        event_index = np.random.randint(num_events)
        MC_step(event_index, event_times)

    # Print final state
    print("Updated event times:", event_times)
    print("Updated system energy U0:", U0)

    # Verify energy drift by recomputing the total energy
    U_final = U(event_times)
    print("Recomputed system energy U_final:", U_final)

    if np.isclose(U0, U_final):
        print("Energy check passed: No drift detected.")
    else:
        print("Energy drift detected! Check your MC moves.")

if __name__ == "__main__":
    num_iterations = 100  # Number of MC steps to perform
    run_simulation(num_iterations)

    # Plotting the event times
    plt.figure(figsize=(10, 5))
    plt.plot(event_times, 'o-', label='Event Times After MC Simulation', color='blue')
    plt.axhline(y=0, color='grey', linestyle='--')
    plt.title('Event Times After Monte Carlo Simulation')
    plt.xlabel('Event Index')
    plt.ylabel('Event Time')
    plt.legend()
    plt.grid()
    plt.show()
