import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
cutoff = 1  # Interaction range
e12 = 1     # Strength of two-body potential
"""Here we are considering a step potential 0 / e12 when outside / inside the cutoff"""

# Initialize system energy
U0 = None  

def MC_step(i):
    """Perform one Monte Carlo move for event `i`."""
    global U0, event_times  # Modify global variables

    # Store old and new times for event `i`
    ti_old = event_times[i]
    ti_new = ti_old + (2 * np.random.rand() - 1) * dt  # Random displacement

    # Ensure new time is within [0, Tmax]
    ti_new = max(0, min(ti_new, Tmax))

    # Calculate change in energy
    dU = Delta_U(i, ti_old, ti_new)

    # Metropolis-Hastings acceptance criterion
    if np.random.rand() < np.exp(-dU):
        event_times[i] = ti_new  # Accept new time
        U0 = U0 + dU  # Update system energy

def u1(t):
    """Single event external potential."""
    return 0  # Example with a constant external potential

def u2(t1, t2):
    """Interaction potential between two events."""
    t12 = abs(t2 - t1)
    return e12 if t12 < cutoff else 0

def U(event_times):
    """Compute the total potential energy."""
    Utot = 0
    for i in range(num_events):
        Utot += u1(event_times[i])
        for j in range(i + 1, num_events):
            Utot += u2(event_times[i], event_times[j])
    return Utot

def Delta_U(i, ti_old, ti_new):
    """Calculate the energy change when event `i` changes time."""
    u_old = u1(ti_old)
    u_new = u1(ti_new)

    # Interactions with other events before `i`
    for j in range(0, i):
        u_old += u2(event_times[j], ti_old)
        u_new += u2(event_times[j], ti_new)

    # Interactions with other events after `i`
    for j in range(i + 1, num_events):
        u_old += u2(event_times[j], ti_old)
        u_new += u2(event_times[j], ti_new)

    return u_new - u_old

# Initialize system energy
U0 = U(event_times)

# Debugging: Print initial setup
print("Initial event times:", event_times)
print("Initial system energy U0:", U0)

# Run multiple MC steps to simulate the system
num_iterations = 100  # Number of MC steps to perform

for step in range(num_iterations):
    # Pick a random event to move
    event_index = np.random.randint(num_events)
    MC_step(event_index)

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
