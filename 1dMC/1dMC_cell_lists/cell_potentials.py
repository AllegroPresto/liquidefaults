import numpy as np
from cell_list import find_nearby_events  # Import the function directly

# Interaction potential parameters
e12 = 1  # Strength of two-body potential

def u1(t):
    """Single event external potential."""
    return 0  # Example with a constant external potential

def u2(t1, t2, cutoff):
    """Interaction potential between two events."""
    t12 = abs(t2 - t1)
    return e12 if t12 < cutoff else 0  # Interaction range is 1

def U(event_times, cell_list, cutoff):
    """Compute the total potential energy."""
    U1 = 0
    U2 = 0
    for i in range(len(event_times)):
        U1 += u1(event_times[i])
        for j in find_nearby_events(i, event_times, cell_list, cutoff):
            if i != j:  # Avoid self-interaction
                U2 += u2(event_times[i], event_times[j], cutoff)
    return U1+U2/2

def Delta_U(i, ti_old, ti_new, event_times, cell_list, cutoff):
    """Calculate the energy change when event `i` changes time."""
    u_old = u1(ti_old)
    u_new = u1(ti_new)

    # Interactions with nearby events
    for j in find_nearby_events(i, event_times, cell_list, cutoff):
        if i != j:  # Avoid self-interaction
            u_old += u2(event_times[j], ti_old, cutoff)
            u_new += u2(event_times[j], ti_new, cutoff)

    return u_new - u_old
