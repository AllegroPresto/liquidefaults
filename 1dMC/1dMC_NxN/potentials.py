import numpy as np

# Interaction potential parameters
e12 = 1# Strength of two-body potential
cutoff = 1	# Interaction range

def u1(t):
    """Single event external potential."""
    return 0  # Example with a constant external potential

def u2(t1, t2):
    """Interaction potential between two events."""
    t12 = abs(t2 - t1)
    return e12 if t12 < cutoff else 0  # Interaction range is cutoff

def U(event_times):
    """Compute the total potential energy."""
    U1 = 0
    U2 = 0
    num_events = len(event_times)

    for i in range(num_events):
        U1 += u1(event_times[i])
        for j in range(i + 1, num_events):  # Only check pairs once
            U2 += u2(event_times[i], event_times[j])

    return U1 + U2

def Delta_U(i, ti_old, ti_new, event_times):
    """Calculate the energy change when event `i` changes time."""
    u_old = u1(ti_old)
    u_new = u1(ti_new)

    # Interactions with all other events
    for j in range(len(event_times)):
        if i != j:  # Avoid self-interaction
            u_old += u2(event_times[j], ti_old)
            u_new += u2(event_times[j], ti_new)

    return u_new - u_old
