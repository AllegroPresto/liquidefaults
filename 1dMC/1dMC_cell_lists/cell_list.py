import numpy as np

def initialize_cell_list(event_times, cell_list, cutoff):
    """Initialize the cell list with event times."""
    cell_size = cutoff / 2  # Ensure cell size is defined
    for i, t in enumerate(event_times):
        cell_index = int(t // cell_size)
        if cell_index < len(cell_list):
            cell_list[cell_index].append(i)

def update_cell_list(i, old_time, new_time, cell_list, cutoff):
    """Update cell list after moving event i."""
    cell_size = cutoff / 2  # Ensure cell size is defined
    old_cell_index = int(old_time // cell_size)
    new_cell_index = int(new_time // cell_size)

    # Remove event from old cell
    if old_cell_index < len(cell_list):
        cell_list[old_cell_index].remove(i)
    
    # Add event to new cell
    if new_cell_index < len(cell_list):
        cell_list[new_cell_index].append(i)

def find_nearby_events(i, event_times, cell_list, cutoff):
    """Find events in the same and adjacent cells."""
    nearby_events = []
    cell_index = int(event_times[i] // (cutoff / 2))

    # Check current and neighboring cells
    for delta in [-1, 0, 1]:
        neighbor_index = cell_index + delta
        if 0 <= neighbor_index < len(cell_list):
            nearby_events.extend(cell_list[neighbor_index])
    
    return nearby_events
