# Monte Carlo simulations of (correlated) event times 

- **1dMC_NxN** contains test codes with energy calculations that scale as NxN (explicit loops over all possible pairs) 
- **1dMC_cell** contain test codes with energy calculations that scale linear in N (using cell lists structures, scales as N * rho * cutoff where rho is the density of events and cutoff is the interaction range)

all simulations are for a square well of range(cutoff) 1 and depth 1