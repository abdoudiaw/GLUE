# requirement: mystic --> pip install mystic
import numpy as np
from mystic.tools import random_state
from zbar import zBar

"""
This module generate npts random samples between given the bounds of our parameter space.
"""

def _random_samples(lb,ub,npts=100):
  """
Inputs:
    lower bounds  --  a list of the lower bounds
    upper bounds  --  a list of the upper bounds
    npts  --  number of sample points [default = 100]
"""
  dim = len(lb)
  pts = random_state(module='numpy.random').rand(dim,npts)
  for i in range(dim):
    pts[i] = (pts[i] * abs(ub[i] - lb[i])) + lb[i]

  return pts
  
# number of samples points
npts=1100
lb=[20, 1e21, 1e21] # temperature (eV), densities (1/cm3)
ub=[500, 1e25, 1e25]

# Samples points
pts =_random_samples(lb,ub,npts)

# Calculated the ionization states for each species given a state point (densities, temperature)
data=[]
for i in pts.T:
    n=i[1:3]
    T=i[0]
    #Estimate Ar and D ionization states using THomas-Fermi
    Z= [1,18] # Nuclear charges of
    ionization= zBar(n, Z, T)
    data.append([T,n[0], n[1],ionization[0], ionization[1]])

# Write points to file for model evaluation
np.savetxt("training.txt", data)

