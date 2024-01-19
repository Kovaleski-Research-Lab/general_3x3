import numpy as np
import matplotlib.pyplot as plt 
from scipy import interpolate
from scipy.interpolate import BSpline

# using three methods here: get_Bsplines() is called from radii_to_phase() and from phase_to_radii()

def get_Bsplines():

    # we're hardcoding the values we got from the LPA sim:
    radii = [0.075, 0.0875, 0.1, 0.1125, 0.125, 0.1375, 0.15, 0.1625, 0.175, 0.1875, 0.2, 0.2125, 0.225, 0.2375, 0.25]
    phase_list = [-3.00185845, -2.89738421, -2.7389328, -2.54946247, -2.26906522, -1.89738599, -1.38868364, -0.78489682, -0.05167712, 0.63232107, 1.22268106, 1.6775137, 2.04169308, 2.34964137, 2.67187105]
    
    radii = np.asarray(radii)
    phase_list = np.asarray(phase_list)
    
    # this is the BSpline interpolating function to complete the mapping from radii <-> phase.
    to_phase = interpolate.splrep(radii, phase_list, s=0, k=3)
    to_radii = interpolate.splrep(phase_list, radii, s=0, k=3)
    
    return to_phase, to_radii

# we have the ability to convert radii to phase and vice versa:
def radii_to_phase(radii):
    
    to_phase, _ = get_Bsplines() 
    
    # uncomment next line if you're starting with torch vals
    #phases = torch.from_numpy(radii)
    
    phases = interpolate.splev(radii, to_phase)
    return phases

def phase_to_radii(phases):
    
    _, to_radii = get_Bsplines()
    
    # uncomment next line if you're starting with torch vals
    #radii = torch.from_numpy(phases)
    
    radii = interpolate.splev(phases, to_radii)
    return radii