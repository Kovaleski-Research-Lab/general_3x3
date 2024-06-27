# This code runs a study to determine the best resolution at which to run our meep simulation.
#
# The simulation is a 3x3 set of identical pillars with a continuous source, so we are repeating
# the LPA study conducted in the single_pillar_sim repo except explicitly setting the nearest 
# (identical) neighbors.
#
# Simulations are run for 50 femtoseconds (sim.run(until=50)), unlike the 200 fs used for the LPA
# study. This gives us different transmission values, but we are still able to settle on a resolution
# and conclude that defining a single pillar vs. defining that single pillar and its most immediate
# nearest neighbors gives the same transmission values.

# To see the analysis of this study, open resolution_study.ipynb.


# To run this code: python3 main.py -config ../../configs/config.yaml -res {resolution} -idx {idx}

import os
import sys
import yaml
import pickle
import csv
import pandas as pd
import meep as mp
import numpy as np
import helpers
import matplotlib.pyplot as plt
import random

from helpers import load_config
from update_config import update



def initial_sim(params):

    geometry = [mp.Block(size=mp.Vector3(mp.inf,mp.inf, params['pml']['thickness']
                                        + params['fusedSilica']['width']),
                         center=mp.Vector3(0,0,params['fusedSilica']['center']),
                         material=mp.Medium(index=params['fusedSilica']['n'])),

                mp.Block(size=mp.Vector3(mp.inf,mp.inf, params['amorphousSi']['height']
                                        + params['PDMS']['width'] + params['pml']['thickness']),
                         center=mp.Vector3(0,0,params['PDMS']['center']),
                         material=mp.Medium(index=params['PDMS']['n']))]

    sources = [mp.Source(mp.ContinuousSource(frequency=params['freq']),
                    component=params['source']['cmpt'],
                    center=mp.Vector3(0,0,params['source']['center']),
                    size=mp.Vector3(params['cell']['x'],params['cell']['y'],0))]

    sim = mp.Simulation(cell_size=params['cell_size'],
                    geometry=geometry,
                    sources=sources,
                    k_point=params['k_point'],
                    boundary_layers=params['pml']['layers'],
                    symmetries=params['symmetries'],
                    resolution=params['resolution'])
    
    flux_region = mp.FluxRegion(center=mp.Vector3(0,0,params['flux']['center']),
                            size=mp.Vector3(params['cell']['x'], params['cell']['y'], 0))

    
    flux_object = sim.add_flux(params['freq'], params['flux']['df'],
                               params['flux']['nfreq'], flux_region)
    
    return geometry, sources, sim, flux_region, flux_object


def get_grid(params):

    a = params['a']
    x_list = [-a, 0, a, -a, 0, a, -a, 0, a]
    y_list = [a, a, a, 0, 0, 0, -a, -a, -a]

    return x_list, y_list

def display_fields(params, sim):

    ## If you want to see the geometry without the fields, remove the second parameter
    ## of sim.plot2D

    plt.figure(figsize=(5,8))
    plot_plane = mp.Volume(center=mp.Vector3(0,0.0*params['cell_y'], 0), 
                            size=mp.Vector3(params['cell_x'], 0, params['cell_z']))
    sim.plot2D(output_plane=plot_plane, fields=params['source']['component'])
    plt.savefig('fields.png')

def write_csv(filepath, data):

    fields = data['fields']
    row = data['row']

    with open(filepath, "a") as csvfile:

        csvwriter = csv.writer(csvfile)
       
        print("Writing csv header...") 
        if os.path.getsize(filepath) == 0:
            csvwriter.writerow(fields)

        print(f"writing {row} to csv...")
        csvwriter.writerow(row) 


# take in a 3x3 group of radii and get transmission - This requires two sims - one for initial transmission (no pillars),
# one for final transmission (with pillars)
def get_transmission(params, radii):

    # This will help us position the pillars in a 3x3 grdi
    x_list, y_list = get_grid(params)

    # First we'll build the sim with substrate only (no pillars)
    geometry, sources, sim, flux_region, flux_object = initial_sim(params)

    sim.run(until=50)

    initial_flux = mp.get_fluxes(flux_object)[0]  # flux through substrate and PDMS - no pillars 
  
    # Important to reset meep before adding to the geometry and running again 
    sim.reset_meep()

    # Add in the pillars on a 3x3 grid
    for index, radius in enumerate(radii):

        geometry.append(mp.Cylinder( material = mp.Medium(index=params['amorphousSi']['n']),
                                     axis = mp.Vector3(0,0,1),
                                     radius = radius,
                                     height = params['amorphousSi']['height'],
                                     center = mp.Vector3(x_list[index], y_list[index], params['amorphousSi']['center'])
                                   )
                       ) 

    # We have to build a new mp.Simulation object with updated geometry
    sim = mp.Simulation(cell_size=params['cell_size'],
                    geometry=geometry,
                    sources=sources,
                    k_point=params['k_point'],
                    boundary_layers=params['pml']['layers'],
                    symmetries=params['symmetries'],
                    resolution=params['resolution'])

    flux_object = sim.add_flux(params['freq'], params['flux']['df'],
                               params['flux']['nfreq'], flux_region)

    sim.run(until=50)

    # We use Meep's built in functions to get transmission info
    res = sim.get_eigenmode_coefficients(flux_object, [1], eig_parity=mp.ODD_Y)
    coeffs = res.alpha
    
    flux = abs(coeffs[0,0,0]**2)
    
    # transmission is a relative value in meep, so we divide by the initial flux
    transmission = flux / initial_flux 

    return transmission

if __name__=="__main__":

    params = load_config(sys.argv)
    params = update(params)

    if params['resolution'] == None:

        err_message = "Need to pass -resolution {resolution value} as command line argument."
        raise NotImplementedError(err_message)

    # 1. We need to be able to recreate the LPA study (transmission only) with identical 3x3 pillars explicitly defined.

    filepath = os.path.join(params['paths']['root'], f"transmission-LPA.csv")
    fields = ['Resolution', 'Radius', 'Transmission']
    
    all_radii = [0.075, 0.1  , 0.125, 0.15 , 0.175, 0.2  , 0.225, 0.25]

    print(f"Getting transmission values for resolution {params['resolution']}")
    for radius in all_radii:

        radii = [radius for _ in range(9)]

        transmission = get_transmission(params, radii)
     
        row = [params['resolution'], radius, transmission]

        if mp.am_master():
            data = {'fields': fields, 'row': row}
            write_csv(filepath, data)
    
    # 2. We need to be able to do the same transmission study extending to random pillars (expect lower transmission values, but hopefully we converge at the same resolution as before.


    """
    neighbors_library = pickle.load(open(params['paths']['library'],'rb'))
    radii = list(neighbors_library[params['idx']])
    """

    
    """
    for radius in options:
        
        radii = [radius for _ in range(9)]
        
        transmission = run_study(params, radii)
        
        lines_to_write = [
            f"Radius = {radius} Transmisson = {transmission}",
        ]
        
        write_to_file("transmission.csv", lines_to_write)

        print()
        print(f"transmission = {transmission}")
        print()

    """
