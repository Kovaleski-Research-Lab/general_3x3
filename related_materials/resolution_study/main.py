# To run this code: python3 main.py -config ../../configs/config.yaml -res {resolution} -idx {idx}

# This code provides an example of saving a simulation state (sim.dump in run()) and loading it
# back in (sim.load() in organize_data()) This isn't necessary for this particular task, we just 
# used this as an opportunity to show this functionality.

import os
import sys
import yaml
import pickle
import meep as mp
import numpy as np

sys.path.append("../../")
from utils.general import load_config, parse_args
from meep_utils import field_monitors, simulation

root = "/develop/data/meep_studies/resolution_study"

# build the meep simulation object
def build_sim(params):
   
    print(f"Loading in neighbors library, assigning index {params['idx']}") 
    neighbors_library = pickle.load(open(params['paths']['library'],'rb'))
    radii = list(neighbors_library[params['idx']])

    print(f"Building sim for resolution {params['simulation']['resolution']}")
    sim, dft_obj, flux_obj, params = simulation.build_sim(params, radii=radii)

    res = params['simulation']['resolution']
    res_str = str(res).zfill(3)

    updated_params = { 'cell_x' : params['cell_x'],
                       'cell_y' : params['cell_y'],
                       'cell_z' : params['cell_z'],
                       'thickness_pml' : params['geometry']['thickness_pml'],
                       'size_z_fused_silica' : params['geometry']['size_z_fused_silica'],
                       'height_pillar' : params['geometry']['height_pillar'],
                       'wavelength' : params['source']['wavelength'],
                       'resolution' : params['simulation']['resolution'],
                      }
    
    # We want to dump out the updated params after the sim object is built.
    filename = os.path.join(root, f"params/params_{res_str}.yaml") 
    with open(filename, 'w') as stream:
        yaml.dump(updated_params, stream)

    return sim, params

# run the simulation and save the sim state 
def run_sim(params, sim):

    sim.run(until=50)

    res = params['simulation']['resolution']
    res_str = str(res).zfill(3)

    print("dumping simulation state...")
    filename = os.path.join(root, f"sim_states/sim_res_{res_str}")
    sim.dump(filename, dump_structure=True, dump_fields=True, single_parallel_file=False)

# load simulation state from previously run simulation and dump out field info
def load_sim(params, sim):

    res = params['simulation']['resolution']
    res_str = str(res).zfill(3)
    filename = os.path.join(root, f"sim_states/sim_res_{res_str}")

    # We are loading the sim state, so we don't have to re-run the simulation.
    sim.load(filename, load_structure=True, load_fields=True, single_parallel_file=False)
    sim.init_sim()
    
    # Now we'll gather the information we want and save it to a pickle file
    data = {}

    data['epsilon'] = sim.get_epsilon()
    data['x'] = sim.get_efield_x()
    data['y'] = sim.get_efield_y()
    data['z'] = sim.get_efield_z() 

    print("dumping field data...")    
    filename = os.path.join(root, f"field_info/sim_res_{res_str}.pkl") 

    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    except EOFError:
        print("Error: Reached end of file while reading")

        
               

if __name__=="__main__":

    params = load_config(sys.argv)
    res = params['res']
    params['simulation']['resolution'] = res

    if res == None:

        err_message = "Need to pass -res {resolution value} as command line argument."
        raise NotImplementedError(err_message)

    params['deployment_mode'] = 0
    params['geometry']['substrate_buffer'] = False
    params['source']['type'] = 'continuous'
  
    # make sure the number of cores you use to *run* a sim is equal to the number of cores
    # you use to *load* a sim 
    sim, params = build_sim(params) 
    #run_sim(params, sim) 
    load_sim(params, sim)
    
    
