
# To run this code: python3 main.py -config ../../configs/config.yaml -idx {idx}

# This code provides an example of saving a simulation 
# state (sim.dump in run()) and loading it

import os
import sys
import yaml
import pickle
import meep as mp
import numpy as np

sys.path.append("../../")
from utils.general import load_config, parse_args, create_folder
from meep_utils import field_monitors, simulation, geometries, sources

root = "/develop/data/meep_studies/save_and_load_simstates"

def dump_yaml(data, name, idx):

    create_folder(os.path.join(root, name)
    filename = os.path.join(root, f"{name}/{idx}.yaml")
    with open(filename, 'w') as stream:
        
        yaml.dump(data, stream)

# build the meep simulation object
def build_sim(params):

    print(f"Loading in neighbors library, assigning index {params['idx']}") 
    neighbors_library = pickle.load(open(params['paths']['library'],'rb'))
    radii = list(neighbors_library[params['idx']])

    print("Building sim...")
    sim, dft_obj, flux_obj, params = simulation.build_sim(params, radii=radii)

    dump_yaml(params, 'params', params['idx'])
    
    return sim, flux_obj, params

# run the simulation and save the sim state 
def run_sim(params, sim):

    sim.run(until=50)

    print("dumping simulation state...")
    create_folder(os.path.join(root,'sim_states'))
    filename = os.path.join(root, f"sim_states/{params['idx']}")
    sim.dump(filename, dump_structure=True, dump_fields=True, single_parallel_file=False)

# load simulation state from previously run simulation and dump out field info
def load_sim(params, sim):

    filename = os.path.join(root, f"sim_states/{params['idx']}")

    # We are loading the sim state, so we don't have to re-run the simulation.
    sim.load(filename, load_structure=True, load_fields=True, single_parallel_file=False)
    sim.init_sim()

    return sim

#just an example of things you could pull out of the sim after re-loading it
def dump_field_data(sim):
    
    data = {}

    data['epsilon'] = sim.get_epsilon()
    data['x'] = sim.get_efield_x()
    data['y'] = sim.get_efield_y()
    data['z'] = sim.get_efield_z() 

    print("dumping field data...")    
    create_folder(os.path.join(root, 'field_info'))
    filename = os.path.join(root, f"field_info/sim_res_{res_str}.pkl") 

    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    except EOFError:
        print("Error: Reached end of file while reading")


if __name__=="__main__":

    params = load_config(sys.argv)

    params['deployment_mode'] = 0
    params['geometry']['substrate_buffer'] = False
    params['source']['type'] = 'continuous'

    # make sure the number of cores used to *run* a sim is equal to the number 
    # of cores used to *load* a sim 

    # always execute this line:
    sim, flux_obj, params = build_sim(params) 

    """
    # only execute this once and comment out call to load_sim()
    run_sim(params, sim) 
    """

    # comment out call to run_sim() if this is being executed.
    sim = load_sim(params, sim)
   
    # an example of things you could pull out of the sim after it has been reloaded. 
    dump_field_data(sim)
    
