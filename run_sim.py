import sys 
import logging
import time
import os
import shutil
from IPython import embed
import yaml
import meep as mp
from mpi4py import MPI
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

import _3x3Pillars
sys.path.append("../")
from utils import parameter_manager, mapping
from core import preprocess_data

def create_folder(path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("creating folder")
    if rank == 0:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"created folder {path}")
            time.sleep(2)
        else:
            pass

    comm.Barrier()
    if rank == 0:
        time.sleep(2)

def dump_geometry_image(model, pm):
    plt.figure()
    plot_plane = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(pm.cell_x, 0, pm.cell_z))    
    model.sim.plot2D(output_plane = plot_plane)
    plt.savefig("geometry.png")

def dump_data(neighbor_index, data, pm): # this is called when we're generating data
    folder_path_sims = pm.path_dataset
    sim_name = "%s.pkl" % (str(neighbor_index).zfill(6))
    filename_sim = os.path.join(folder_path_sims, sim_name)
    print(f"dumping data to {filename_sim}")
    with open(filename_sim, "wb") as f:
        pickle.dump(data,f)
   
    # Make sure pickle is written  

    time.sleep(20)
    
    print("\nEverything done\n")

def run(radii_list, index, pm, dataset=None):
    a = pm.lattice_size
    # Initialize model #
    model = _3x3Pillars._3x3PillarSim()
    
    # Build geometry for initial conditions (no pillar) #
    model.build_geometry(pm.geometry_params)
    
    pm.geometry = [model.fusedSilica_block, model.PDMS_block]
   
    # should make this general, so it is dependent on grid size (currently hardcoded for 3x3) 
    x_list = [-a, 0, a, -a, 0, a, -a, 0, a]
    y_list = [a, a, a, 0, 0, 0, -a, -a, -a]
    for i, neighbor in enumerate(radii_list):
        pm.radius = neighbor
        pm.x_dim = x_list[i]
        pm.y_dim = y_list[i]
        model.build_geometry(pm.geometry_params)
        pm.geometry.append(model.pillar)
    # Build Source object #
    model.build_source(pm.source_params)
     
    # Build Simulation object # 
    pm.source = model.source
    model.build_sim(pm.sim_params)
    
    # Build DFT monitor and populate field info #
    model.build_dft_mon(pm.dft_params)  
    start_time = time.time()
    model.run_sim(pm.sim_params, )
    elapsed_time = time.time() - start_time
    elapsed_time = round(elapsed_time / 60,2)
    
    model.collect_field_info()
    
    data = {}

    data["near_fields_1550"] = {}
    data["near_fields_1550"]["ex"] = model.dft_field_ex_1550
    data["near_fields_1550"]["ey"] = model.dft_field_ey_1550
    data["near_fields_1550"]["ez"] = model.dft_field_ez_1550
    
    data["near_fields_1060"] = {}
    data["near_fields_1060"]["ex"] = model.dft_field_ex_1060
    data["near_fields_1060"]["ey"] = model.dft_field_ey_1060
    data["near_fields_1060"]["ez"] = model.dft_field_ez_1060

    data["near_fields_1300"] = {}
    data["near_fields_1300"]["ex"] = model.dft_field_ex_1300
    data["near_fields_1300"]["ey"] = model.dft_field_ey_1300
    data["near_fields_1300"]["ez"] = model.dft_field_ez_1300

    data["near_fields_1650"] = {}
    data["near_fields_1650"]["ex"] = model.dft_field_ex_1650
    data["near_fields_1650"]["ey"] = model.dft_field_ey_1650
    data["near_fields_1650"]["ez"] = model.dft_field_ez_1650

    data["near_fields_2881"] = {}
    data["near_fields_2881"]["ex"] = model.dft_field_ex_2881
    data["near_fields_2881"]["ey"] = model.dft_field_ey_2881
    data["near_fields_2881"]["ez"] = model.dft_field_ez_2881

    data["eps_data"] = model.eps_data
    data["sim_time"] = elapsed_time
    data["radii"] = radii_list
    
    if(pm.resim == 0):
        dump_data(index, data, pm) 
    elif(pm.resim == 1):
        eval_name = f"sample_{index}_preprocessed.pkl"
        path_results = "/develop/results/spie_journal_2023/resim_results"
        path_resim = os.path.join(path_results, pm.exp_name + pm.training_stage, dataset) 
        create_folder(path_resim)

        # let's preprocess the data so we don't have such a big file to transfer
        preprocessed_data = preprocess_data.preprocess(pm, data)
        filename = os.path.join(path_resim, eval_name) 
        f = open(filename, "wb")
        pickle.dump(preprocessed_data, f) 

def charlies_test(pm, index):
    path_results = "/develop/results/spie_journal_2023/resim_params/charlie_test"
    path_resims = os.path.join(path_results, "results.pkl")

    model_results = pickle.load(open(os.path.join(path_resims), "rb"))
    
    preds = model_results['preds'][0]
    labels = model_results['labels'][0]

    single_pred = preds[index]
    single_truth = labels[index]

    radii_list = mapping.phase_to_radii(single_pred)
    radii_list = np.round(radii_list, 6)
    radii_list = list(radii_list)

    embed();exit()
    run(radii_list, index, pm, dataset="train")
    print(f"done with charlie's test, index {index}")
    # let's do another one:
    single_pred = preds[index+1]
    single_truth = labels[index+1]

    radii_list = mapping.phase_to_radii(single_pred)
    radii_list = np.round(radii_list, 6)
    radii_list = list(radii_list)
    run(radii_list, index, pm, dataset="train")
    print(f"done with charlie's test, index {index+1}")

def run_resim(idx, pm, dataset):

    path_results = "/develop/results/spie_journal_2023/resim_params"
    path_resims = os.path.join(path_results, pm.exp_name + pm.training_stage, dataset + '_info') # might need to change this too 
    model_results = pickle.load(open(os.path.join(path_resims,'resim.pkl'), 'rb'))

    phases = model_results['phase_pred'][idx]
    radii_list = mapping.phase_to_radii(phases)
    radii_list = np.round(radii_list, 6)
    radii_list = list(radii_list)
    run(radii_list, idx, pm, dataset=dataset)
    print(f"{dataset} resim complete.")

if __name__=="__main__":

    # Run experiment

    params = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader).copy()
    params['exp_name'] = "charlie_test"
    pm = parameter_manager.ParameterManager(params=params)
    pm.resim = 1
    pm.training_stage = ""
    print(f"resolution is {pm.resolution}")

    if pm.resim == 0: # datagen
        print("run_sim.py set to generate data")
        parser.add_argument("-index", type=int, help="The index matching the index in radii_neighbors")
        parser.add_argument("-path_out_sims", help="This is the path that simulations get dumped to") # this is empty in our config file. gets set in the kubernetes job file
           
        args = parser.parse_args() 

        idx = args.index
        path_out_sims = args.path_out_sims
        pm.path_dataset = path_out_sims
            
        neighbors_library = pickle.load(open("neighbors_library_allrandom.pkl", "rb"))
        radii_list = neighbors_library[idx]
        run(radii_list, idx, pm)
             

    # RESIMS
    elif pm.resim == 1:
        print("run_sim.py set to do resims.")
    
        
        parser = argparse.ArgumentParser()
        parser.add_argument("-index", help="")       
        args = parser.parse_args() 

        idx = int(args.index)

        charlies_test(pm, idx)
        """
        # Need to get phase values from the model's predictions.
        dataset="train"
        print(f"running resim for {pm.exp_name}, {dataset} set")
        run_resim(idx, pm, dataset)
         
        dataset="valid"
        print(f"running resim for {pm.exp_name}, {dataset} set")
        run_resim(idx, pm, dataset)
        """
#    elif pm.resim == 1:
#        print("run_sim.py set to do resims.")
#    
#        pm.training_stage = ""
#        pm.exp_name = "all_on"
#        
#        parser = argparse.ArgumentParser()
#        parser.add_argument("-index", help="")       
#        parser.add_argument("-dataset", help="")       
#        args = parser.parse_args() 
#
#        #radii = radii_list.strip('[]').split(',')
#        #radii_list = [float(radius.strip()) for radius in radii]
#        idx = int(args.index)
#        dataset = args.dataset
#
#        path_results = "/develop/results/spie_journal_2023/resim_params"
#
#        # Need to get phase values from the model's predictions.
#        path_resims = os.path.join(path_results, pm.exp_name + pm.training_stage, dataset + '_info') # might need to change this too 
#        model_results = pickle.load(open(os.path.join(path_resims,'resim.pkl'), 'rb'))
#
#        phases = model_results['phase_pred'][idx]
#
#        radii_list = mapping.phase_to_radii(phases)
#        radii_list = np.round(radii_list, 6)
#        radii_list = list(radii_list)
#        run(radii_list, idx, pm, dataset=dataset)
       
