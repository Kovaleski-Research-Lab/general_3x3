import os
import yaml
import pickle
import meep as mp
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import get_slice
import time
import h5py

import simulation
import field_monitors
import argparse


def create_folder(path):

    try: 
        os.makedirs(path)
        print(f"\ncreated folder {path}\n")

    except:
        print(f"\nfolder {path} already exists.\n")

def get_z_location(params,eps_data,nf):

    # put value on a (0 to pm.cell_z) scale - meep defines the cell on a (-cell_z/2 to cell_z/2) scale
    value = params['geometry']['loc_top_fused_silica'] + params['geometry']['height_pillar'] + params['source']['wavelength'] / 2

    cell_z = params['geometry']['cell_size'][2]
    value = value + cell_z / 2 
    # length of the cell in microns 
    cell_min = 0  # um
    cell_max = cell_z  # um

    # length of the cell in pixels
    pix_min = 0
    pix_max = eps_data.squeeze().shape[2]

    temp = int(((value - cell_min) / (cell_max - cell_min)) * (pix_max - pix_min) + pix_min)

    # temp is number of pixels based on eps data, which includes pml. we need to adjust for this
    # since we are getting a slice of dft_fields, which does NOT include the pml region.
    pml_pix = (eps_data.squeeze().shape[2] - nf.squeeze().shape[2]) // 2

    print(f"value = {value}")
    print(f"cell_max = {cell_max}")
    print(f"temp = {temp}")
    return temp - pml_pix

def mod_dft_fields(params, dft_fields, eps_data):

    wl_list = params['monitor']['wavelength_list']
    new_dft_fields = {}

    for wl, comps in dft_fields.items():
        x = comps[0]
        y = comps[1]
        z = comps[2]       
       
        z_loc = get_z_location(params,eps_data,x)
        print(f"z_loc = {z_loc}")        
        x_slice = x[:,:,z_loc] 
        y_slice = y[:,:,z_loc]
        z_slice = z[:,:,z_loc]

        new_dft_fields[wl] = [x_slice, y_slice, z_slice]
         
    return new_dft_fields

def get_slice_from_metadata(params,meta_data,dft_data):

    wl_list = params['source_params']['wavelength_list']    
    z_loc = params['loc_z_timedep_mon']
    
    x, y, z, w = meta_data
    z_loc = np.where(z > z_loc)[0][0]

    slices = {}
    for i, wl in enumerate(wl_list):

        x_field = np.asarray(dft_data[f'ex_{i}.r']) + 1j*np.asarray(dft_data[f'ex_{i}.i'])
        y_field = np.asarray(dft_data[f'ey_{i}.r']) + 1j*np.asarray(dft_data[f'ey_{i}.i'])
        z_field = np.asarray(dft_data[f'ez_{i}.r']) + 1j*np.asarray(dft_data[f'ez_{i}.i'])
       
        # ask marshall about this ->
        #z_loc = np.where(z > (-2.39 + (1.02/2) + (1.55/2)))[0][0]
        
        x_slice = x_field[:,:,z_loc]
        y_slice = y_field[:,:,z_loc]
        z_slice = z_field[:,:,z_loc]

        slices[wl] = [x_slice, y_slice, z_slice]

    return slices

def mod_axes(ax):

    font = {
        'family': 'sans-serif',
        'size': 16
           } 
    ax.set_xlabel('X [$\mu$m]', fontdict=font)
    ax.set_ylabel('Z [$\mu$m]', fontdict=font)
    ax.tick_params(axis='both', labelsize=14)
    return ax

def get_vis(params,until,sim,path_results,idx,animation=True,image=True):
        
    cell_x = params['cell_x']
    cell_y = params['cell_y']
    cell_z = params['cell_z']
   
    center_x = 0
    center_y = 0
    center_z = 0

    plot_plane = mp.Volume( center = mp.Vector3(center_x, center_y, center_z), 
                            size=mp.Vector3(cell_x, 0, cell_z))

    plot_modifiers = [mod_axes]
    f = plt.figure(dpi=100, figsize=(8,15))

    Animate = mp.Animate2D( output_plane = plot_plane,
                                fields = mp.Ey,
                                f = f,
                                realtime = False,
                                normalize = True,
                                plot_modifiers = plot_modifiers)
 
    sim.run(mp.at_every(0.1, Animate), until=until)

    if animation==True:
   
        print("saving animation...") 
        Animate.to_mp4(20, os.path.join(path_results, 'animation_idx_{}.mp4'.format(idx)))

    if image==True:
        fig,ax = plt.subplots(1,1,figsize = (5,5))
        sim.plot2D(output_plane = plot_plane, ax=ax)
        print("saving png image...")
        fig.savefig(os.path.join(path_results, 'plot2D_idx_{}.png'.format(idx)))

    return sim

if __name__ == "__main__":

    until = 9 * 2

    print("loading in params...")
    params = yaml.load(open("config.yaml", 'r'), Loader = yaml.FullLoader)

    print("parsing args...")
    parser = argparse.ArgumentParser()
    parser.add_argument("-idx", help="An integer value used to grab a radii list from radii library")

    args = parser.parse_args()
    idx = int(args.idx) 
        
    #path_results = "/develop/results/buffer_study"
    #path_results = "/develop/results/random_set"
    path_results = "/develop/results"

    folder_name = f"{str(idx).zfill(4)}"
    create_folder(os.path.join(path_results, folder_name))

    subfolder_name = "slices"
    create_folder(os.path.join(path_results, subfolder_name))

    ## folder for dumping general dataset (volume, not slices)
    path_results = os.path.join(path_results, folder_name)
    
    print("loading in neighbors library...")
    neighbors_library = pickle.load(open("neighbors_library_allrandom.pkl","rb"))
    #buffer_study = pickle.load(open("buffer-study.pkl","rb")) # all random, no fixed mu
    #radii = buffer_study['radii'] # 40 variance levels, 3 sets of radii each
    #neighbors_library = [item for sublist in radii for item in sublist]   # 120 sims 
   
    print(f"assigning neighborhood for idx {idx}...")
    radii = list(neighbors_library[idx])
    radii = np.array(radii).reshape(3,3)
    radii = np.flip(radii,axis=0).flatten()
    radii = list(radii)
    
    #6 7 8  -->  0 1 2
    #3 4 5       3 4 5
    #0 1 2       6 7 8

    #radii = [0.18664, 0.09511, 0.13333,
    #         0.16552, 0.19670, 0.13635,
    #         0.20876, 0.10517, 0.09009]
    #radii = [0.20876, 0.10517, 0.09009, 0.16552, 0.19670, 0.13635, 0.18664, 0.09511, 0.13333]
    
    print("building sim...")
    sim, dft_obj, flux_obj, params = simulation.build_sim(params, radii = radii)

    start_time = time.time()
    
    #sim = get_vis(params,until,sim,path_results,idx,animation=True,image=True)

    ## Do not run this if sim = get_vis() is called. (You'll extraneously run the simulation twice)
    sim.run(until=until)
    
    meta_data = sim.get_array_metadata(dft_cell = dft_obj)
    print("dumping metadata...")
    pickle.dump(meta_data, open(os.path.join(path_results, 'metadata_{}.pkl'.format(str(idx).zfill(5))), 'wb'))

    # this outputs a 3GB file   
    print("outputting full dft volume...")
    sim.output_dft(dft_obj, os.path.join(path_results, 'dft_{}'.format(str(idx).zfill(5))))


    #print("dumping eps data...")
    #pickle.dump(eps_data, open(os.path.join(path_results, '{}_epsdata_with_buffer_{:.03f}_rad_idx_{}.pkl'.format(source,_buffer,idx)), 'wb'))


    ## Everything we need to train the surrogate model goes in the 'slices' folder ##
    ##-----------------------------------------------------------------------------##

    dft_fields = h5py.File(os.path.join(path_results,'dft_{}.h5'.format(str(idx).zfill(5))))
    z_slice = get_slice_from_metadata(params,meta_data,dft_fields)
    #z_slice = mod_dft_fields(params,dft_fields,eps_data)

    training_data = {'slices': z_slice,
                     'radii':  radii,
                    }                  

    print(f"dumping surrogate model training data to {subfolder_name}...")
    #path_results = os.path.join(path_results, subfolder_name)
    #pickle.dump(training_data, open(os.path.join(path_results, '../slices', f'{str(idx).zfill(5)}.pkl'),'wb'))
    filename = os.path.join(path_results, '../slices', f'str(idx).zfill(5)}.pkl')
    with open(filename,'wb') as f:
        pickle.dump(training_data, f)
 
    ###
    end_time = round((time.time() - start_time) / 60, 3)
    print(f"all done. elapsed time: {end_time} minutes.")
