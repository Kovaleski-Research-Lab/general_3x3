import os
import yaml
import pickle
import meep as mp
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import get_slice

import simulation
import field_monitors

import argparse

font = {
    'family': 'sans-serif',
    'size': 16
}

def mod_axes(ax):
    ax.set_xlabel('X [$\mu$m]', fontdict=font)
    ax.set_ylabel('Z [$\mu$m]', fontdict=font)
    ax.tick_params(axis='both', labelsize=14)
    return ax

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

    return temp - pml_pix

def mod_dft_fields(params, dft_fields, eps_data):

    wl_list = params['monitor']['wavelength_list']
    new_dft_fields = {}

    for wl, comps in dft_fields.items():
        x = comps[0]
        y = comps[1]
        z = comps[2]       
       
        z_loc = get_z_location(params,eps_data,x)
        
        x_slice = x[:,:,z_loc] 
        y_slice = y[:,:,z_loc]
        z_slice = z[:,:,z_loc]

        new_dft_fields[wl] = [x_slice, y_slice, z_slice]
         
    return new_dft_fields

if __name__ == "__main__":

    print("loading in params...")
    params = yaml.load(open("config.yaml", 'r'), Loader = yaml.FullLoader)

    print("parsing args...")
    parser = argparse.ArgumentParser()
    parser.add_argument("-idx", help="An integer value used to grab a radii list from radii library")

    args = parser.parse_args()

    #path_results = "/develop/results/buffer_study"
    path_results = "/develop/results"

    idx = int(args.idx) 
    print("loading in neighbors library...")
    #neighbors_library = pickle.load(open("buffer_study_library.pkl", "rb"))
    #neighbors_library = pickle.load(open("buffer_study_random_radii_only.pkl","rb"))
    #neighbors_library = pickle.load(open("short_incy.pkl","rb"))
    neighbors_library = pickle.load(open("neighbors_library_allrandom.pkl","rb"))
    
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

    cell_x = params['cell_x']
    cell_y = params['cell_y']
    cell_z = params['cell_z']
   
    center_x = 0
    center_y = 0
    center_z = 0
    
    source = params['source']['type']
    _buffer = params['geometry']['size_x_buffer']

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

    #sim.run(mp.at_every(0.1, Animate), until=15)
    sim.run(until=15)
    
    folder_name = f"{str(idx).zfill(5)}"
    create_folder(os.path.join(path_results, folder_name))

    subfolder_name = "slices"
    create_folder(os.path.join(path_results, subfolder_name))

    dft_fields, flux, eps_data = field_monitors.collect_fields(params, sim, flux_obj, dft_obj)
    meta_data = sim.get_array_metadata(dft_cell = dft_obj)
    
    path_results = os.path.join(path_results, folder_name)

    #print("saving animation...") 
    #Animate.to_mp4(20, os.path.join(path_results, 'test_anim_rad_idx_{}.mp4'.format(idx)))

    # these are sliced. 
    sliced_dft_fields = mod_dft_fields(params,dft_fields,eps_data)
    data = {'slices': sliced_dft_fields,
            'radii': radii,
           }

    print(f"dumping sliced field and radii info to {subfolder_name}...")
    pickle.dump(data, open(os.path.join(path_results, '../slices', f'{str(idx).zfill(5)}.pkl'),'wb'))

    #fig,ax = plt.subplots(1,1,figsize = (5,5))
    #sim.plot2D(output_plane = plot_plane, ax=ax)
    #print("saving png image...")
    #fig.savefig(os.path.join(path_results, 'test_plot2D_with_buffer_rad_idx_{}.png'.format(idx)))
    #from IPython import embed; embed() 
    
    # this outputs a 3GB file   
    print("outputting full dft volume...")
    sim.output_dft(dft_obj, os.path.join(path_results, 'outputdft_{}.pkl'.format(str(idx).zfill(5))))

    print("dumping metadata...")
    pickle.dump(meta_data, open(os.path.join(path_results, 'test_metadata_{}.pkl'.format(str(idx).zfill(5))), 'wb'))

    #print("dumping eps data...")
    #pickle.dump(eps_data, open(os.path.join(path_results, '{}_epsdata_with_buffer_{:.03f}_rad_idx_{}.pkl'.format(source,_buffer,idx)), 'wb'))

    print("all done.")
