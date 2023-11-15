import os
import yaml
import pickle
import meep as mp
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt


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
    neighbors_library = pickle.load(open("buffer_study_library.pkl", "rb"))
    print("assigning neighborhood...")
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
    sim, dft_obj, flux_obj = simulation.build_sim(params, radii = radii)

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

    sim.run(mp.at_every(0.1, Animate), until=25)
    #dft_fields, flux, eps_data = field_monitors.collect_fields(params, sim, flux_obj, dft_obj)
    meta_data = sim.get_array_metadata(dft_cell = dft_obj)
    eps_data = sim.get_epsilon()

    folder_name = f"idx_{idx}"
    create_folder(os.path.join(path_results, folder_name))
    path_results = os.path.join(path_results, folder_name)
    #sim.output_dft(dft_obj, os.path.join(path_results, '{}_outputdft_with_buffer_{:.03f}_rad_idx_{}.pkl'.format(source,_buffer,idx)))
    #pickle.dump(meta_data, open(os.path.join(path_results, '{}_metadata_with_buffer_{:.03f}_rad_idx_{}.pkl'.format(source,_buffer,idx)), 'wb'))
    #pickle.dump(eps_data, open(os.path.join(path_results, '{}_epsdata_with_buffer_{:.03f}_rad_idx_{}.pkl'.format(source,_buffer,idx)), 'wb'))
    #Animate.to_mp4(20, os.path.join(path_results, '{}_animation_with_buffer_{:.03f}_rad_idx_{}.mp4'.format(source,_buffer,idx)))
     
    source = params['source']['type']
    _buffer = params['geometry']['size_x_buffer']
    print("outputting dfts...")
    
    sim.output_dft(dft_obj, os.path.join(path_results, '{}_outputdft_with_buffer_{:.03f}_rad_idx_{}.pkl'.format(source,_buffer,idx)))
    print("dumping metadata...")
    pickle.dump(meta_data, open(os.path.join(path_results, '{}_metadata_with_buffer_{:.03f}_rad_idx_{}.pkl'.format(source,_buffer,idx)), 'wb'))
    print("dumping eps data...")
    pickle.dump(eps_data, open(os.path.join(path_results, '{}_epsdata_with_buffer_{:.03f}_rad_idx_{}.pkl'.format(source,_buffer,idx)), 'wb'))
    
    print("saving animation...")
    Animate.to_mp4(20, os.path.join(path_results, '{}_animation_with_buffer_{:.03f}_rad_idx_{}.mp4'.format(source,_buffer,idx)))

    fig,ax = plt.subplots(1,1,figsize = (5,5))
    sim.plot2D(output_plane = plot_plane, ax=ax)
    #fig.savefig(os.path.join(path_results, '{}_plot2D_with_buffer_{:.03f}_rad_idx_{}.png'.format(source,_buffer,idx)))
    fig.savefig(os.path.join(path_results, '{}_plot2D_with_buffer_{:.03f}_rad_idx_{}.png'.format(source,_buffer,idx)))
    print("all done")
