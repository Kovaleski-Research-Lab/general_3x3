import pickle
import torch
import numpy as np
import os
import h5py
from IPython import embed
import re

path_results = "/develop/results"
dump_path = os.path.join(path_results, "slices")

def create_folder(path):

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"\ncreated folder {path}\n")
    else:
        print(f"\nfolder {path} already exists.\n")
        pass

def get_slice(path_results, folder, meta_data):

    x,y,z,w = pickle.load(open(os.path.join(path_results, folder, meta_data), 'rb'))
    field_data = h5py.File(os.path.join(path_results, folder, dft_data))    
    y_field = np.asarray(field_data['ey_2.r']) + 1j*np.asarray(field_data['ey_2.i'])
    z_slice = np.where(z > (-2.39 + (1.02/2) + (1.55/2)))[0][0]

    return y_field[:,:,z_slice]

def get_eps():

    eps = pickle.load(open(os.path.join(path_results, folder, eps_data), 'rb'))
    z_dim = eps.shape[2]
    z_slice = eps[:,:,int(z_dim/4)]
    
    return z_slice

def get_index(folder):
    for filename in os.listdir(os.path.join(path_results,folder)):
        if "epsdata" in filename:
            match = re.search(r'idx_(\d+)',filename)
            idx = int(match.group(1))
            return idx

def get_cropped_im(image):

    full_pix = image.shape[0]
    crop_pix = 166 

    start_row = (full_pix - crop_pix) // 2
    end_row = start_row + crop_pix
    
    start_col = (full_pix - crop_pix) // 2
    end_col = start_col + crop_pix
    
    cropped = image[start_row:end_row, start_col:end_col]
    
    return cropped


if __name__=="__main__":
    slices = {}
    radii = pickle.load(open("buffer_study_library.pkl","rb"))
    print(f"path_results: {path_results}")
    print("beginning slicing...")
    for folder in os.listdir(path_results):
        print(folder, type(folder))
        if folder == "current_logs":
            continue
        if folder == "current_logs":
            continue
        print(f" got {folder}, assigning index...")
        idx = get_index(folder) 
        print(f" folder index is {idx}")
        slices[f'index_{idx}'] = {}
        print(f"dictionary now has keys {slices.keys()}")
        
        meta_data = f"gaussian_metadata_with_buffer_5.000_rad_idx_{idx}.pkl"
        dft_data = f"gaussian_outputdft_with_buffer_5.000_rad_idx_{idx}.pkl.h5"
        eps_data = f"gaussian_epsdata_with_buffer_5.000_rad_idx_{idx}.pkl"

        z_slice = get_slice(path_results, folder, meta_data)
        z_slice = get_cropped_im(z_slice)

        slices[f'index_{idx}']['slice'] = z_slice
        print(f"Assigned slice to index {idx}, folder {folder}")
        
        slices[f'index_{idx}']['radii'] = radii[idx]
    
    create_folder(dump_path)
    filename = os.path.join(dump_path, "dft_slices.pkl")
    print(f"dumping to {filename}."
    with open(filename, "wb") as f:
        pickle.dump(slices, f)
  
    print("all done") 
    #filename = os.path.join(dump_path, "z_slice.pkl")
    #with open(filename, "wb") as f:
    #    pickle.dump(z_slice, f)

    #create_folder(dump_path)
    #eps_slice = get_eps()
    #filename = os.path.join(dump_path, "eps_slice.pkl")
    #with open(filename, "wb") as f:
    #    pickle.dump(eps_slice, f)
#
#idx = 10
#folder = f"idx_{idx}"
#
#path_results = "/develop/results/buffer_study"
#meta_data = f"gaussian_metadata_with_buffer_5.000_rad_idx_{idx}.pkl"
#dft_data = f"gaussian_outputdft_with_buffer_5.000_rad_idx_{idx}.pkl.h5"
#eps_data = f"gaussian_epsdata_with_buffer_5.000_rad_idx_{idx}.pkl"
#
#dump_path = os.path.join(path_results, folder, "slice")
#
#def create_folder(path):
#
#    if not os.path.exists(path):
#        os.makedirs(path)
#        print(f"\ncreated folder {path}\n")
#    else:
#        print(f"\nfolder {path} already exists.\n")
#        pass
#
#def get_slice():
#
#    x,y,z,w = pickle.load(open(os.path.join(path_results, folder, meta_data), 'rb'))
#    field_data = h5py.File(os.path.join(path_results, folder, dft_data))    
#    y_field = np.asarray(field_data['ey_2.r']) + 1j*np.asarray(field_data['ey_2.i'])
#    z_slice = np.where(z > (-2.39 + (1.02/2) + (1.55/2)))[0][0]
#
#    return y_field[:,:,z_slice]
#
#def get_eps():
#
#    eps = pickle.load(open(os.path.join(path_results, folder, eps_data), 'rb'))
#    z_dim = eps.shape[2]
#    z_slice = eps[:,:,int(z_dim/4)]
#    
#    return z_slice
#
#if __name__=="__main__":
#
#    z_slice = get_slice()
#    
#    create_folder(dump_path)
#    
#    filename = os.path.join(dump_path, "z_slice.pkl")
#    with open(filename, "wb") as f:
#        pickle.dump(z_slice, f)
#
#    #create_folder(dump_path)
#    #eps_slice = get_eps()
#    #filename = os.path.join(dump_path, "eps_slice.pkl")
#    #with open(filename, "wb") as f:
#    #    pickle.dump(eps_slice, f)
#
