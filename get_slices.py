import h5py
import numpy as np
import torch
import os

path_results = "/develop/results"

def create_folder(path):

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"\ncreated folder {path}\n")
    else:
        print(f"\nfolder {path} already exists.\n")
        pass

def get_slice():

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

def get_cropped_im(params, image):

    full_pix = image.shape[0]
    crop_pix = 166

    start_row = (full_pix - crop_pix) // 2
    end_row = start_row + crop_pix
    
    start_col = (full_pix - crop_pix) // 2
    end_col = start_col + crop_pix
    
    cropped = image[start_row:end_row, start_col:end_col]
    
    return cropped

if __name__=="__main__":

    for 
        folder = f"idx_{idx}"
        
        meta_data = f"gaussian_metadata_with_buffer_5.000_rad_idx_{idx}.pkl"
        dft_data = f"gaussian_outputdft_with_buffer_5.000_rad_idx_{idx}.pkl.h5"
        eps_data = f"gaussian_epsdata_with_buffer_5.000_rad_idx_{idx}.pkl"
        
        dump_path = os.path.join(path_results, folder, "slice")

        z_slice = get_slice()

        create_folder(dump_path)

        filename = os.path.join(dump_path, "z_slice.pkl")
        with open(filename, "wb") as f:
            pickle.dump(z_slice, f)
