import pickle
import torch
import numpy as np
import os
import h5py
import re

path_results = "/develop/results/all_random" #KUBE
#path_results = "/develop/data/buffer_study/all_random" # MARGE TEST
dump_path = os.path.join(path_results, "slices")

def create_folder(path):

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"\ncreated folder {path}\n")
    else:
        print(f"\nfolder {path} already exists.\n")
        pass

def get_slice(path_results, folder, meta_data, dft_data):

    print("assigning x, y, z, w...")
    try:
        x,y,z,w = pickle.load(open(os.path.join(path_results, folder, meta_data), 'rb'))
        print("successfully assigned x, y, z, w")
    except Exception as e:
        print(f"an error occurred trying to assign meta data. excluding {folder} from results.")
        return 0

    print("assigning field_data...")
    try:
        field_data = h5py.File(os.path.join(path_results, folder, dft_data))    
        print("successfully assigned field data")
    except:
        print(f"an error occurred trying to assign field data. exculding {folder} from results.")
        return 0
    
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
        if "metadata" in filename:
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
    print("test get_cropped_im() successful")    
    return cropped

#exclude_indices = [ 10,114,12,149,156,158,159,169,18,271,38,55,7, 
#                    139, 156, 149, 151, 156, 158, 159, 169, 171, 172,
#                    174, 179, 18, 231, 232, 250, 271, 272, 33, 38,
#                    39, 41, 48, 5, 55, 58, 60, 61, 64, 7,
#                    77, 8, 82, 84, 85, 86, 89, 90, 93, 95]

if __name__=="__main__":

    create_folder(dump_path)
    print(f"folder created: {dump_path}")
    slices = {}
    #radii = pickle.load(open("buffer_study_library.pkl","rb"))
    print(f"path_results: {path_results}")
    print("beginning slicing...")
    for folder in os.listdir(path_results):
        print(folder, type(folder))
        if folder == "current_logs":
            continue
        if folder == "initial_buffer_study":
            continue
        if folder == "slices":
            continue
        if folder == "kube_logs":
            continue

        if folder.startswith('idx_'):
            folder_index = int(folder.split('_')[1])
            #if folder_index not in exclude_indices:
            if folder_index:
                print(f"got {folder}, assigning index...")
                idx = get_index(folder) 
                print(f" folder index is {idx}")
                #slices[f'index_{idx}'] = {}
                #print(f"dictionary now has keys {slices.keys()}")
                print("assigning meta_data...")
                meta_data = f"gaussian_metadata_with_buffer_5.000_rad_idx_{idx}.pkl"
                print("assigning dft data...")
                dft_data = f"gaussian_outputdft_with_buffer_5.000_rad_idx_{idx}.pkl.h5"
                #eps_data = f"gaussian_epsdata_with_buffer_5.000_rad_idx_{idx}.pkl"
                
                print("getting slice...")
                z_slice = get_slice(path_results, folder, meta_data, dft_data)
                print(f"z_slice type is {type(z_slice)}")
                if isinstance(z_slice, np.ndarray):
                    print("cropping...")
                    z_slice = get_cropped_im(z_slice)

                    filename = os.path.join(dump_path, f"dft_slice_{idx.zfill(3)}.pkl")
                    print(f"dumping to {filename}.")
                    with open(filename, "wb") as f:
                        pickle.dump(z_slice, f)
                else:
                    continue
            else:
                print(f"Excluded folder {folder}")
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
