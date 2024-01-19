import h5py
import numpy as np
import pickle
import os
import re
import yaml

path_data = '/develop/data/meep-dataset-v2'
#path_library = '/develop/code/general_3x3/neighbors_library_allrandom.pkl'
dump_path = os.path.join(path_data,"volumes")

def get_volumes(params,meta_data,dft_data):

    loc_z_timedep_mon = 0.14500000000000013
    wl_list = params['monitor']['wavelength_list']
    #z_top = params['loc_z_timedep_mon']
    z_top = loc_z_timedep_mon
    z_bottom = z_top - 0.775

    x, y, z, w = meta_data
    z_loc1 = np.where(z > z_top)[0][0] + 1 
    z_loc2 = np.where(z > z_bottom)[0][0]

    volumes = {}
    for i, wl in enumerate(wl_list):

        x_field = np.asarray(dft_data[f'ex_{i}.r']) + 1j*np.asarray(dft_data[f'ex_{i}.i'])
        y_field = np.asarray(dft_data[f'ey_{i}.r']) + 1j*np.asarray(dft_data[f'ey_{i}.i'])
        z_field = np.asarray(dft_data[f'ez_{i}.r']) + 1j*np.asarray(dft_data[f'ez_{i}.i'])

        x_vol = x_field[:,:,z_loc2:z_loc1]
        y_vol = y_field[:,:,z_loc2:z_loc1]
        z_vol = z_field[:,:,z_loc2:z_loc1]

        volumes[wl] = [x_vol, y_vol, z_vol] 

    return volumes 

def dump_volumes(volumes,idx):

    filename = str(idx).zfill(4)
    filename = filename + ".pkl"

    with open(os.path.join(dump_path,filename),'wb') as f:
        pickle.dump(volumes, f)
        print(f"{filename} dumped successfully.")

if __name__=='__main__':    

    params = yaml.load(open("/develop/code/general_3x3/config.yaml","r"),Loader=yaml.FullLoader)

    exclude = ['volumes','preprocessed_data','0000','0001','0002','0003','0004']
    with os.scandir(path_data) as entries:
        
        for entry in entries:

            #if entry.is_dir() and entry.name != "volumes":
            if entry.is_dir() and entry.name not in exclude:
                     
                with os.scandir(entry.path) as files:
                     
                    for file_entry in files:

                        if file_entry.is_file():
                        
                            match = re.search(r'\d+',file_entry.name)
                            idx = int(match.group())
                            
                            if file_entry.name.endswith(".h5"):
                                try: 
                                   
                                    h5_path = os.path.join(path_data, entry, file_entry)
                                    h5_file = h5py.File(h5_path)

                                except:
        
                                    print("h5py File Error")

                            elif file_entry.name.endswith(".pkl"):
                                try:
                                    pkl_path = os.path.join(path_data, entry, file_entry)
                                    pkl_file = pickle.load(open(pkl_path, "rb"))
                            
                                except:
                                    print("pickle error")

                    volumes = get_volumes(params,pkl_file,h5_file)
                    dump_volumes(volumes,idx)
