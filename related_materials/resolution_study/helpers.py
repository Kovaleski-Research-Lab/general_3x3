import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml
import os
import meep as mp
import cv2

def get_colors(num_colors):

    cmap_viridis = plt.cm.get_cmap('viridis')
    colors = [cmap_viridis(i / num_colors) for i in range(num_colors)]

    return colors

def get_raw_data(res, root):

    res_str = str(res).zfill(3)
    filename = os.path.join(root, f"field_info/sim_res_{res_str}.pkl")
    
    results = pickle.load(open(filename,"rb"))

    eps_data = results['epsilon']
    ex_data = results['x']
    ey_data = results['y']
    ez_data = results['z']

    try:
        with open(os.path.join(root, f"params/params_{res_str}.yaml")) as stream:
            params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

    if res != params['resolution']:
        raise ValueError("Mismatch between resolution and yaml file")

    #print(f"for resolution {params['resolution']}, eps_data.shape is {eps_data.shape}")
    return params, eps_data, ex_data, ey_data, ez_data

def get_slice(params, eps_data, ex_data, ey_data, ez_data):

    cell_x = params['cell_x']
    cell_y = params['cell_y']
    cell_z = params['cell_z']

    top_pillar = params['thickness_pml'] + params['size_z_fused_silica'] + (params['height_pillar'] - cell_z * 0.5)
    wavelength = params['wavelength']
    target = top_pillar + wavelength / 2
    
    cell_size = mp.Vector3(cell_x, cell_y, cell_z)
    
    delta_z = cell_size[2] / eps_data.shape[2]

    num_pix_to_top_of_pillar = int(np.ceil(top_pillar / delta_z))
    num_pix_to_field_monitor = int(np.ceil((wavelength / 2) / delta_z))
    slice_location_pix = num_pix_to_top_of_pillar + num_pix_to_field_monitor

    return slice_location_pix

def get_components(slice, ex_data, ey_data, ez_data):
    
    ex_comp = ex_data[:,:,slice]
    ey_comp = ey_data[:,:,slice]
    ez_comp = ez_data[:,:,slice]

    return ex_comp, ey_comp, ez_comp

def get_min_and_max(ex_comp, ey_comp, ez_comp):

    min_evalue = np.min((ex_comp, ey_comp, ez_comp))

    ex = ex_comp - min_evalue
    ey = ey_comp - min_evalue
    ez = ez_comp - min_evalue

    max_value = np.max((ex, ey, ez))

    return min_evalue, max_value

def get_norm_fields(ex_comp, ey_comp, ez_comp, global_min, global_max):

    ex = ex_comp - global_min
    ey = ey_comp - global_min
    ez = ez_comp - global_min

    ex_norm = ex / global_max
    ey_norm = ey / global_max
    ez_norm = ez / global_max

    return ex_norm, ey_norm, ez_norm

def resize_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def compare_images(image1, image2):
    # Find the common size (e.g., the size of the smallest image)
    common_size = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0]))
    
    # Resize both images to the common size
    resized_image1 = resize_image(image1, common_size)
    resized_image2 = resize_image(image2, common_size)
    
    # Compute the absolute difference
    difference = np.abs(resized_image1 - resized_image2)
    
    return difference

# Example usage:
# Load images (assuming grayscale images for simplicity)
# im_at_20 = cv2.imread('path_to_image_at_20.png', cv2.IMREAD_GRAYSCALE)
# im_at_30 = cv2.imread('path_to_image_at_30.png', cv2.IMREAD_GRAYSCALE)

# difference_20_30 = compare_images(im_at_20, im_at_30)

def plot_fields(ex, ey, ez, res, prior=None, ex_minus1=None, ey_minus1=None, ez_minus1=None):

    if prior == None:
        fig, ax = plt.subplots(1,3,figsize=(12,5))
        fig.suptitle(f"E Fields, normalized, resolution {res}",size=16)
    
        ax[0].set_title("x component")
        ax[1].set_title("y component")
        ax[2].set_title("z component")
    
        im_0 = ax[0].imshow(ex)
        im_1 = ax[1].imshow(ey)
        im_2 = ax[2].imshow(ez)
    
        im_0.set_clim(0,1)
        im_1.set_clim(0,1)
        im_2.set_clim(0,1)
    
        for axs in ax:
            axs.grid(False)
            
        fig.colorbar(im_1,ax=ax[:3],location='bottom',shrink=0.4)

    else:
        fig, ax = plt.subplots(2,6,figsize=(12,5))
        fig.suptitle(f"E Fields, normalized, resolution {res}",size=16)
    
        ax[0,0].set_title(f"Ex, Res {res}")
        ax[0,1].set_title(f"Ey, Res {res}")
        ax[0,2].set_title(f"Ez, Res {res}")

        ax[1,0].set_title(f"Ex, Res {res-5}")
        ax[1,1].set_title(f"Ey, Res {res-5}")
        ax[1,2].set_title(f"Ey, Res {res-5}")
    
        im_0 = ax[0,0].imshow(ex)
        im_1 = ax[0,1].imshow(ey)
        im_2 = ax[0,2].imshow(ez)

        im_3 = ax[1,0].imshow(ex_minus1)
        im_4 = ax[1,1].imshow(ey_minus1)
        im_5 = ax[1,2].imshow(ez_minus1)
    
        im_0.set_clim(0,1)
        im_1.set_clim(0,1)
        im_2.set_clim(0,1)

        im_3.set_clim(0,1)
        im_4.set_clim(0,1)
        im_5.set_clim(0,1)

        ax[0,3].set_title(f"Ex, Abs Diff")
        ax[0,4].set_title(f"Ey, Abs Diff")
        ax[0,5].set_title(f"Ey, Abs Diff")

        im_7 = ax[0,3].imshow(compare_images(ex, ex_minus1))
        im_8 = ax[0,4].imshow(compare_images(ey, ey_minus1))
        im_9 = ax[0,5].imshow(compare_images(ez, ez_minus1))

        ax[1,3].remove()
        ax[1,4].remove()
        ax[1,5].remove()

        for axs in ax.flat:
            axs.grid(False)

        left = ax[0, 3].get_position().x0 + 0.1
        right = ax[0, 5].get_position().x1
        bottom = ax[1, 3].get_position().y0 + 0.1  # Adjust this value as needed
        height = 0.05  # Height of the colorbar
        
        # Add a horizontal colorbar at the calculated position
        cbar_ax = fig.add_axes([left, bottom, right - left, height])
        fig.colorbar(im_1, cax=cbar_ax, orientation='horizontal')

        #fig.colorbar(im_1,ax=ax,location='bottom',shrink=0.4)
        fig.tight_layout()