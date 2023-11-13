import numpy as np
import yaml
from IPython import embed
import pickle

if __name__=="__main__":

    params = yaml.load(open("config.yaml", 'r'), Loader = yaml.FullLoader)
    list_len = 9
    
    library = []
    
    #idx 0 - 2, uniform grids 
    #library.append([params['geometry']['radius_min'] for _ in range(list_len)]) # uniform, min rad
    #library.append([(params['geometry']['radius_max']/2) for _ in range(list_len)]) # uniform, central rad
    #library.append([params['geometry']['radius_max'] for _ in range(list_len)]) # uniform, max rad

    # idx 5 through 15, random with small variation

    rows, cols = 3, 3
   
    i = 0 
    _min = params['geometry']['radius_min']
    _max = params['geometry']['radius_max']
    
    while i < 10: 

        initial_value = params['geometry']['radius_max'] / 2 
        std_dev = 0.005 * (i + 1)
        print(std_dev)
 
        radii = [initial_value + np.random.normal(0, std_dev) for _ in range(9)]

        radii = np.clip(radii, _min, _max)
        
        # Ensure no duplicates in the sublist
        while len(set(radii)) < len(radii):
            radii = [initial_value + np.random.normal(0, std_dev) for _ in range(9)]
            radii = np.clip(radii, _min, _max)

        # Append the generated sublist to the list of lists
        library.append(list(radii))
        i += 1 
    pickle.dump(library, open("buffer_study_library.pkl", "wb"))
    #embed()
