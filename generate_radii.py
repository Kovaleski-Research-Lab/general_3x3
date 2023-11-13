import numpy as np
import yaml
from IPython import embed
import pickle

if __name__=="__main__":

    params = yaml.load(open("config.yaml", 'r'), Loader = yaml.FullLoader)
    list_len = 9
    
    library = []
    
    #idx 0 - 2, uniform grids 
    library.append([params['geometry']['radius_min'] for _ in range(list_len)]) # uniform, min rad
    library.append([(params['geometry']['radius_max']/2) for _ in range(list_len)]) # uniform, central rad
    library.append([params['geometry']['radius_max'] for _ in range(list_len)]) # uniform, max rad

    #idx 3, linear increasing with y
    row_1 = params['geometry']['radius_max'] / 2
    row_0 = row_1 - params['geometry']['radius_max'] / 6
    row_2 = row_1 + params['geometry']['radius_max'] / 6

    list_3 = [[row_0 for _ in range(3)],
           [row_1 for _ in range(3)],
           [row_2 for _ in range(3)]]
    library.append( [item for sublist in list_3 for item in sublist] )

   #idx 4, transpose idx 3 -- linear increasing with x 
    temp = np.asarray(list_3)
    temp = temp.T
    temp = list(temp)
    
    library.append( [item for sublist in temp for item in sublist] )

    # idx 5 through 15, random with small variation

    rows, cols = 3, 3
   
    i = 0 
#    while i < 10:
#        central_pillar = np.random.uniform(params['geometry']['radius_min'], params['geometry']['radius_max'])
#        
#        gaussian_vals = np.random.normal(0, 0.005 * (i +1), (rows, cols))
#        
#        grid = central_pillar + gaussian_vals
#
#        grid = np.clip(grid, params['geometry']['radius_min'], params['geometry']['radius_max'])
#        
#        grid = grid.flatten()
#
#        if len(np.unique(grid)) != len(grid):
#            print("We skipped ", grid)
#        else:
#            library.append(grid.tolist())
#            i += 1
#    pickle.dump(library, open("buffer_study_library.pkl", "wb"))
    _min = params['geometry']['radius_min']
    _max = params['geometry']['radius_max']
    
    while i < 10: 

        initial_value = np.random.uniform(_min, _max)
        std_dev = 0.005 * (i + 1)
        print(std_dev)
 
        radii = [initial_value + np.random.normal(0, std_dev) for _ in range(9)]

        radii = np.clip(radii, _min, _max)
        
        # Ensure no duplicates in the sublist
        while len(set(radii)) < len(radii):
            radii = [initial_value + np.random.normal(0, std_dev) for _ in range(9)]
            radii = np.clip(radii, _min, _max)

        # Append the generated sublist to the list of lists
        library.append(radii)
        i += 1 
    #pickle.dump(library, open("buffer_study_library.pkl", "wb"))
    embed()
