
## Some variables cannot be set in the config file so they are updated here ##
##--------------------------------------------------------------------------##

import meep as mp


def update(params):

    ## Update simulation params
    params['cell']['x'] = params['a'] * 3
    params['cell']['y'] = params['a'] * 3     

    params['cell']['z'] = (params['pml']['thickness'] + params['PDMS']['width'] + 
                           params['amorphousSi']['height'] + params['fusedSilica']['width'] + 
                           params['pml']['thickness'])
    
    params['freq'] = 1 / params['wavelength']

    params['cell_size'] = mp.Vector3(params['cell']['x'],params['cell']['y'],params['cell']['z'])

    params['k_point'] = mp.Vector3(0,0,0)

    ## Update geometry params

    params['PDMS']['center'] = (-0.5 * params['cell']['z'] + params['pml']['thickness'] + 
                            params['fusedSilica']['width'] + 
                            0.5 * (params['amorphousSi']['height'] + params['PDMS']['width'] + 
                            params['pml']['thickness']))

    params['fusedSilica']['center'] = (-0.5 * params['cell']['z'] + 
                                    0.5 * (params['pml']['thickness'] + 
                                    params['fusedSilica']['width']))

    params['amorphousSi']['center'] = (-0.5 * params['cell']['z'] + params['pml']['thickness'] + 
                              params['fusedSilica']['width'] + 0.5 * params['amorphousSi']['height'])

    params['pml']['layers'] = [mp.PML(thickness = params['pml']['thickness'], 
                            direction = mp.Z)]

    ## Update source params

    params['source']['center'] = round(params['pml']['thickness'] + 
                                 params['fusedSilica']['width']*0.2 - 
                                 0.5*params['cell']['z'], 3) 

    params['source']['cmpt'] = mp.Ey

    if params['source']['cmpt'] == mp.Ey:
        params['symmetries'] = [mp.Mirror(mp.X, phase=+1), #epsilon has mirror symmetry in x and y, phase doesn't matter
                                mp.Mirror(mp.Y, phase=-1)] #but sources have -1 phase when reflected normal to their direction
    elif params['source']['cmpt'] == mp.Ex:                      #use of symmetries important here, significantly speeds up sim
        params['symmetries'] = [mp.Mirror(mp.X, phase=-1),
                                mp.Mirror(mp.Y, phase=+1)]
    elif params['source']['cmpt'] == mp.Ez:
        params['symmetries'] = [mp.Mirror(mp.X, phase=+1),
                                mp.Mirror(mp.Y, phase=+1)]
    else:
        raise NotImplementedError

    ## Update flux params 
   
    params['flux']['center'] = round(0.5*params['cell']['z'] - params['pml']['thickness'] - 0.3*params['PDMS']['width'], 3)
        
    return params
