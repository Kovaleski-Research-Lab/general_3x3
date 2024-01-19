import numpy as np

def m_derivatives(phases):
   
    #delta_x = delta_y = 680e-9  # this is the lattice spacing 
    delta_x = delta_y = 1
    dims = phases.shape
    #print(f"dims={dims}")
    phases = np.reshape(phases, (1, dims[0]*dims[-1]))
    
    # taylor series expansion kernels
    fx_mat = [ [0,0,0],
              [-1,0,1],
              [0,0,0] ]
    fx_mat = np.array(fx_mat).flatten()

    fy_mat = [ [0,1,0],
              [0,0,0],
              [0,-1,0] ]
    fy_mat = np.array(fy_mat).flatten()

    fxx_mat = [ [0,0,0],
                [1,-2,1],
                [0,0,0] ]
    fxx_mat = np.array(fxx_mat).flatten()

    fyy_mat = [ [0,1,0],
                [0,-2,0],
                [0,1,0] ]
    fyy_mat = np.array(fyy_mat).flatten()
    
    fxy_mat = [ [1,0,-1],
                [0,0,0],
                [-1,0,1] ]
    fxy_mat = np.array(fxy_mat).flatten()

    fx_mat = np.reshape(fx_mat, (1,9))
    fy_mat = np.reshape(fy_mat, (1,9))
    fxx_mat = np.reshape(fxx_mat, (1,9))
    fyy_mat = np.reshape(fyy_mat, (1,9))
    fxy_mat = np.reshape(fxy_mat, (1,9))

    # The following math is just batch wise dot product. This is done to keep us
    # from needing to do weird indexing to get the derivatives. The indexing used here
    # is adapted from https://stackoverflow.com/questions/69230570/how-to-do-batched-dot-product-in-pytorch

    fx = ((phases / (2*delta_x)) * fx_mat[None, ...]).sum(axis=-1).squeeze()
    fy = ((phases / (2*delta_y)) * fy_mat[None, ...]).sum(axis=-1).squeeze()
    
    fxx = ((phases/(delta_x**2)) * fxx_mat[None, ...]).sum(axis=-1).squeeze()
    fyy = ((phases/(delta_y**2)) * fyy_mat[None, ...]).sum(axis=-1).squeeze()

    fxy = ((phases/(4 * delta_x**2)) * fxy_mat[None, ...]).sum(axis=-1).squeeze()
    
    fx = round(fx.tolist(), 8)
    fy = round(fy.tolist(), 8)
    fxx = round(fxx.tolist(), 8)
    fyy = round(fyy.tolist(), 8)
    fxy = round(fxy.tolist(), 8)

    return [fx, fy, fxx, fyy, fxy]