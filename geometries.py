###################
# Standard imports
###################
import meep as mp
from loguru import logger



##############################################################################
# Simple build functions on the MEEP primatives. Might be overkill - 
# the current primatives are pretty abstract already. These are nice for 
# logging purposes at least.
##############################################################################

def build_cylinder(loc:list, axis:list, height:float, radius:float, material_index:float) -> mp.Cylinder:
    logger.info("Building a MEEP cylinder")

    logger.info("Creating cylinder material. Index = {}".format(material_index))
    material = mp.Medium(index = material_index)
    center = mp.Vector3(loc[0], loc[1], loc[2])
    axis = mp.Vector3(axis[0], axis[1], axis[2])

    return mp.Cylinder( radius = radius,
                        height = height,
                        axis = axis,
                        center = center,
                        material = material )

def build_block(size:list, loc:list, material_index:float) -> mp.Block:
    logger.info("Building a MEEP block")

    logger.info("Creating block material. Index = {}".format(material_index))
    material = mp.Medium(index = material_index)
    size = mp.Vector3(size[0], size[1], size[2])
    center = mp.Vector3(loc[0], loc[1], loc[2])

    return mp.Block( size = size,
                     center = center,
                     material = material )


##############################################################################
# More complex build functions. Combines the ones above into more complicated
# structures, i.e., a metasurface.
##############################################################################

def build_silica_pdms_substrate(params:dict) -> list:

    ##################################
    #     _____________________
    #    |        PDMS         |
    #    |_____________________|
    #    |    FUSED SILICA     |
    #    |_____________________|
    #
    ##################################

    logger.info("Building silica + PDMS substrate.")
    logger.info("Reading fused silica parameters")
    size_x_fused_silica = params['size_x_fused_silica']
    size_y_fused_silica = params['size_y_fused_silica']
    size_z_fused_silica = params['size_z_fused_silica']
    
    loc_x_fused_silica = params['loc_x_fused_silica']
    loc_y_fused_silica = params['loc_y_fused_silica']
    loc_z_fused_silica = params['loc_z_fused_silica']
    material_index_fused_silica = params['material_index_fused_silica']

    logger.info("Creating fused silica material. Index = {}".format(material_index_fused_silica))

    fused_silica = build_block( size = [size_x_fused_silica, size_y_fused_silica, size_z_fused_silica],
                                loc = [loc_x_fused_silica, loc_y_fused_silica, loc_z_fused_silica],
                                material_index = material_index_fused_silica)

    logger.info("Reading PDMS parameters")
    loc_x_pdms = params['loc_x_pdms']
    loc_y_pdms = params['loc_y_pdms']
    loc_z_pdms = params['loc_z_pdms']
    
    size_x_pdms = params['size_x_pdms']
    size_y_pdms = params['size_y_pdms']
    size_z_pdms = params['size_z_pdms']

    material_index_pdms = params['material_index_pdms']

    logger.info("Creating fused silica material. Index = {}".format(material_index_fused_silica))

    pdms = build_block( size = [size_x_pdms, size_y_pdms, size_z_pdms],
                        loc = [loc_x_pdms, loc_y_pdms, loc_z_pdms],
                        material_index = material_index_pdms )
    
    return [fused_silica, pdms]

def build_andy_metasurface_neighborhood(params):
    '''
    This is basically the same code as the parameter manager's calculate_dependencies
    from the surrogate model code. Just with additional comments and different
    organization.
    Builds a fused silica + pdms substrate with a 3x3 pillar neighborhood.
    '''

    geometry_params = params['geometry']
    Nx, Ny = geometry_params['neighborhood_size']
    logger.info("Creating metasurface neighborhood. Nx,Ny : {},{}".format(Nx,Ny))

    atom_type = geometry_params['atom_type']
    logger.info("Meta-atom type: {}".format(atom_type))

    unit_cell_size = geometry_params['unit_cell_size']
    logger.info("Meta-atom size: {}".format(unit_cell_size))

    #Define the z stack size
    thickness_pml = geometry_params['thickness_pml']
    height_pillar = geometry_params['height_pillar']
    size_z_pdms = round(geometry_params['size_z_pdms'] + height_pillar, 3)
    size_z_fused_silica = geometry_params['size_z_fused_silica']

    #Add all of the z sizes together to get the total z size
    size_z_cell = round(thickness_pml + size_z_fused_silica + size_z_pdms + thickness_pml, 3)
    #Multiply the unit cell size by the numer of unit cells to get the x and y sizes
    size_x_cell = round(unit_cell_size * Nx, 3)
    size_y_cell = round(unit_cell_size * Ny, 3)
    params['cell_x'] = size_x_cell
    params['cell_y'] = size_y_cell
    params['cell_z'] = size_z_cell

    logger.info("Size of total geometry cell : {} x {} x {} [um]".format(size_x_cell, size_y_cell, size_z_cell))
    cell_size = mp.Vector3(size_x_cell, size_y_cell, size_z_cell)

    #Get the z locations (centers)  of all of the geometry objects
    loc_z_fused_silica = round(thickness_pml + 0.5 * size_z_fused_silica,3)
    logger.info("Center Z of fused silica : {} [um]".format(loc_z_fused_silica))
    loc_z_pdms = round(thickness_pml + size_z_fused_silica + 0.5 * size_z_pdms, 3)
    logger.info("Center Z of pdms : {} [um]".format(loc_z_pdms))
    loc_z_pillar = round(thickness_pml + size_z_fused_silica + 0.5 * height_pillar, 3)
    logger.info("Center Z of pillars : {} [um]".format(loc_z_pillar))

    loc_top_fused_silica = round(thickness_pml + size_z_fused_silica, 3)
    logger.info("Top of the fused silica : {} [um]".format(loc_top_fused_silica))
    loc_top_pdms = round(thickness_pml + size_z_fused_silica + size_z_pdms, 3)
    logger.info("Top of the pdms : {} [um]".format(loc_top_pdms))

    #self.pml_layers = [mp.PML(thickness = self.pml_thickness, direction = mp.Z)]

    material_index_fused_silica = geometry_params['material_index_fused_silica']
    material_index_pdms = geometry_params['material_index_pdms']

    substrate_params = {
            'size_x_fused_silica': mp.inf,
            'size_y_fused_silica': mp.inf,
            'size_z_fused_silica': size_z_fused_silica,
            'loc_x_fused_silica': size_x_cell/2,
            'loc_y_fused_silica': size_y_cell/2,
            'loc_z_fused_silica': loc_z_fused_silica,
            'material_index_fused_silica': material_index_fused_silica,
            'size_x_pdms': mp.inf,
            'size_y_pdms': mp.inf,
            'size_z_pdms': size_z_pdms,
            'loc_x_pdms': size_x_cell/2,
            'loc_y_pdms': size_y_cell/2,
            'loc_z_pdms': loc_z_pdms,
            'material_index_pdms': material_index_pdms,
            }

    params['substrate_params'] = substrate_params

    substrate = build_silica_pdms_substrate(substrate_params)

    metasurface = [i for i in substrate]

    #Now for the pillars
    material_index_pillars = geometry_params['material_index_meta_atom']

    radii = [0.2 for _ in range(0,Nx*Ny)]
    logger.info("Radii of the pillars : {}".format(radii))
    count = 0
    for nx in range(0,Nx):
        for ny in range(0,Ny):
            loc_x_pillar = round((unit_cell_size * nx) + 0.5 * unit_cell_size, 3)
            loc_y_pillar = round((unit_cell_size * ny) + 0.5 * unit_cell_size,3)
            metasurface.append(build_cylinder(loc = [loc_x_pillar, loc_y_pillar, loc_z_pillar],
                                              axis = [0,0,1],
                                              height = height_pillar,
                                              radius = radii[count],
                                              material_index = material_index_pillars))

            count += 1

    #Now for the pml layers
    pml_layers = [mp.PML(thickness = thickness_pml, direction = mp.Z)]
    
    return metasurface, pml_layers

if __name__ == "__main__":
    import yaml
    params = yaml.load(open('config.yaml'), Loader = yaml.FullLoader)
    metasurface, pml = build_andy_metasurface_neighborhood(params)
    from IPython import embed; embed()



