###################
# Standard imports
###################
import meep as mp
from loguru import logger
import random


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

def build_silica_air_substrate(params:dict) -> list:

    logger.info("Building silica + air substrate.")
    ##################################
    #     _____________________
    #    |         AIR         |
    #    |_____________________|
    #    |    FUSED SILICA     |
    #    |_____________________|
    #
    ##################################

    logger.info("Reading fused silica parameters")

    size_x_buffer = params['size_x_buffer']
    size_y_buffer = params['size_y_buffer']
    size_z_buffer = params['size_z_buffer']

    offset_x_buffer = params['offset_x_buffer']
    offset_y_buffer = params['offset_y_buffer']
    offset_z_buffer = params['offset_z_buffer']
    
    size_x_fused_silica = params['size_x_fused_silica'] + size_x_buffer
    size_y_fused_silica = params['size_y_fused_silica'] + size_y_buffer
    size_z_fused_silica = params['size_z_fused_silica']
    
    loc_x_fused_silica = params['loc_x_fused_silica'] + round(size_x_buffer / 2, 4)
    loc_y_fused_silica = params['loc_y_fused_silica'] + round(size_y_buffer / 2, 4)
    loc_z_fused_silica = params['loc_z_fused_silica']
    material_index_fused_silica = params['material_index_fused_silica']

    logger.info("Creating fused silica material. Index = {}".format(material_index_fused_silica))

    fused_silica = build_block( size = [size_x_fused_silica, size_y_fused_silica, size_z_fused_silica],
                                loc = [loc_x_fused_silica, loc_y_fused_silica, loc_z_fused_silica],
                                material_index = material_index_fused_silica)
    logger.info("Fused silica size: {}".format(fused_silica.size))
    logger.info("Fused silica center: {}".format(fused_silica.center))

    logger.info("Reading air parameters")
    loc_x_air = params['loc_x_air'] + offset_x_buffer
    loc_y_air = params['loc_y_air'] + offset_y_buffer
    loc_z_air = params['loc_z_air']
    
    size_x_air = params['size_x_air'] + size_x_buffer
    size_y_air = params['size_y_air'] + size_y_buffer
    size_z_air = params['size_z_air']

    material_index_air = params['material_index_air']

    logger.info("Creating fused silica material. Index = {}".format(material_index_fused_silica))

    air = build_block( size = [size_x_air, size_y_air, size_z_air],
                        loc = [loc_x_air, loc_y_air, loc_z_air],
                        material_index = material_index_air )

    logger.info("air size : {}".format(air.size))
    logger.info("air loc : {}".format(air.center))
    
    return [fused_silica, air]

def build_andy_metasurface_neighborhood(params, radii = None):
    '''
    This is basically the same code as the parameter manager's calculate_dependencies
    from the surrogate model code. Just with additional comments and different
    organization.
    Builds a fused silica + air substrate with a Nx x Ny pillar neighborhood.
    '''
    geometry_params = params['geometry']
    Nx, Ny = geometry_params['neighborhood_size']
    logger.info("Creating metasurface neighborhood. Nx,Ny : {},{}".format(Nx,Ny))

    atom_type = geometry_params['atom_type']
    logger.info("Meta-atom type: {}".format(atom_type))

    unit_cell_size = geometry_params['unit_cell_size']
    logger.info("Meta-atom size: {}".format(unit_cell_size))

    if geometry_params['substrate_buffer']:
        logger.info("Buffer is enabled")
        size_x_buffer = geometry_params['size_x_buffer']
        size_y_buffer = geometry_params['size_y_buffer']
        size_z_buffer = geometry_params['size_z_buffer']
        logger.info("Buffer : {}, {}, {}".format(size_x_buffer, size_y_buffer, size_z_buffer))

        offset_x_buffer = round(size_x_buffer / 2, 4)
        offset_y_buffer = round(size_y_buffer / 2, 4)
        offset_z_buffer = round(size_z_buffer / 2, 4)
        logger.info("Center offset added by buffer : {}, {}, {}".format(offset_x_buffer, offset_y_buffer, offset_z_buffer))
        params['geometry']['offset_x_buffer'] = offset_x_buffer
        params['geometry']['offset_y_buffer'] = offset_y_buffer
        params['geometry']['offset_z_buffer'] = offset_z_buffer
    else:
        logger.info("Buffer is disabled")
        size_x_buffer = 0
        size_y_buffer = 0
        size_z_buffer = 0
        offset_x_buffer = 0
        offset_y_buffer = 0
        offset_z_buffer = 0
    
    #Define the z stack size
    thickness_Abs = geometry_params['thickness_Abs']
    height_pillar = geometry_params['height_pillar']
    size_z_air = round(geometry_params['size_z_air'] + height_pillar + thickness_Abs + size_z_buffer, 4)
    size_z_fused_silica = geometry_params['size_z_fused_silica'] + thickness_Abs
    params['geometry']['size_z_air'] = size_z_air

    #Add all of the z sizes together to get the total z size
    #I added the Abs first above, so they are not included here.
    size_z_cell = round(size_z_fused_silica + size_z_air, 4)
    #Multiply the unit cell size by the numer of unit cells to get the x and y sizes
    size_x_cell = round(unit_cell_size * Nx, 4) + size_x_buffer
    size_y_cell = round(unit_cell_size * Ny, 4) + size_y_buffer

    params['cell_x'] = size_x_cell
    params['cell_y'] = size_y_cell
    params['cell_z'] = size_z_cell

    logger.info("Size of total geometry cell : {} x {} x {} [um]".format(size_x_cell, size_y_cell, size_z_cell))
    cell_size = mp.Vector3(size_x_cell, size_y_cell, size_z_cell)
    params['geometry']['cell_size'] = cell_size

    #Get the z locations (centers)  of all of the geometry objects
    #Here, things can be a little confusing. We need to subtract out 1/2 the
    #total cell size to keep things centered on (0,0)



    loc_z_fused_silica = round(0.5 * size_z_fused_silica - 0.5 * size_z_cell,4)
    logger.info("Center Z of fused silica : {} [um]".format(loc_z_fused_silica))
    loc_z_air = round(size_z_fused_silica + 0.5 * size_z_air - 0.5 * size_z_cell, 4)
    logger.info("Center Z of air : {} [um]".format(loc_z_air))
    loc_z_pillar = round(size_z_fused_silica + 0.5 * height_pillar - 0.5 * size_z_cell, 4)

    logger.info("Center Z of pillars : {} [um]".format(loc_z_pillar))

    params['geometry']['loc_z_fused_silica'] = loc_z_fused_silica
    params['geometry']['loc_z_air'] = loc_z_air
    params['geometry']['loc_z_pillar'] = loc_z_pillar

    loc_top_fused_silica = round(size_z_fused_silica - 0.5 * size_z_cell, 4)
    logger.info("Top of the fused silica : {} [um]".format(loc_top_fused_silica))

    #The top of the air - the whole size minus the amount in the Abs
    loc_top_air = round(size_z_fused_silica + (size_z_air - thickness_Abs) - 0.5 * size_z_cell, 4)
    logger.info("Top of the air : {} [um]".format(loc_top_air))

    logger.info("Updating params with calculated locations")
    params['geometry']['loc_top_fused_silica'] = loc_top_fused_silica
    params['geometry']['loc_top_air'] = loc_top_air

    #Get the center of the simulation cell
    loc_z_center_cell = 0
    loc_x_center_cell = 0
    loc_y_center_cell = 0
    center_sim_cell = mp.Vector3(loc_x_center_cell, loc_y_center_cell, loc_z_center_cell)
    logger.info("Center of the simulation cell : {}".format(center_sim_cell))
    params['geometry']['center_sim_cell'] = center_sim_cell

    #Get the size of the non Abs region
    size_z_non_Abs = size_z_fused_silica + size_z_air - 2*thickness_Abs
    logger.info("Size of the non Abs volume : {}".format(size_z_non_Abs))
    params['geometry']['size_z_non_Abs'] = size_z_non_Abs


    material_index_fused_silica = geometry_params['material_index_fused_silica']
    material_index_air = geometry_params['material_index_air']

    substrate_params = {
            'size_x_fused_silica': mp.inf,
            'size_y_fused_silica': mp.inf,
            'size_z_fused_silica': size_z_fused_silica,
            'loc_x_fused_silica': 0,
            'loc_y_fused_silica': 0,
            'loc_z_fused_silica': loc_z_fused_silica,
            'material_index_fused_silica': material_index_fused_silica,
            'size_x_air': mp.inf,
            'size_y_air': mp.inf,
            'size_z_air': size_z_air,
            'loc_x_air': 0,
            'loc_y_air': 0,
            'loc_z_air': loc_z_air,
            'material_index_air': material_index_air,
            'substrate_buffer': geometry_params['substrate_buffer'],
            'size_x_buffer': size_x_buffer,
            'size_y_buffer': size_y_buffer,
            'size_z_buffer': size_z_buffer,
            'offset_x_buffer': offset_x_buffer,
            'offset_y_buffer': offset_y_buffer,
            'offset_z_buffer': offset_z_buffer
            }

    params['substrate_params'] = substrate_params

    substrate = build_silica_air_substrate(substrate_params)

    metasurface = [i for i in substrate]

    #Now for the pillars
    material_index_pillars = geometry_params['material_index_meta_atom']
    random_pil = params['geometry']['random_pil']
    radius_min = params['geometry']['radius_min']
    radius_max = params['geometry']['radius_max']
    seed = params['seed']

    if radii == None:
        if random_pil == False:
            radius = params['geometry']['radius_pillar']                            #####ADDED IN TO FACILITATE UNIFORM PILLAR RADIUS NEIGHBORHOOD
            radii = [radius for _ in range(0,Nx*Ny)]                              
        else:
            radii = []
            random.seed(seed)
            for i in range(0,Nx*Ny):
                num = random.uniform(radius_min, radius_max)
                radii.append(num)

    logger.info("Radii of the pillars : {}".format(radii))
    count = 0

    if(atom_type == 'cylinder'):
        for ny in range(0,Ny):
            for nx in range(0,Nx):
                loc_x_pillar = round((unit_cell_size * nx) + 0.5 * unit_cell_size - 0.5 * size_x_cell, 4) + offset_x_buffer
                loc_y_pillar = round((unit_cell_size * ny) + 0.5 * unit_cell_size - 0.5 * size_y_cell, 4) + offset_y_buffer
                params['geometry']['loc_x_pillar_{}'.format(count)] = loc_x_pillar
                params['geometry']['loc_y_pillar_{}'.format(count)] = loc_y_pillar
                metasurface.append(build_cylinder(loc = mp.Vector3(loc_x_pillar, loc_y_pillar, loc_z_pillar),
                                                axis = mp.Vector3(0,0,1),
                                                height = height_pillar,
                                                radius = radii[count],
                                                material_index = material_index_pillars))

                count += 1

    else:
        for ny in range(0,Ny):
            for nx in range(0,Nx):
                loc_x_pillar = round((unit_cell_size * nx) + 0.5 * unit_cell_size - 0.5 * size_x_cell, 4) + offset_x_buffer
                loc_y_pillar = round((unit_cell_size * ny) + 0.5 * unit_cell_size - 0.5 * size_y_cell, 4) + offset_y_buffer
                params['geometry']['loc_x_pillar_{}'.format(count)] = loc_x_pillar
                params['geometry']['loc_y_pillar_{}'.format(count)] = loc_y_pillar
                metasurface.append(build_block( size = mp.Vector3(radii[count]*2, radii[count]*2, height_pillar),
                                                loc = mp.Vector3(loc_x_pillar, loc_y_pillar, loc_z_pillar),
                                                material_index = material_index_pillars))

                count += 1

    #Now for the Abs layers
    Abs_layers = [mp.Absorber(thickness = thickness_Abs, direction = mp.Z)]

    params['geometry']['Abs_layers'] = Abs_layers
    #Abs_layers = []

    #Get the volume not in the Abs for the monitors
    monitor_volume = mp.Volume(center = center_sim_cell,
                               size = mp.Vector3(size_x_cell, size_y_cell, size_z_non_Abs))
    
    params['geometry']['monitor_volume'] = monitor_volume
    return metasurface, Abs_layers, monitor_volume

if __name__ == "__main__":
    import yaml
    params = yaml.load(open('config.yaml'), Loader = yaml.FullLoader)
    metasurface, Abs, monitor_volume = build_andy_metasurface_neighborhood(params)

