import yaml
import meep as mp

import geometries, sources, boundaries, field_monitors


def build_sim(params):

    geometry, pml_layers, monitor_volume = geometries.build_andy_metasurface_neighborhood(params)
    source = sources.build_andy_source(params)
    k_point = mp.Vector3(0,0,0)

    size_x_cell = params['cell_x']
    size_y_cell = params['cell_y']
    size_z_cell = params['cell_z']

    size_cell = mp.Vector3(size_x_cell, size_y_cell, size_z_cell)

    center_x_cell = round(size_x_cell / 2, 3)
    center_y_cell = round(size_y_cell / 2, 3)
    center_z_cell = round(size_z_cell / 2, 3)
    center_cell = mp.Vector3(center_x_cell, center_y_cell, center_z_cell)
    resolution = int(params['simulation']['resolution'])

    sim = mp.Simulation( 
                         geometry_center = center_cell,
                         cell_size = size_cell,
                         geometry = geometry,
                         sources = source,
                         k_point = k_point,
                         boundary_layers = pml_layers,
                         resolution = resolution,
                         symmetries=None)
    
    field_monitors.build_dft_monitor(params, sim, monitor_volume)

    return sim


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    params = yaml.load(open("config.yaml"), Loader = yaml.FullLoader)
    params_simulation = params['simulation']
    sim = build_sim(params)


    center_mon_z = round(params['cell_z'] / 2, 3)
    source_component = mp.Ey

    cell_x = params['cell_x']
    cell_y = params['cell_y']
    cell_z = params['cell_z']

    center_x = round(cell_x / 2, 3)
    center_y = round(cell_y / 2, 3)
    center_z = round(cell_z / 2, 3)

    plot_plane = mp.Volume( center = mp.Vector3(center_x, center_y, center_z), 
                            size=mp.Vector3(cell_x, 0, cell_z))


    #sim.run(until_after_sources = mp.stop_when_fields_decayed(  dt = params_simulation['dt'],
    #                                                            c = params_simulation['source_component'],
    #                                                            pt = mp.Vector3(0, 0, center_mon_z),
    #                                                            decay_by = params_simulation['decay_rate']))
    
    sim.run(until=200)
    fig,ax = plt.subplots(1,1,figsize=(15,15))
    sim.plot2D(output_plane = plot_plane, ax=ax)
    plt.show()
    plt.savefig('test.png')
