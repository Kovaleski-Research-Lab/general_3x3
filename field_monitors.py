from numpy import diag_indices_from
import meep as mp
import yaml
from loguru import logger
import geometries, sources, boundaries, simulation


def collect_fields(params, sim, flux_obj = None, dft_obj = None):

    monitor_params = params['monitor']
    wavelength_list = monitor_params['wavelength_list']

    if flux_obj == None:
        flux = None
    else:
        flux = mp.get_fluxes(flux_obj)[0]

    if dft_obj == None:
        dft_fields = None
    else:
        dft_fields = {}
        for i, wl in enumerate(wavelength_list):
            dft_fields[wl] = [sim.get_dft_array(dft_obj, mp.Ex, i),
                                sim.get_dft_array(dft_obj, mp.Ey, i),
                                sim.get_dft_array(dft_obj, mp.Ez, i)]

    eps_data = sim.get_epsilon()
    return dft_fields, flux, eps_data

def build_dft_monitor(params, sim, monitor_volume):

    monitor_params = params['monitor']
    components = monitor_params['components_dft_monitor']
    if components == 'all':
        components = [mp.Ex, mp.Ey, mp.Ez]
    else:
        logger.error("Monitor component {} not supported".format(components))

    freq_list = monitor_params['freq_list']
    wavelength_list = monitor_params['wavelength_list']

    if freq_list is None and wavelength_list is None:
        logger.error("You need to specify either a frequency list or a wavelength list for the DFT monitor")
        exit()

    if freq_list is None:
        freq_list = [1/wl for wl in wavelength_list]
        monitor_params['freq_list'] = freq_list
    
    where = monitor_volume
    dft_obj = sim.add_dft_fields(components, freq_list, where = where)
    return dft_obj

def build_timedep_monitor(params, sim):

    monitor_params = params['monitor']

    #Going to make the field monitor at 775nm above the pillars
    loc_top_pdms = params['geometry']['loc_top_pdms']
    height_pillar = params['geometry']['height_pillar']

    loc_z_timedep_mon = loc_top_pdms + height_pillar + 0.775
    mon_pt = mp.Vector3(0,0,loc_z_timedep_mon)
    size = mp.Vector3(params['cell_x'], params['cell_y'], 0)
    flux_region = mp.FluxRegion(center = mon_pt, size = size)

    freq_list = monitor_params['freq_list']
    wavelength_list = monitor_params['wavelength_list']
    if freq_list is None and wavelength_list is None:
        logger.error("You need to specify either a frequency list or a wavelength list for the DFT monitor")
        exit()

    if freq_list is None:
        freq_list = [1/wl for wl in wavelength_list]
        monitor_params['freq_list'] = freq_list

    flux_obj = sim.add_flux(freq_list, flux_region)
    return flux_obj

if __name__ == "__main__":

    params = yaml.load(open("config.yaml"), Loader = yaml.FullLoader)
    geometry, pml_layer, monitor_volume = geometries.build_andy_metasurface_neighborhood(params)
    source = sources.build_andy_source(params)
    sim = simulation.build_sim(params)
    dft_obj = build_dft_monitor(params, sim, monitor_volume)
    flux_obj = build_timedep_monitor(params, sim)


