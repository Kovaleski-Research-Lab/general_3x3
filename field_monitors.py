import meep as mp
import yaml
from loguru import logger
import geometries, sources, boundaries, simulation


#def add_dft_fields(cs, fcen, df, nfreq, freq, where=None, center=None, size=None, yee_grid=False, decimation_factor=0, persist=False):

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
    
    where = monitor_volume
    sim.add_dft_fields(components, freq_list, where = where)

def build_timedep_monitor(params, sim):
    pass

if __name__ == "__main__":

    params = yaml.load(open("config.yaml"), Loader = yaml.FullLoader)

    geometry, pml_layer, monitor_volume = geometries.build_andy_metasurface_neighborhood(params)
    source = sources.build_andy_source(params)
    sim = simulation.build_sim(params)
    build_dft_monitor(params, sim, monitor_volume)


