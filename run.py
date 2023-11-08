
import yaml
import pickle
import meep as mp
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt


import simulation
import field_monitors

if __name__ == "__main__":

    params = yaml.load(open("config.yaml", 'r'), Loader = yaml.FullLoader)
    
    #6 7 8
    #3 4 5
    #0 1 2

    #radii = [0.18664, 0.09511, 0.13333,
    #         0.16552, 0.19670, 0.13635,
    #         0.20876, 0.10517, 0.09009]

    radii = [0.20876, 0.10517, 0.09009,0.16552, 0.19670, 0.13635, 0.18664, 0.09511, 0.13333]
    
    sim, dft_obj, flux_obj = simulation.build_sim(params, radii = radii)
    sim.run(until=200)
    #from IPython import embed; embed()
    dft_fields, flux, eps_data = field_monitors.collect_fields(params, sim, flux_obj, dft_obj)
    sim.dump_structure('sim_test.meep')
    pickle.dump([dft_fields, flux, eps_data],open('../../results/gaussian_random.pkl', 'wb'))

