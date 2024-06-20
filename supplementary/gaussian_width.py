import numpy as np
import matplotlib.pyplot as plt
from pint import UnitRegistry
from scipy.constants import c

ureg = UnitRegistry()
c = c * ureg.meters / ureg.seconds
c = c.to(ureg.micrometers / ureg.femtoseconds)
a = 1 * ureg.micrometer

def get_colors(num_colors):

    cmap_viridis = plt.cm.get_cmap('viridis')
    colors = [cmap_viridis(i / num_colors) for i in range(num_colors)]

    return colors

def get_phys_freq(meep_freq):

    freq_hz = (meep_freq * c) / a 
    
    return freq_hz.to('terahertz') 


if __name__=="__main__":

    wl_list = [2.881, 1.650, 1.550, 1.300, 1.060]
    wl_list.sort()
    freq_list = [1 / x for x in wl_list]
    wl_cen = 1.550
    thz_list = get_phys_freq(freq_list)

    fcen = 1 / wl_cen
    fmax = 1 / min(wl_list)
    fmin = 1 / max(wl_list)
    fwidth = fmax - fmin
    
    colors = get_colors(len(wl_list))
    
    plt.style.use("ggplot")
    
    fig, ax = plt.subplots(figsize=(16,8))
    
    ax.plot(wl_list, freq_list, c='black')
    
    for wl, freq, thz, col in zip(wl_list, freq_list, thz_list, colors):
    
        ax.plot(wl, freq, 'o', label=f'{wl:.3f} um', markersize=10, color = col)
        ax.annotate(f'{thz:.2f}', (wl, freq), textcoords='offset points', xytext=(5,5), ha='left')
    
    
    ax.set_xticks(wl_list)
    ax.set_xticklabels([f'{wl:.3f}' for wl in wl_list])
    
    ax.set_yticks(freq_list)
    ax.set_yticklabels([f'{freq:.3f}' for freq in freq_list])
    
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('Meep Frequency')
    ax.set_title(f'fwidth = {fwidth}')
    
    ax.legend(title='wl')
    
    plt.show()
