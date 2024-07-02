import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 1
center_wavelength = 1.550
center_frequency = c / center_wavelength
bandwidth_factor = 1.2
bandwidth_frequency = bandwidth_factor * center_frequency

# Calculate standard deviation
bandwidth_wavelength = c / bandwidth_frequency
sigma = bandwidth_wavelength / (2 * np.sqrt(2 * np.log(2)))

# Create a time array (assuming Gaussian pulse centered at t=0)
time = np.linspace(-5 * sigma, 5 * sigma, 1000)

# Gaussian function
gaussian_pulse = np.exp(-(time / sigma)**2 / 2)

# Create a frequency array
frequency = np.fft.fftfreq(len(time), time[1] - time[0])
frequency = np.fft.fftshift(frequency)  # Shift zero frequency to the center

# Fourier transform of the Gaussian pulse
gaussian_frequency = np.fft.fft(gaussian_pulse)
gaussian_frequency = np.fft.fftshift(gaussian_frequency)  # Shift zero frequency to the center

# Set the figure size
plt.figure(figsize=(12, 4))

# Create the first subplot for the Gaussian pulse in terms of wavelength
plt.subplot(1, 3, 1)
plt.plot(time, gaussian_pulse, label=f'Center Wavelength: {center_wavelength:.3f} \nBandwidth: {bandwidth_wavelength:.3f}')
plt.title('Gaussian Pulse in Wavelength Domain')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(loc='upper left')

# Create the second subplot for the Gaussian pulse in terms of frequency
plt.subplot(1, 3, 2)
plt.plot(frequency, np.abs(gaussian_frequency), label=f'Center Frequency: {center_frequency:.3f} \nBandwidth: {bandwidth_frequency:.3f}')
plt.title('Gaussian Pulse in Frequency Domain')
plt.xlabel('Frequency')
plt.xlim([-15,15])
plt.ylabel('Amplitude')
plt.legend(loc='upper left')

colors = ['grey','orange','blue','green','purple']
# Create the third subplot for frequency vs. wavelength
wavelengths = np.linspace(1, 3, 100)
frequencies = c / wavelengths
plt.subplot(1, 3, 3)
plt.plot(wavelengths, frequencies, label=r'f vs. $\lambda$')
plt.scatter([1.06],[c / 1.06],label = '1060',color=colors[0])
plt.scatter([1.30],[c / 1.30],label = '1300',color=colors[0])
plt.scatter([1.55],[c / 1.55],label = '1550',color=colors[2])
plt.scatter([1.65],[c / 1.65],label = '1650',color=colors[0])
plt.scatter([2.881],[c / 2.881],label = '2881',color=colors[0])
plt.axvline(x=1.06, linestyle='--',color=colors[0], alpha=0.4)
plt.axhline(y=c/1.06, linestyle='--',color=colors[0],alpha=0.4)
plt.axvline(x=1.30, linestyle='--',color=colors[0],alpha=0.4)
plt.axhline(y=c/1.30, linestyle='--',color=colors[0],alpha=0.4)
plt.axvline(x=1.55, linestyle='--',color=colors[2],alpha=0.4)
plt.axhline(y=c/1.55, linestyle='--',color=colors[2],alpha=0.4)
plt.axvline(x=1.65, linestyle='--',color=colors[0],alpha=0.4)
plt.axhline(y=c/1.65, linestyle='--',color=colors[0],alpha=0.4)
plt.axvline(x=2.881, linestyle='--',color=colors[0],alpha=0.4)
plt.axhline(y=c/2.881, linestyle='--',color=colors[0],alpha=0.4)

plt.title('Frequency vs. Wavelength')
plt.xlabel('Wavelength')
plt.ylabel('Frequency')
plt.ylim([0.2,1.1])
plt.legend(loc='upper right')
# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()

