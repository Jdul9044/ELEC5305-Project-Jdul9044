#BF Main

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import numpy as np
from pathlib import Path


# Setup ########################################
fs = 48e3 #Hz
nfft = 70000
hop = nfft/2
max_freq = 343
################################################

print(fs/nfft)

folder_path = r"C:\Users\John Conrad\Desktop\ELEC5305\Linear - 0.5m Spacing\Azimuth 60, Elevation 0"
folder = Path(folder_path)

# pcm_files = list(folder.glob("*.pcm"))
signal_list = []

for file in folder.glob("*.pcm"):
    signal = np.fromfile(file, dtype=np.int16)
    signal_list.append(signal)

signal_array = np.array(signal_list)
del signal_list

num_mic = signal_array.shape[0]

#Check clip
selected_window=("kaiser", 2)
# selected_window='hann'


# Compute Spectrogram for all channels
spectrogram_list = []

for i in range(num_mic):
    freq_bins, time_bins, X = stft(
        signal_array[i,:],
        fs=fs,
        window=selected_window,
        nperseg=nfft,
        noverlap=nfft - hop,
        axis=0,
        return_onesided=True,
        boundary="zeros"
    )
    
    spectrogram_list.append(X)
    
spectrogram_array = np.array(spectrogram_list)
del spectrogram_list

test = 20*np.log10(np.abs(spectrogram_array))


#Check Single Channel
selected_mic = 3


#Check Single Channel
plt.imshow(
    20*np.log10(np.abs(spectrogram_array[selected_mic,:,:])),
    aspect='auto',
    origin='lower',
    vmin = np.percentile(20*np.log10(np.abs(spectrogram_array[selected_mic,:,:])),5),
    vmax = np.percentile(20*np.log10(np.abs(spectrogram_array[selected_mic,:,:])),95),
    extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]]
)

plt.xlabel("Elapsed Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar()

# plt.ylim(0,400)
plt.ylim(100,300)
plt.show()

# Beamforming Stage

# Array Setup ##################################
c = 343 #m/s
mic_num = 8
spacing = 0.5 #m
mic_num = 8 # 8 channels
mic_pos = np.zeros((mic_num,3)) # 3D position information of the microphone array

# Array Type
mic_pos[:,0] = np.arange(mic_num) * spacing # for linear array only
################################################



# Target Information ###########################
azi_deg = 300 #deg
ele_deg = 0 #deg
################################################
azi = np.deg2rad(azi_deg) #rad
ele = np.deg2rad(ele_deg) #rad

# Unit vector
u = np.array([
    np.cos(ele) * np.cos(azi),
    np.cos(ele) * np.sin(azi),
    np.sin(ele)
    ])

# Time Delay
tau = np.dot(mic_pos,u) / c

# Beamforming
Y = np.zeros((len(freq_bins),len(time_bins)), dtype=np.complex128)

for tt in range(len(time_bins)):
    for ff in range(len(freq_bins)):
        steering_vect = np.exp(-1j * 2 * np.pi* freq_bins[ff] * tau)
        Y[ff,tt] = np.dot(np.conj(steering_vect), spectrogram_array[:,ff,tt]) / mic_num
        
Y_dB = 20 * np.log10(np.abs(Y))
    
#Check Beamformer Output
plt.imshow(
    20*np.log10(np.abs(Y[:,:])),
    aspect='auto',
    origin='lower',
    vmin = np.percentile(20*np.log10(np.abs(Y[:,:])),5),
    vmax = np.percentile(20*np.log10(np.abs(Y[:,:])),95),
    extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]]
)

plt.xlabel("Elapsed Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar()

# plt.ylim(0,400)
plt.ylim(100,300)
plt.show()





