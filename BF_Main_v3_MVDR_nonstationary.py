#BF Main

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, spectrogram
from pathlib import Path


# Setup ########################################
fs = 48e3 #Hz
# nfft = 50000
nfft = 20000
hop = round(nfft/2)
max_freq = 343
################################################

print(fs/nfft)

folder_path = r"C:\Users\Danut\OneDrive\Documents\ELEC5305-Project-Jdul9044-main\Simulation\Azimuth 60, Elevation 0"
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
# selected_window=("kaiser", 2)
selected_window='hann'

# Compute Spectrogram for all channels
spectrogram_list = []
    
for i in range(num_mic):
    
    freq_bins, time_bins, X = stft(
    signal_array[i, :],
    fs=fs,
    window=selected_window,
    nperseg=nfft,
    noverlap=nfft - hop,
    nfft=nfft,
    return_onesided=True,
    boundary=None,
    padded=False
)
    
    spectrogram_list.append(X)
    
spectrogram_array = np.array(spectrogram_list)
del spectrogram_list

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
plt.xlim(0,10)
plt.ylim(100,300)
plt.show()

#Check Graph - for single microphone/hydrophone
spectrogram_avg = np.mean(np.abs(spectrogram_array[selected_mic,:,:]),axis=1)

plt.plot(freq_bins, 20*np.log10(np.abs(spectrogram_avg)))
plt.xlim(100,300)
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
azi_deg = 60 #deg
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
        
# MVDR NONSTATIONARY
window_len = 20
# import numba

for ff in range(len(freq_bins)):
    steering_vect = np.exp(-1j * 2 * np.pi * freq_bins[ff] * tau)
    steering_vect = steering_vect.reshape(-1, 1)

    for tt in range(len(time_bins)):
        start_idx = max(0, tt - window_len)
        end_idx = min(len(time_bins), tt + window_len)
        
        X_f = spectrogram_array[:, ff, start_idx:end_idx]
        R = (X_f @ X_f.conj().T) / X_f.shape[1]
        Rinv = np.linalg.pinv(R)
        
        #short version
        # Y[ff, tt] = 1 / (np.dot(np.conj(steering_vect).T, Rinv, steering_vect))
        Y[ff, tt] = 1/(steering_vect.conj().T @ Rinv @ steering_vect).squeeze()
        
        # long version
        # w = (Rinv @ steering_vect)/(steering_vect.conj().T @ Rinv @ steering_vect)                
        # Y[ff, tt] = np.dot(np.conj(w).flatten(), spectrogram_array[:, ff, tt])
        
# Y_dB = 20 * np.log10(np.abs(Y)) change this for long version
Y_dB = np.log10(np.abs(Y)) # use for short version

    
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
plt.title(str(window_len))

plt.colorbar()

# plt.ylim(0,400)
plt.xlim(0,10)
plt.ylim(100,300)
plt.show()

#Check Graph - Beamformer Output
spectrogram_avg_bf = np.mean(np.abs(Y),axis=1)

plt.plot(freq_bins, 20*np.log10(np.abs(spectrogram_avg_bf)))
plt.xlim(100,300)
# plt.xlim(0,1000)
plt.show()





