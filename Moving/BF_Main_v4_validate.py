#BF Main

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import stft, istft, spectrogram
from pathlib import Path
from scipy.signal import welch, decimate
from scipy.io import wavfile, loadmat
import pandas as pd
from arlpy import bf
# Setup ########################################
fs = 48e3 #Hz
nfft = 1028
hop = round(nfft/2)
################################################
print("before downsampling",fs/nfft)

folder_path = r"C:\Users\John Conrad\Desktop\ELEC5305\Moving"
folder = Path(folder_path)

# pcm_files = list(folder.glob("*.pcm"))
signal_list = []

for file in folder.glob("*.pcm"):
    signal = np.fromfile(file, dtype=np.int16)
    
    # downsample
    signal = decimate(signal, int(fs/2000))
    signal_list.append(signal)
    
fs = 2000
print("after downsampling",fs/nfft)
signal_array = np.array(signal_list)
signal_array = signal_array / np.max(np.abs(signal_array))

mic_num = 8
spacing = 1.875 #m [c/(2d) = max freq -> 343 with 0.5m is 343 Hz]
mic_pos = np.zeros((mic_num,3))
mic_pos[:, 1] = np.arange(mic_num) * spacing
################################################################################################

#Check clip
mic_num = signal_array.shape[0]
# selected_window=("kaiser", 2)
selected_window='hann'

# Compute Spectrogram for all channels
spectrogram_list = []
    
# for i in range(num_mic):
for i in range(mic_num):
    
    freq_bins, time_bins, X = stft(
    signal_array[i, :],
    fs=fs,
    window=selected_window,
    nperseg=nfft,
    noverlap=nfft - hop,
    nfft=nfft,
    return_onesided=True,
    boundary=None,
    padded=False,
    scaling='psd'
)
    
    spectrogram_list.append(X)
    
spectrogram_array = np.array(spectrogram_list)
del spectrogram_list

spectrogram_psd = np.abs(spectrogram_array)**2
spectrogram_psd[:,1:-1,:] *= 2

#Check Single Channel
selected_mic = 3

#Check Graph - for single microphone/hydrophone
f, Pxx = welch(signal_array[selected_mic,:], fs=fs, window=selected_window,
               nperseg=nfft, noverlap=nfft-hop, nfft=nfft, detrend=False)

# Beamforming Stage

mic_pos = mic_pos #- np.mean(mic_pos, axis=0)

# Array Setup ##################################
c = 1500 #m/s
################################################
# Target Information ###########################
azi_start = 60
azi_end = 90
azi_deg = np.linspace(azi_start,azi_end,len(time_bins))
ele_deg = np.zeros(len(time_bins)) #deg
# azi_deg = -70 #deg
# ele_deg = 0 #deg

# azi_deg = -70 * np.ones(len(time_bins)) #deg
# ele_deg = 0 * np.ones(len(time_bins))#deg
################################################
azi = np.deg2rad(azi_deg) #rad
ele = np.deg2rad(ele_deg) #rad
################################################

# Unit vector
u = np.array([
    np.cos(ele) * np.sin(azi),
    np.cos(ele) * np.cos(azi),
    np.sin(ele)
    ])

# Time Delay
tau = np.dot(mic_pos,u) / c

# Beamforming
Y = np.zeros((len(freq_bins),len(time_bins)), dtype=np.complex128)
Y_mvdr = np.zeros((len(freq_bins),len(time_bins)), dtype=np.complex128)

# MVDR NONSTATIONARY
window_len = 20

# MVDR
for ff in range(len(freq_bins)):

    for tt in range(len(time_bins)):
        start_idx = max(0, tt - window_len)
        end_idx = min(len(time_bins), tt + window_len)
        
        steering_vect = np.exp(-1j * 2 * np.pi * freq_bins[ff] * tau[:,tt])
        steering_vect = steering_vect.reshape(-1, 1)
        
        X_f = spectrogram_array[:, ff, start_idx:end_idx]
        R = (X_f @ X_f.conj().T) / X_f.shape[1]
        
        loading = 0.1* np.trace(R).real / mic_num
        # loading = 0.1* np.trace(R).real / mic_num
        R_loaded = R + loading * np.eye(mic_num)
        Rinv = np.linalg.inv(R_loaded)
        
        w = (Rinv @ steering_vect)/(steering_vect.conj().T @ Rinv @ steering_vect)                            
        Y_mvdr[ff, tt] = np.dot(np.conj(w).flatten(), spectrogram_array[:, ff, tt])
        

plt.imshow(
    np.abs(Rinv),
    aspect='auto',
    origin='lower',
)

plt.title(r'$R^{-1}$ Diagonal Matrix' + f", Window Length={window_len+1}" + ", Loading=0.1")

# # CONVENTIONAL BEAMFORMER
for tt in range(len(time_bins)):
    for ff in range(len(freq_bins)):
        steering_vect = np.exp(-1j * 2 * np.pi* freq_bins[ff] * tau[:,tt])
        Y[ff,tt] = np.dot(np.conj(steering_vect), spectrogram_array[:,ff,tt]) / mic_num


Y_psd = np.abs(Y)**2
Y_psd[1:-1] *= 2
Y_dB = 10*np.log10(Y_psd + 1e-12)

Y_mvdr_psd = np.abs(Y_mvdr)**2
Y_mvdr_psd[1:-1] *= 2
Y_mvdr_dB = 10*np.log10(Y_mvdr_psd + 1e-12)

#Check Graph - Beamformer & MVDR Output
spectrogram_avg_bf = np.mean(np.abs(Y)**2, axis=1)
spectrogram_avg_bf[1:-1] *= 2 

spectrogram_mvdr_avg_bf = np.mean(np.abs(Y_mvdr)**2, axis=1)
spectrogram_mvdr_avg_bf[1:-1] *= 2 
###############################################################################
# GRAPH 1
plt.plot(f, 10*np.log10(Pxx), label="Channel " + str(selected_mic) + " Hydrophone")
plt.plot(freq_bins, 10*np.log10(spectrogram_avg_bf + 1e-12), label="Beamformer Output")
plt.plot(freq_bins, 10*np.log10(spectrogram_mvdr_avg_bf + 1e-12), label="MVDR Beamformer Output")
plt.xlim(0,400)
# plt.ylim(60,70)
plt.title("Beam Focused on Target 1(moving), NFFT=" + str(nfft) + ", Overlap=50%")
plt.ylabel('Time Averaged PSD (dBFS/Hz)')
plt.xlabel("Frequency (Hz)")
plt.legend()
plt.show()
###############################################################################
Y_validate = np.zeros((len(freq_bins),len(time_bins)), dtype=np.complex128)
Y_mvdr_validate = np.zeros((len(freq_bins),len(time_bins)), dtype=np.complex128)
sd = tau.T


for tt in range(len(time_bins)):
    for ff in range(len(freq_bins)):
        
        X = spectrogram_array[:, ff, tt:tt+1]
        tau_new = sd[tt,:].reshape(1,-1)
        
        Y_temp = bf.bartlett( X,freq_bins[ff],tau_new,complex_output=True)
        Y_validate[ff,tt] = np.asarray(Y_temp).item() / np.sqrt(mic_num)

spectrogram_avg_bf_validate = np.mean(np.abs(Y_validate)**2, axis=1)
spectrogram_avg_bf_validate[1:-1] *= 2 

for tt in range(len(time_bins)):
    for ff in range(len(freq_bins)):
        
        start_idx = max(0, tt - window_len)
        X = spectrogram_array[:, ff,start_idx:tt+1]
        tau_new = sd[tt,:].reshape(1,-1)
        
        Y_temp = bf.capon( X,freq_bins[ff],tau_new,complex_output=True)
        Y_mvdr_validate[ff,tt] = np.asarray(Y_temp).flatten()[-1] / np.sqrt(mic_num)

spectrogram_mvdr_avg_bf_validate = np.mean(np.abs(Y_validate)**2, axis=1)
spectrogram_mvdr_avg_bf_validate[1:-1] *= 2 

# GRAPH 1
plt.plot(f, 10*np.log10(Pxx), label="Channel " + str(selected_mic) + " Hydrophone")
plt.plot(freq_bins, 10*np.log10(spectrogram_avg_bf + 1e-12), label="Beamformer Output")
plt.plot(freq_bins, 10*np.log10(spectrogram_avg_bf_validate + 1e-12), label="Beamformer Output Validation")
plt.plot(freq_bins, 10*np.log10(spectrogram_mvdr_avg_bf + 1e-12), label="MVDR Beamformer Output")
plt.plot(freq_bins, 10*np.log10(spectrogram_mvdr_avg_bf + 1e-12), label="MVDR Beamformer Output Validation")
plt.xlim(0,400)
# plt.ylim(-19.35,-19.2)
# plt.ylim(60,70)
plt.title("Beam Focused on Target 1(moving), NFFT=" + str(nfft) + ", Overlap=50%")
plt.ylabel('Time Averaged PSD (dBFS/Hz)')
plt.xlabel("Frequency (Hz)")
plt.legend()
plt.show()

error_bf = 10*np.log10(spectrogram_avg_bf + 1e-12) - 10*np.log10(spectrogram_avg_bf_validate + 1e-12)
error_bf = np.mean(np.abs(error_bf))
print(error_bf,"dB")

error_bf_mvdr = 10*np.log10(spectrogram_mvdr_avg_bf + 1e-12) - 10*np.log10(spectrogram_mvdr_avg_bf_validate + 1e-12)
error_bf_mvdr = np.mean(np.abs(error_bf_mvdr))
print(error_bf_mvdr,"dB")
