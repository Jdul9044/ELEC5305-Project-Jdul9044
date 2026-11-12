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
fs = 48e3 #sampling freq
nfft = 1028 # number of FFT 
hop = round(nfft/2) # 50% overlap
################################################
# folder_path = r"Simulation_Data\Moving" #load folder location
# folder = Path(folder_path)
folder = Path(__file__).parent

signal_list = [] # create list

for file in folder.glob("*.pcm"):
    print(file)
    signal = np.fromfile(file, dtype=np.int16) # load PCM
    
    # downsample
    signal = decimate(signal, int(fs/2000)) #downsample 
    signal_list.append(signal)
    
fs = 2000 # new downsampled frequency
signal_array = np.array(signal_list) # create array
signal_array = signal_array / np.max(np.abs(signal_array)) # convert to full scale

mic_num = 8 # number of sensors
spacing = 1.875  # spacing
mic_pos = np.zeros((mic_num,3)) # 3D array setup for array geometry
mic_pos[:, 1] = np.arange(mic_num) * spacing # create 1D array
################################################################################################
# selected_window=("kaiser", 2) # choose window
selected_window='hann'

# Compute Spectrogram for all channels
spectrogram_list = []
    
# for i in range(num_mic):
for i in range(mic_num): # create spectrogram for each sensor
    
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
    
    spectrogram_list.append(X) # put into list
    
spectrogram_array = np.array(spectrogram_list) # create array
del spectrogram_list

spectrogram_psd = np.abs(spectrogram_array)**2 # convert magnitude into power
spectrogram_psd[:,1:-1,:] *= 2 # into PSD

#Check Single Channel
selected_mic = 3

#Check Graph - for single microphone/hydrophone
f, Pxx = welch(signal_array[selected_mic,:], fs=fs, window=selected_window,
               nperseg=nfft, noverlap=nfft-hop, nfft=nfft, detrend=False)

# Beamforming Stage
# Array Setup ##################################
c = 1500 #m/s
################################################
# Target Information ###########################
azi_start = 60 # start track of target
azi_end = 90 # end track of target
azi_deg = np.linspace(azi_start,azi_end,len(time_bins))
ele_deg = np.zeros(len(time_bins)) #deg
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
tau = -np.dot(mic_pos,u) / c

# Beamforming
Y = np.zeros((len(freq_bins),len(time_bins)), dtype=np.complex128)
Y_mvdr = np.zeros((len(freq_bins),len(time_bins)), dtype=np.complex128)

# MVDR NONSTATIONARY
window_len = 100 # select window size

# MVDR
for ff in range(len(freq_bins)):

    for tt in range(len(time_bins)):
        start_idx = max(0, tt - window_len) # find start index for moving window
        end_idx = min(len(time_bins), tt + window_len) # find end index for moving window
        
        steering_vect = np.exp(-1j * 2 * np.pi * freq_bins[ff] * tau[:,tt]) # steering vector
        steering_vect = steering_vect.reshape(-1, 1)
        
        X_f = spectrogram_array[:, ff, start_idx:end_idx] # select section of spectrogram for covariance matrix
        R = (X_f @ X_f.conj().T) / X_f.shape[1] # covariance matrix
        
        loading = 0.1* np.trace(R).real / mic_num # apply loading
        R_loaded = R + loading * np.eye(mic_num)
        Rinv = np.linalg.inv(R_loaded)
        
        w = (Rinv @ steering_vect)/(steering_vect.conj().T @ Rinv @ steering_vect) # compute weights
        Y_mvdr[ff, tt] = np.dot(np.conj(w).flatten(), spectrogram_array[:, ff, tt]) # compute MVDR beamformer
        
# section is for checking inverse covariance diagonal
# plt.imshow(
#     np.abs(Rinv),
#     aspect='auto',
#     origin='lower',
# )

# plt.title(r'$R^{-1}$ Diagonal Matrix' + f", Window Length={window_len+1}" + ", Loading=0.1")

# # CONVENTIONAL BEAMFORMER
for tt in range(len(time_bins)):
    for ff in range(len(freq_bins)):
        steering_vect = np.exp(-1j * 2 * np.pi* freq_bins[ff] * tau[:,tt])
        Y[ff,tt] = np.dot(np.conj(steering_vect), spectrogram_array[:,ff,tt]) / mic_num #normalise the CBF


Y_psd = np.abs(Y)**2 # convert magnitude into power
Y_psd[1:-1] *= 2     # convert into PSD
Y_dB = 10*np.log10(Y_psd + 1e-12) # log scale

Y_mvdr_psd = np.abs(Y_mvdr)**2 # convert magnitude into power
Y_mvdr_psd[1:-1] *= 2 # convert into PSD
Y_mvdr_dB = 10*np.log10(Y_mvdr_psd + 1e-12) # log scale

#Check Graph - Beamformer & MVDR Output
spectrogram_avg_bf = np.mean(np.abs(Y)**2, axis=1) # time average
spectrogram_avg_bf[1:-1] *= 2 # convert into PSD

spectrogram_mvdr_avg_bf = np.mean(np.abs(Y_mvdr)**2, axis=1) # time average
spectrogram_mvdr_avg_bf[1:-1] *= 2 # convert int PSD
###############################################################################
# GRAPH 1
plt.plot(f, 10*np.log10(Pxx), label="Channel " + str(selected_mic) + " Hydrophone")
plt.plot(freq_bins, 10*np.log10(spectrogram_avg_bf + 1e-12), label="Beamformer Output")
plt.plot(freq_bins, 10*np.log10(spectrogram_mvdr_avg_bf + 1e-12), label="MVDR Beamformer Output")
plt.xlim(0,400)
plt.ylim(-70,10)
plt.title("Beam Focused on Target 1(moving), NFFT=" + str(nfft) + ", Overlap=50%")
plt.ylabel('Time Averaged PSD (dBFS/Hz)')
plt.xlabel("Frequency (Hz)")
plt.legend()
plt.show()
###############################################################################
Y_validate = np.zeros((len(freq_bins),len(time_bins)), dtype=np.complex128) # create array
Y_mvdr_validate = np.zeros((len(freq_bins),len(time_bins)), dtype=np.complex128) # create array
sd = tau.T # transpose time delay array for the validation tool

for tt in range(len(time_bins)):
    for ff in range(len(freq_bins)):
        
        X = spectrogram_array[:, ff, tt:tt+1]
        tau_new = sd[tt,:].reshape(1,-1) # change format for validation tool to work
        
        Y_temp = bf.bartlett( X,freq_bins[ff],tau_new,complex_output=True) # perform CBF computation
        Y_validate[ff,tt] = np.asarray(Y_temp).item() / np.sqrt(mic_num)   # adjust magnitude

spectrogram_avg_bf_validate = np.mean(np.abs(Y_validate)**2, axis=1) # time average 
spectrogram_avg_bf_validate[1:-1] *= 2 # convert into PSD

for tt in range(len(time_bins)):
    for ff in range(len(freq_bins)):
        
        start_idx = max(0, tt - window_len)
        X = spectrogram_array[:, ff,start_idx:tt+1]
        tau_new = sd[tt,:].reshape(1,-1)
        
        Y_temp = bf.capon( X,freq_bins[ff],tau_new,complex_output=True) # perform MVDR beamformer validation
        Y_mvdr_validate[ff,tt] = np.asarray(Y_temp).flatten()[-1] / np.sqrt(mic_num) # adjust magnitude

spectrogram_mvdr_avg_bf_validate = np.mean(np.abs(Y_validate)**2, axis=1) # time average
spectrogram_mvdr_avg_bf_validate[1:-1] *= 2 # convert into PSD

# GRAPH 1
plt.plot(f, 10*np.log10(Pxx), label="Channel " + str(selected_mic) + " Hydrophone")
plt.plot(freq_bins, 10*np.log10(spectrogram_avg_bf + 1e-12), label="Beamformer Output")
plt.plot(freq_bins, 10*np.log10(spectrogram_avg_bf_validate + 1e-12), label="Beamformer Output Validation")
plt.plot(freq_bins, 10*np.log10(spectrogram_mvdr_avg_bf + 1e-12), label="MVDR Beamformer Output")
plt.plot(freq_bins, 10*np.log10(spectrogram_mvdr_avg_bf + 1e-12), label="MVDR Beamformer Output Validation")
plt.xlim(0,400)
plt.title("Beam Focused on Target 1(moving), NFFT=" + str(nfft) + ", Overlap=50%")
plt.ylabel('Time Averaged PSD (dBFS/Hz)')
plt.xlabel("Frequency (Hz)")
plt.legend()
plt.show()

# compute mean absolute error
error_bf = 10*np.log10(spectrogram_avg_bf + 1e-12) - 10*np.log10(spectrogram_avg_bf_validate + 1e-12)
error_bf = np.mean(np.abs(error_bf))
print(error_bf,"dB")

error_bf_mvdr = 10*np.log10(spectrogram_mvdr_avg_bf + 1e-12) - 10*np.log10(spectrogram_mvdr_avg_bf_validate + 1e-12)
error_bf_mvdr = np.mean(np.abs(error_bf_mvdr))
print(error_bf_mvdr,"dB")

###############################################################################
# GRAPH 2
fig = plt.figure(figsize=(25, 5))
specgrid = GridSpec(nrows=1,ncols=3,figure=fig)
#------------------------------------------------------------------------------
# HYDROPHONE
ax1 = fig.add_subplot(specgrid[0])
ax1.imshow(
    10*np.log10(spectrogram_psd[selected_mic,:,:]),
    aspect='auto',
    origin='lower',
    # vmin = np.percentile(10*np.log10(spectrogram_psd[selected_mic,:,:]),5),
    # vmax = np.percentile(10*np.log10(spectrogram_psd[selected_mic,:,:]),95),
    extent=[time_bins[0], time_bins[-1],freq_bins[0], freq_bins[-1]]
)

ax1.set_ylim(0,400)
ax1.set_title("Channel " + str(selected_mic) + " Hydrophone")
ax1.set_xlabel("Elapsed Time (s)")
ax1.set_ylabel("Frequency (Hz)")
#------------------------------------------------------------------------------
# BF
ax2 = fig.add_subplot(specgrid[1])
ax2.imshow(
    Y_dB,
    aspect='auto',
    origin='lower',
    # vmin = np.percentile(Y_dB,5),
    # vmax = np.percentile(Y_dB,95),
    extent=[time_bins[0], time_bins[-1],freq_bins[0], freq_bins[-1]]
)

ax2.set_ylim(0,400)
ax2.set_title("Conventional Beamformer Output")
ax2.set_xlabel("Elapsed Time (s)")
ax2.set_ylabel("Frequency (Hz)")
#------------------------------------------------------------------------------
# MVDR BF
ax3 = fig.add_subplot(specgrid[2])
ax3.imshow(
    Y_mvdr_dB,
    aspect='auto',
    origin='lower',
    # vmin = np.percentile(Y_dB,5),
    # vmax = np.percentile(Y_dB,95),
    extent=[time_bins[0], time_bins[-1],freq_bins[0], freq_bins[-1]]
)

ax3.set_ylim(0,500)
ax3.set_title("MVDR Beamformer Output")
ax3.set_xlabel("Elapsed Time (s)")
ax3.set_ylabel("Frequency (Hz)")

plt.show()
###############################################################################