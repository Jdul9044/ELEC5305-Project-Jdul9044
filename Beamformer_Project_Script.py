#BF Main

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import stft, istft, spectrogram
from pathlib import Path
from scipy.signal import welch, decimate
from scipy.io import wavfile, loadmat
import pandas as pd
# Setup ########################################
c = 1500 #m/s
nfft = 1028
hop = round(nfft/2)
simulation_mode = False
################################################################################################
if simulation_mode == True:
    fs = 48e3
    
    folder_path = r"C:\Users\John Conrad\Desktop\ELEC5305\MainProject\Simulation_Data\Stationary"
    folder = Path(folder_path)
    
    # pcm_files = list(folder.glob("*.pcm"))
    signal_list = []
    
    for file in folder.glob("*.pcm"):
        print(file)
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
###############################################################################################
else:
    fs = 3276.8 #HLA
    # fs = 1500 #VLA
    
    # file_path = "Real_Data\VLA_100000_SAMPLES.mat"
    file_path = "Real_Data\HLA_100000_SAMPLES.mat"
    signal_array = loadmat(file_path)['ans'].T
    # signal_array = signal_array[::-1,:] # FOR VLA ONLY
    mic_num = signal_array.shape[0]
    
    # mic_pos = pd.read_excel("Real_Data\Hyd_Pos_VLA.xlsx",header=None).to_numpy()
    mic_pos = pd.read_excel("Real_Data\Hyd_Pos_HLA.xlsx",header=None).to_numpy()
    mic_pos = mic_pos[:,1:4]
    
    # select hydrophones
    signal_array = signal_array[0:5,:]
    mic_pos = mic_pos[0:5,:]
################################################################################################
signal_array = signal_array / np.max(np.abs(signal_array))

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
selected_mic = 2

#Check Single Channel
# plt.imshow(
#     10*np.log10(spectrogram_psd[selected_mic,:,:]),
#     aspect='auto',
#     origin='lower',
#     vmin = np.percentile(10*np.log10(spectrogram_psd[selected_mic,:,:]),5),
#     vmax = np.percentile(10*np.log10(spectrogram_psd[selected_mic,:,:]),95),
#     extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]]
# )

# plt.title("Channel " + str(selected_mic) + " Hydrophone")
# plt.xlabel("Elapsed Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.colorbar()
# plt.ylim(0,500)
# plt.xlim(0,10)
# plt.ylim(100,300)
# plt.show()

#Check Graph - for single microphone/hydrophone
f, Pxx = welch(signal_array[selected_mic,:], fs=fs, window=selected_window,
               nperseg=nfft, noverlap=nfft-hop, nfft=nfft, detrend=False)

# plt.plot(f, 10*np.log10(Pxx))
# plt.xlim(0,500)
# # plt.ylim(-20,20)
# plt.title("Channel " + str(selected_mic) + " Hydrophone, NFFT=" + str(nfft) + ", Overlap=50%")
# plt.ylabel('PSD (dBFS/Hz)')
# plt.xlabel("Frequency (Hz)")
# plt.show()



# Beamforming Stage
# mic_pos[:,1] = -(mic_pos[:,1] - np.mean(mic_pos[:,1], axis=0))
# mic_pos[:,0] = -(mic_pos[:,0] - np.mean(mic_pos[:,0], axis=0))
# Array Setup ##################################
c = 1500 #m/s
################################################
# Target Information ###########################
# azi_deg = 309 #deg
# azi_deg = 171 #deg
# azi_deg = 60
azi_deg = 60
ele_deg = 0 #deg
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
window_len = 80

# MVDR
for ff in range(len(freq_bins)):
    steering_vect = np.exp(-1j * 2 * np.pi * freq_bins[ff] * tau)
    steering_vect = steering_vect.reshape(-1, 1)

    for tt in range(len(time_bins)):
        start_idx = max(0, tt - window_len)
        end_idx = min(len(time_bins), tt + window_len)
        
        X_f = spectrogram_array[:, ff, start_idx:end_idx]
        R = (X_f @ X_f.conj().T) / X_f.shape[1]
        
        loading = 0.1 * np.trace(R).real / mic_num
        R_loaded = R + loading * np.eye(mic_num)
        Rinv = np.linalg.inv(R_loaded)
        
        w = (Rinv @ steering_vect)/(steering_vect.conj().T @ Rinv @ steering_vect)                
        Y_mvdr[ff, tt] = np.dot(np.conj(w).flatten(), spectrogram_array[:, ff, tt])
        
# plt.imshow(
#     np.abs(Rinv),
#     aspect='auto',
#     origin='lower',
# )
        

# CONVENTIONAL BEAMFORMER
for tt in range(len(time_bins)):
    for ff in range(len(freq_bins)):
        steering_vect = np.exp(-1j * 2 * np.pi* freq_bins[ff] * tau)
        Y[ff,tt] = np.dot(np.conj(steering_vect), spectrogram_array[:,ff,tt]) / mic_num

Y_psd = np.abs(Y)**2
Y_psd[1:-1] *= 2
Y_dB = 10*np.log10(Y_psd + 1e-12)

Y_mvdr_psd = np.abs(Y_mvdr)**2
Y_mvdr_psd[1:-1] *= 2
Y_mvdr_dB = 10*np.log10(Y_mvdr_psd + 1e-12)

#Check Beamformer Output
# plt.imshow(
#     Y_dB,
#     aspect='auto',
#     origin='lower',
#     vmin = np.percentile(Y_dB,5),
#     vmax = np.percentile(Y_dB,95),
#     extent=[time_bins[0], time_bins[-1], freq_bins[0], freq_bins[-1]]
# )

# plt.ylim(0,500)
# plt.title("Beamformer Output")
# plt.xlabel("Elapsed Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.colorbar()
# plt.ylim(0,400)
# plt.xlim(0,10)
# plt.ylim(100,300)
# plt.show()

#Check Graph - Beamformer & MVDR Output
spectrogram_avg_bf = np.mean(np.abs(Y[:,:])**2, axis=1)
spectrogram_mvdr_avg_bf = np.mean(np.abs(Y_mvdr[:,:])**2, axis=1)
spectrogram_avg_bf[1:-1] *= 2 
spectrogram_mvdr_avg_bf[1:-1] *= 2 

# plt.plot(freq_bins, 10*np.log10(spectrogram_avg_bf + 1e-12))
# plt.xlim(0,500)
# # plt.ylim(-20,20)
# plt.title("Beamformed Output, NFFT=" + str(nfft) + ", Overlap=50%")
# plt.ylabel('PSD (dBFS/Hz)')
# plt.xlabel("Frequency (Hz)")
# plt.show()

###############################################################################
# GRAPH 1
plt.figure()
plt.plot(freq_bins, 10*np.log10(Pxx), label="Channel " + str(selected_mic) + " Hydrophone")
plt.plot(freq_bins, 10*np.log10(spectrogram_avg_bf + 1e-12), label="Beamformer Output")
plt.plot(freq_bins, 10*np.log10(spectrogram_mvdr_avg_bf + 1e-12), label="MVDR Beamformer Output")
plt.xlim(0,400)
# plt.xlim(100,120)
# plt.ylim(-20,20)
plt.title(f"Beam Focused at a={azi_deg} and e={ele_deg}, NFFT=" + str(nfft) + ", Overlap=50%")
plt.ylabel('Time Averaged PSD (dBFS/Hz)')
plt.xlabel("Frequency (Hz)")
plt.legend()
plt.show()
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
# Azimuth Scan
azi_scan_deg = np.arange(0, 360, 1)
ele_scan_deg = np.zeros_like(azi_scan_deg)
azi = np.deg2rad(azi_scan_deg) #rad
ele = np.deg2rad(ele_scan_deg) #rad

selected_freq_bin = 112
freq_bin_idx = np.argmin(np.abs(freq_bins - selected_freq_bin))

Y_freq = np.zeros((len(time_bins),len(azi)), dtype=complex)
Y_freq_mvdr = np.zeros((len(time_bins),len(azi)), dtype=complex)
for aa in range(len(azi)):
    
        u = np.array([
            np.cos(ele[aa]) * np.sin(azi[aa]),
            np.cos(ele[aa]) * np.cos(azi[aa]),
            np.sin(ele[aa])
            ])
    
        tau = -np.dot(mic_pos,u) / c
    
        steering_vect = np.exp(-1j * 2 * np.pi* freq_bins[freq_bin_idx] * tau)
        Y_freq[:,aa] = np.dot(np.conj(steering_vect), spectrogram_array[:,freq_bin_idx,:]) / mic_num
        
        X_f = spectrogram_array[:, freq_bin_idx, :]
        R = (X_f @ X_f.conj().T) / X_f.shape[1]
        
        loading = 0.1 * np.trace(R).real / mic_num
        # loading = 1e-3 * np.trace(R).real / mic_num
        R_loaded = R + loading * np.eye(mic_num)
        Rinv = np.linalg.inv(R_loaded)
        
        w = (Rinv @ steering_vect)/(steering_vect.conj().T @ Rinv @ steering_vect)                
        Y_freq_mvdr[:,aa] = np.dot(np.conj(w).flatten(), spectrogram_array[:, freq_bin_idx, :])


Y_freq = np.abs(Y_freq)**2
spectrogram_avg_azi = np.mean(Y_freq, axis=0)
spectrogram_avg_azi_dB = 10*np.log10(spectrogram_avg_azi + 1e-12)

Y_freq_mvdr = np.abs(Y_freq_mvdr)**2
spectrogram_avg_azi = np.mean(Y_freq_mvdr, axis=0)
spectrogram_avg_azi_mvdr_dB = 10*np.log10(spectrogram_avg_azi + 1e-12)

plt.figure()
plt.plot(azi_scan_deg,spectrogram_avg_azi_dB,label="CBF")
plt.plot(azi_scan_deg,spectrogram_avg_azi_mvdr_dB,label="MVDR")
plt.title(f"Azimuth Scan for Frequency {selected_freq_bin} Hz")
plt.xlabel("Azimuth (degrees)")
plt.ylabel("Magnitude (dB)")
plt.legend()

# peak_idx = np.argmax(spectrogram_avg_azi_dB)
# peak_idx = np.argmax(spectrogram_avg_azi_dB[200:])
peak_idx = np.argmax(spectrogram_avg_azi_mvdr_dB)
# peak_idx = np.argmax(spectrogram_avg_azi_mvdr_dB[200:])
peak_angle = azi_scan_deg[peak_idx]
# peak_angle = azi_scan_deg[peak_idx+200]

print(peak_angle)
###############################################################################
###############################################################################
# # Azimuth Scan (Moving Target)
# # azi_scan_deg = np.arange(0, 360, 1)
# azi_scan_deg = np.arange(0, 360, 0.1)
# ele_scan_deg = np.zeros_like(azi_scan_deg)
# azi = np.deg2rad(azi_scan_deg) #rad
# ele = np.deg2rad(ele_scan_deg) #rad

# time_start = 63
# # time_end = 63 #10 seconds
# # time_end = 127 #20 seconds
# # time_end = 191 #30 seconds
# time_end = 95

# selected_freq_bin = 112
# freq_bin_idx = np.argmin(np.abs(freq_bins - selected_freq_bin))

# # Y_freq = np.zeros((len(time_bins),len(azi)), dtype=complex)
# # Y_freq_mvdr = np.zeros((len(time_bins),len(azi)), dtype=complex)

# Y_freq = np.zeros((time_end-time_start,len(azi)), dtype=complex)
# Y_freq_mvdr = np.zeros((time_end-time_start,len(azi)), dtype=complex)
# for aa in range(len(azi)):
    
#         u = np.array([
#             np.cos(ele[aa]) * np.sin(azi[aa]),
#             np.cos(ele[aa]) * np.cos(azi[aa]),
#             np.sin(ele[aa])
#             ])
    
#         tau = -np.dot(mic_pos,u) / c
    
#         steering_vect = np.exp(-1j * 2 * np.pi* freq_bins[freq_bin_idx] * tau)
#         Y_freq[:,aa] = np.dot(np.conj(steering_vect), spectrogram_array[:,freq_bin_idx,time_start:time_end]) / mic_num
        
#         X_f = spectrogram_array[:, freq_bin_idx, time_start:time_end]
#         R = (X_f @ X_f.conj().T) / X_f.shape[1]
        
#         loading = 0.1 * np.trace(R).real / mic_num
#         # loading = 1e-3 * np.trace(R).real / mic_num
#         R_loaded = R + loading * np.eye(mic_num)
#         Rinv = np.linalg.inv(R_loaded)
        
#         w = (Rinv @ steering_vect)/(steering_vect.conj().T @ Rinv @ steering_vect)                
#         Y_freq_mvdr[:,aa] = np.dot(np.conj(w).flatten(), spectrogram_array[:, freq_bin_idx, time_start:time_end])


# Y_freq = np.abs(Y_freq)**2
# spectrogram_avg_azi = np.mean(Y_freq, axis=0)
# spectrogram_avg_azi_dB = 10*np.log10(spectrogram_avg_azi + 1e-12)

# Y_freq_mvdr = np.abs(Y_freq_mvdr)**2
# spectrogram_avg_azi = np.mean(Y_freq_mvdr, axis=0)
# spectrogram_avg_azi_mvdr_dB = 10*np.log10(spectrogram_avg_azi + 1e-12)

# plt.figure()
# plt.plot(azi_scan_deg,spectrogram_avg_azi_dB,label="CBF")
# plt.plot(azi_scan_deg,spectrogram_avg_azi_mvdr_dB,label="MVDR")
# plt.title(f"Azimuth Scan for Frequency {selected_freq_bin} Hz")
# plt.xlabel("Azimuth (degrees)")
# plt.ylabel("Magnitude (dB)")
# plt.legend()

# # peak_idx = np.argmax(spectrogram_avg_azi_dB)
# # peak_idx = np.argmax(spectrogram_avg_azi_dB[200:])
# # peak_idx = np.argmax(spectrogram_avg_azi_mvdr_dB)
# peak_idx = np.argmax(spectrogram_avg_azi_mvdr_dB[:2000])
# # peak_idx = np.argmax(spectrogram_avg_azi_mvdr_dB[2000:])
# peak_angle = azi_scan_deg[peak_idx]
# # peak_angle = azi_scan_deg[peak_idx+2000]

# print(peak_angle)
###############################################################################
d = np.linalg.norm(mic_pos[:,None,:] - mic_pos[None,:,:], axis=2)
###############################################################################
# WNG
azi_scan_deg = np.arange(0, 360, 1)
ele_scan_deg = np.zeros_like(azi_scan_deg)
azi = np.deg2rad(azi_scan_deg) #rad
ele = np.deg2rad(ele_scan_deg) #rad

# azi_target_deg = 316.7
azi_target_deg = azi_deg
ele_target_deg = 0
azi_target = np.deg2rad(azi_target_deg) #rad
ele_target = np.deg2rad(ele_target_deg) #rad

WNG = np.zeros(len(freq_bins))
WNG_mvdr = np.zeros(len(freq_bins))

u_target = np.array([
    np.cos(ele_target) * np.sin(azi_target),
    np.cos(ele_target) * np.cos(azi_target),
    np.sin(ele_target)
    ])

tau_target = -np.dot(mic_pos,u_target) / c

for ff in range(len(freq_bins)):
    
    steering_vect_target = np.exp(-1j * 2 * np.pi* freq_bins[ff] * tau_target)
    WNG[ff] = 1 / np.real((steering_vect_target / mic_num).conj().T @ (steering_vect_target / mic_num)).item()

    R = np.sinc(2 * freq_bins[ff] * d / c)
    loading = 0.1 * np.trace(R).real / mic_num
    R_loaded = R + loading * np.eye(mic_num)
    Rinv = np.linalg.inv(R_loaded)
    w = (Rinv @ steering_vect_target)/(steering_vect_target.conj().T @ Rinv @ steering_vect_target)     
    WNG_mvdr[ff] = 1 / np.real(w.conj().T @ w).item()


WNG = np.abs(WNG)
WNG_dB = 10*np.log10(WNG + 1e-12)

WNG_mvdr = np.abs(WNG_mvdr)
WNG_mvdr_dB = 10*np.log10(WNG_mvdr + 1e-12)

plt.figure()
plt.plot(freq_bins,WNG_dB,label="CBF")
plt.plot(freq_bins,WNG_mvdr_dB,label="MVDR")
plt.title("White Noise Gain Comparison (Main Source)")
plt.xlim(0,400)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend()
###############################################################################
###############################################################################
# DI
azi_scan_deg = np.arange(0, 360, 1)
ele_scan_deg = np.zeros_like(azi_scan_deg)
azi = np.deg2rad(azi_scan_deg) #rad
ele = np.deg2rad(ele_scan_deg) #rad

# azi_target_deg = 316.7
azi_target_deg = azi_deg
ele_target_deg = 0
azi_target = np.deg2rad(azi_target_deg) #rad
ele_target = np.deg2rad(ele_target_deg) #rad

Y_freq = np.zeros((len(freq_bins),len(azi)), dtype=complex)
Y_freq_mvdr = np.zeros((len(freq_bins),len(azi)), dtype=complex)

Y_target = np.zeros(len(freq_bins), dtype=complex)
Y_target_mvdr = np.zeros(len(freq_bins), dtype=complex)

u_target = np.array([
    np.cos(ele_target) * np.sin(azi_target),
    np.cos(ele_target) * np.cos(azi_target),
    np.sin(ele_target)
    ])

tau_target = -np.dot(mic_pos,u_target) / c

for ff in range(len(freq_bins)):
    
    steering_vect_target = np.exp(-1j * 2 * np.pi* freq_bins[ff] * tau_target)

    R = np.sinc(2 * freq_bins[ff] * d / c)
    loading = 0.1 * np.trace(R).real / mic_num
    R_loaded = R + loading * np.eye(mic_num)
    Rinv = np.linalg.inv(R_loaded)
    w = (Rinv @ steering_vect_target)/(steering_vect_target.conj().T @ Rinv @ steering_vect_target)     

    Y_target[ff] = np.dot(np.conj(steering_vect_target), steering_vect_target) / mic_num
    Y_target_mvdr[ff] = np.dot(np.conj(w), steering_vect_target)

    for aa in range(len(azi)):
        
        u = np.array([
            np.cos(ele[aa]) * np.sin(azi[aa]),
            np.cos(ele[aa]) * np.cos(azi[aa]),
            np.sin(ele[aa])
            ])
    
        tau = -np.dot(mic_pos,u) / c
    
        steering_vect = np.exp(-1j * 2 * np.pi* freq_bins[ff] * tau)
        Y_freq[ff,aa] = np.dot(np.conj(steering_vect), steering_vect_target) / mic_num
        Y_freq_mvdr[ff,aa] = np.dot(np.conj(w), steering_vect)
    
Y_target = np.abs(Y_target)**2
Y_target_mvdr = np.abs(Y_target_mvdr)**2

Y_freq = np.abs(Y_freq)**2
Y_freq_mvdr = np.abs(Y_freq_mvdr)**2

DI_freq = 10 * np.log10(Y_target / np.mean(Y_freq, axis=1))
DI_freq_mvdr = 10 * np.log10(Y_target_mvdr / np.mean(Y_freq_mvdr, axis=1))

plt.figure()
plt.plot(freq_bins,DI_freq,label="CBF")
plt.plot(freq_bins,DI_freq_mvdr,label="MVDR")
plt.title("Directivity Index (Main Source)")
plt.xlabel("Frequency (Hz)")
plt.xlim(0,400)
plt.ylabel("DI (dB)")
plt.legend()
###############################################################################
###############################################################################
# BEAMPATTERN
azi_scan_deg = np.arange(0, 360, 1)
ele_scan_deg = np.zeros_like(azi_scan_deg)
azi = np.deg2rad(azi_scan_deg) #rad
ele = np.deg2rad(ele_scan_deg) #rad


freq_list = [10,100,200,300]

Y_freq = np.zeros((len(freq_list),len(azi)), dtype=complex)
Y_freq_mvdr = np.zeros((len(freq_list),len(azi)), dtype=complex)

#ENDFIRE
u_beam = np.array([
    np.cos(0) * np.sin(0),
    np.cos(0) * np.cos(0),
    np.sin(0)
    ])

# u_beam = np.array([
#     np.cos(0) * np.sin(np.radians(90)),
#     np.cos(0) * np.cos(np.radians(90)),
#     np.sin(0)
#     ])

tau_beam = -np.dot(mic_pos,u_beam) / c

for ff, freq in enumerate(freq_list):
    
    steering_vect_target = np.exp(-1j * 2 * np.pi* freq * tau_beam)
    freq_index = np.argmin(np.abs(freq_bins-freq))
    R = np.sinc(2 * freq * d / c)
    loading = 0.1 * np.trace(R).real / mic_num
    R_loaded = R + loading * np.eye(mic_num)
    Rinv = np.linalg.inv(R_loaded)
    w = (Rinv @ steering_vect_target)/(steering_vect_target.conj().T @ Rinv @ steering_vect_target)                

    for aa in range(len(azi)):
        
        u = np.array([
            np.cos(ele[aa]) * np.sin(azi[aa]),
            np.cos(ele[aa]) * np.cos(azi[aa]),
            np.sin(ele[aa])
            ])
    
        tau = -np.dot(mic_pos,u) / c
    
        steering_vect = np.exp(-1j * 2 * np.pi* freq * tau)
        Y_freq[ff,aa] = np.dot(np.conj(steering_vect), steering_vect_target) / mic_num
        Y_freq_mvdr[ff,aa] = np.dot(np.conj(w), steering_vect)
    
Y_freq = np.abs(Y_freq)**2
Y_freq = Y_freq / np.max(Y_freq, axis=1, keepdims=True)
Y_azi_dB = 10*np.log10(Y_freq + 1e-12)

Y_freq_mvdr = np.abs(Y_freq_mvdr)**2
Y_freq_mvdr = Y_freq_mvdr / np.max(Y_freq_mvdr, axis=1, keepdims=True)
Y_azi_mvdr_dB = 10*np.log10(Y_freq_mvdr + 1e-12)

fig = plt.figure(figsize=(25, 5))
specgrid = GridSpec(nrows=1,ncols=len(freq_list),figure=fig)
#------------------------------------------------------------------------------
# BEAM PATTERN COMPARISON PLOT
for i in range(len(freq_list)):
    ax = fig.add_subplot(specgrid[i], projection='polar')
    ax.plot(azi,Y_azi_dB[i,:],label="CBF")
    ax.plot(azi,Y_azi_mvdr_dB[i,:],label="MVDR")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Azimuth Scan for Frequency {freq_list[i]} Hz")
    ax.set_xlabel("Azimuth (degrees)")
    ax.set_ylabel("Magnitude (dB)")
    ax.legend()