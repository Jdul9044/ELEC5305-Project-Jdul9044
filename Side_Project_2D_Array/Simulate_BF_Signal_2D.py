import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft

# Setup ########################################
fs = 48e3 #Hz
c = 1500 #m/s
mic_num_1D = 4
mic_num_2D = mic_num_1D**2
spacing = 2 #m -> 400Hz max 
duration = 30 #seconds
noise_level = 0.1 # amplitude
azi_deg = 60 #deg
ele_deg = 0 #deg
################################################
#MAKE 2D
mic_pos = np.zeros((mic_num_2D,3))
mic_pos_1D = np.arange(mic_num_1D) * spacing

for i in range(mic_num_1D):
# for i in range(1,6):
    mic_pos[mic_num_1D*i:mic_num_1D*(i+1),0] = mic_pos_1D[i]
    mic_pos[mic_num_1D*i:mic_num_1D*(i+1),1] = mic_pos_1D
################################################
azi = np.deg2rad(azi_deg) #rad
ele = np.deg2rad(ele_deg) #rad

N = int(fs * duration)
n = np.arange(N)
t = n/fs

# Unit vector
u = np.array([
    np.cos(ele) * np.sin(azi),
    np.cos(ele) * np.cos(azi),
    np.sin(ele)
    ])

tau = np.dot(mic_pos, u) / c

# Broadband
freq_range = 200,350
broadband_signal = np.fft.rfft(np.random.rand(N))
freqs = np.fft.rfftfreq(N, d=1/fs)
freq_band = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
broadband_signal[~freq_band] = 0



audio = np.zeros((N, mic_num_2D))
    
for m in range(mic_num_2D):
    
    audio[:, m] = (
        0.1 * np.sin(2 * np.pi * 250 * (t + tau[m]))
    ) #+ np.fft.irfft(broadband_signal * np.exp(-1j * 2 * np.pi * freqs * tau[m]), n=N)
    

azi_deg = 10 #deg
ele_deg = 0 #deg
azi = np.deg2rad(azi_deg) #rad
ele = np.deg2rad(ele_deg) #rad

u = np.array([
    np.cos(ele) * np.sin(azi),
    np.cos(ele) * np.cos(azi),
    np.sin(ele)
    ])

tau = np.dot(mic_pos, u) / c

for m in range(mic_num_2D):
    
    audio[:, m] += 0.1 * np.sin(2 * np.pi * 300 * (t + tau[m])) + np.fft.irfft(broadband_signal * np.exp(1j * 2 * np.pi * freqs * tau[m]), n=N)
    

audio += (noise_level*np.random.randn(N, mic_num_2D))
audio = audio / np.max(np.abs(audio))

for i in range(mic_num_2D):
    audio_channel = audio[:,i]
    audio_pcm = np.int16(audio_channel * (2**15-1))
    audio_pcm.tofile("hydrophone" + str(i) + ".pcm")
    
