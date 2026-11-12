import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft

# Setup ########################################
fs = 48e3 #Hz
c = 1500 #m/s
mic_num = 8
spacing = 1.875 #m -> 400Hz max 
duration = 30 #seconds
noise_level = 0.1 # amplitude
mic_pos = np.zeros((mic_num,3)) # create 3D array
mic_pos[:, 1] = np.arange(mic_num) * spacing # create only 1D array
################################################
N = int(fs * duration) # number of samples
n = np.arange(N) # sample
t = n/fs # time sample

#MOVING
ele_deg = np.zeros(len(t)) # 0 elevation
azi_deg = np.linspace(60,90,N) # azimuth shifts from 60 to 90 degrees
azi = np.deg2rad(azi_deg) #rad
ele = np.deg2rad(ele_deg) #rad

# Unit vector
u = np.array([
    np.cos(ele) * np.sin(azi),
    np.cos(ele) * np.cos(azi),
    np.sin(ele)
    ])

# Time delay
tau = -np.dot(mic_pos, u) / c

# Broadband
freq_range = 200,350 # broadband range of synthetic
broadband_signal = np.fft.rfft(np.random.rand(N)) # random noise
freqs = np.fft.rfftfreq(N, d=1/fs) # create frequency
freq_band = (freqs >= freq_range[0]) & (freqs <= freq_range[1]) # select frequency band
broadband_signal[~freq_band] = 0 # set everything to 0 if not in band



audio = np.zeros((N, mic_num)) # create signal
    
# TARGET 1
for m in range(mic_num):
    
    # generate only tone
    audio[:, m] = (
        0.1 * np.sin(2 * np.pi * 250 * (t - tau[m]))
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

tau = -np.dot(mic_pos, u) / c

# TARGET 2
for m in range(mic_num):
    
    # generate tone and broadband
    audio[:, m] += 0.1 * np.sin(2 * np.pi * 300 * (t - tau[m])) + np.fft.irfft(broadband_signal * np.exp(-1j * 2 * np.pi * freqs * tau[m]), n=N)

audio += (noise_level * np.random.randn(N, mic_num))
audio = audio / np.max(np.abs(audio))

for i in range(mic_num):
    audio_channel = audio[:,i]
    audio_pcm = np.int16(audio_channel * (2**15-1))
    audio_pcm.tofile("hydrophone" + str(i) + ".pcm")
    
